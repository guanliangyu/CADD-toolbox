#!/usr/bin/env python3
"""
后台构象生成工具 - 独立于Streamlit运行
解决页面刷新中断问题，支持大规模分子库处理
"""

import os
import sys
import time
import json
import signal
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

def create_background_conformer_script(
    input_file,
    output_dir,
    num_conformers=10,
    max_attempts=100,
    random_seed=42,
    num_workers=None,
    processing_limit=100000,
    file_type="sdf",
    smiles_column="SMILES",
    script_name=None
):
    """创建后台构象生成脚本"""
    
    if num_workers is None:
        import multiprocessing
        num_workers = min(64, multiprocessing.cpu_count() * 2)
    
    if script_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = f"background_conformer_{timestamp}"
    
    # 处理processing_limit参数
    if processing_limit == float('inf'):
        processing_limit_str = "float('inf')"
    else:
        processing_limit_str = str(int(processing_limit))
    
    script_content = f'''#!/usr/bin/env python3
"""
后台分子3D构象生成脚本 - 独立运行
生成时间: {datetime.now().isoformat()}
"""

import os
import sys
import time
import pickle
import logging
import signal
import psutil
import gc
import threading
from datetime import datetime
from multiprocessing import Pool, cpu_count
import multiprocessing
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import traceback

# 信号处理
def signal_handler(signum, frame):
    logger.info(f"收到信号 {{signum}}，正在安全退出...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 添加项目路径
sys.path.append('/home/liangyu/CADD-toolbox')

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError as e:
    print(f"错误: 无法导入RDKit: {{e}}")
    print("请确保已激活正确的conda环境")
    sys.exit(1)

# 配置参数
CONFIG = {{
    'input_file': '{input_file}',
    'output_dir': '{output_dir}',
    'num_conformers': {num_conformers},
    'max_attempts': {max_attempts},
    'random_seed': {random_seed},
    'num_workers': {num_workers},
    'processing_limit': {processing_limit_str},
    'file_type': '{file_type}',
    'smiles_column': '{smiles_column}',
    'checkpoint_interval': 1000,  # 每1000个分子保存一次检查点
    'batch_size': 1000  # 分批处理大小
}}

# 设置日志
log_file = os.path.join(CONFIG['output_dir'], '{script_name}.log')
logging.basicConfig(
    level=logging.INFO,  # 改为INFO级别，减少冗余debug信息
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, 'w'),
        logging.StreamHandler()
    ]
)

# 设置进度文件
progress_file = os.path.join(CONFIG['output_dir'], '{script_name}.progress')

def update_progress(current, total, phase="processing", extra_info=""):
    """更新进度信息到文件"""
    try:
        progress_data = {{
            'current': current,
            'total': total,
            'percentage': (current / total * 100) if total > 0 else 0,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            'extra_info': extra_info
        }}
        
        import json
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"更新进度失败: {{e}}")

logger = logging.getLogger(__name__)

# 全局计数器和锁
task_counter_lock = threading.Lock()
completed_tasks = 0
failed_tasks = 0

def monitor_resources():
    """监控系统资源"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        return {{
            'memory_mb': memory_mb,
            'cpu_percent': cpu_percent,
            'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
        }}
    except:
        return {{'memory_mb': 0, 'cpu_percent': 0, 'available_memory_gb': 0}}

def save_checkpoint(results, checkpoint_file, metadata):
    """保存检查点"""
    try:
        checkpoint_data = {{
            'results': results,
            'metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }}
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"检查点已保存: {{checkpoint_file}}")
    except Exception as e:
        logger.warning(f"保存检查点失败: {{e}}")

def load_checkpoint(checkpoint_file):
    """加载检查点"""
    try:
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            logger.info(f"已加载检查点: {{checkpoint_file}}")
            return checkpoint_data
    except Exception as e:
        logger.warning(f"加载检查点失败: {{e}}")
    return None

def generate_conformers_for_mol(args):
    """多进程worker函数 - 为单个分子生成3D构象"""
    global completed_tasks, failed_tasks
    
    mol_data, mol_id, num_confs, max_attempts, random_seed = args
    
    try:
        # 重建分子对象 - 智能判断数据类型
        if isinstance(mol_data, str):
            # 判断是SMILES还是MolBlock
            if "\\n" in mol_data and ("M  END" in mol_data or "V2000" in mol_data):
                # MolBlock格式
                mol = Chem.MolFromMolBlock(mol_data, removeHs=False, sanitize=False)
                if not mol:
                    with task_counter_lock:
                        failed_tasks += 1
                    return (mol_id, None, "MolBlock解析失败")
            else:
                # SMILES字符串
                mol = Chem.MolFromSmiles(mol_data)
                if not mol:
                    with task_counter_lock:
                        failed_tasks += 1
                    return (mol_id, None, "SMILES解析失败")
        else:
            with task_counter_lock:
                failed_tasks += 1
            return (mol_id, None, "未知数据类型")
        
        original_smiles = Chem.MolToSmiles(mol)

        try:
            # 确保分子被正确sanitize
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                # 尝试部分sanitize
                try:
                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except:
                    pass  # 继续尝试处理
            
            # 添加氢原子
            mol_h = Chem.AddHs(mol)
            
            # 设置ETKDG参数
            params = AllChem.ETKDGv3()
            params.randomSeed = random_seed
            if hasattr(params, 'maxIterations'):
                params.maxIterations = max_attempts
            params.numThreads = 0  # 重要：每个进程内部不使用多线程
            
            # 生成构象
            cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params)
            
            if not cids or len(cids) == 0:
                with task_counter_lock:
                    failed_tasks += 1
                return (mol_id, None, f"RDKit构象生成失败")
            
            # 转换为MolBlock以便传输
            mol_block = Chem.MolToMolBlock(mol_h)
            
            with task_counter_lock:
                completed_tasks += 1
            
            return (mol_id, mol_block, f"成功生成{{len(cids)}}个构象")
            
        except Exception as e:
            error_msg = f"构象生成异常: {{str(e)}}"
            logger.warning(f"分子 {{mol_id}} {{error_msg}}")
            with task_counter_lock:
                failed_tasks += 1
            return (mol_id, None, error_msg)
            
    except Exception as e:
        error_msg = f"Worker异常: {{str(e)}}"
        logger.error(f"分子 {{mol_id}} {{error_msg}}")
        with task_counter_lock:
            failed_tasks += 1
        return (mol_id, None, error_msg)

def load_molecules_from_file(file_path, file_type, smiles_column, processing_limit, start_idx=0):
    """从文件加载分子数据，支持断点续传"""
    molecules_data = []
    mol_ids = []
    
    logger.info(f"📂 开始加载分子: {{file_path}}")
    logger.info(f"文件类型: {{file_type}}, 处理限制: {{processing_limit}}, 起始索引: {{start_idx}}")
    
    # 初始化加载进度
    update_progress(0, 1, "loading", "正在分析文件...")
    
    count = 0
    skipped = 0
    load_start_time = time.time()
    
    if file_type == "csv":
        import pandas as pd
        df = pd.read_csv(file_path)
        total_rows = len(df)
        logger.info(f"CSV文件包含 {{total_rows}} 行")
        
        if smiles_column not in df.columns:
            raise ValueError(f"CSV文件中未找到SMILES列 '{{smiles_column}}'")
        
        update_progress(0, total_rows, "loading", f"开始解析{{total_rows}}行CSV数据")
        
        for idx, row in df.iterrows():
            if idx < start_idx:
                skipped += 1
                continue
                
            if processing_limit != float('inf') and count >= processing_limit:
                break
            
            smi = str(row[smiles_column])
            mol = Chem.MolFromSmiles(smi)
            if mol:
                molecules_data.append(smi)
                mol_ids.append(f"row_{{idx+1}}")
                count += 1
                
                # 更新加载进度
                if idx % 5000 == 0:
                    elapsed = time.time() - load_start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    extra_info = f"已加载{{count}}个有效分子, {{rate:.1f}}分子/秒"
                    update_progress(idx, total_rows, "loading", extra_info)
    
    elif file_type in ["txt", "smi"]:
        logger.info(f"📄 开始检查文本文件...")
        # 先快速计算总行数
        with open(file_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        update_progress(0, total_lines, "loading", f"开始解析{{total_lines}}行文本数据")
        
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i < start_idx:
                    skipped += 1
                    continue
                    
                if processing_limit != float('inf') and count >= processing_limit:
                    break
                
                smi = line.strip()
                if smi:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        molecules_data.append(smi)
                        mol_ids.append(f"line_{{i+1}}")
                        count += 1
                        
                        # 更新加载进度
                        if i % 5000 == 0:
                            elapsed = time.time() - load_start_time
                            rate = count / elapsed if elapsed > 0 else 0
                            extra_info = f"已加载{{count}}个有效分子, {{rate:.1f}}分子/秒"
                            update_progress(i, total_lines, "loading", extra_info)
    
    elif file_type == "sdf":
        logger.info(f"🧪 开始检查SDF文件...")
        # SDF文件无法预先计算总数，使用动态进度
        update_progress(0, 1, "loading", "开始解析SDF文件...")
        
        supplier = Chem.ForwardSDMolSupplier(file_path, removeHs=False, sanitize=True)
        for i, mol in enumerate(supplier):
            if i < start_idx:
                skipped += 1
                continue
                
            if processing_limit != float('inf') and count >= processing_limit:
                break
            
            if mol is not None:
                mol_block = Chem.MolToMolBlock(mol)
                molecules_data.append(mol_block)
                mol_id = mol.GetProp('_Name') if mol.HasProp('_Name') else f"sdf_mol_{{i+1}}"
                mol_ids.append(mol_id)
                count += 1
                
                # 更新加载进度
                if count % 1000 == 0:
                    elapsed = time.time() - load_start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    extra_info = f"已加载{{count}}个有效分子, {{rate:.1f}}分子/秒"
                    # SDF使用已处理数量作为进度
                    update_progress(count, max(count + 1000, 10000), "loading", extra_info)
    
    load_elapsed = time.time() - load_start_time
    logger.info(f"✅ 加载完成: {{len(molecules_data)}} 个有效分子 (跳过{{skipped}}个)，耗时: {{load_elapsed:.2f}}秒")
    
    # 完成加载阶段
    update_progress(1, 1, "loading", f"加载完成: {{len(molecules_data)}}个分子")
    
    return molecules_data, mol_ids

def process_batch_with_monitoring(tasks, num_workers, batch_size=1000, processed_offset=0):
    """分批处理任务，增加监控和错误恢复，包含进度条功能"""
    all_results = []
    total_tasks = len(tasks)
    total_overall = total_tasks + processed_offset  # 总体任务数
    batch_count = (total_tasks + batch_size - 1) // batch_size
    
    logger.info(f"将 {{total_tasks}} 个任务分为 {{batch_count}} 批，每批 {{batch_size}} 个")
    
    # 初始化进度
    update_progress(processed_offset, total_overall, "processing", f"开始处理 {{total_tasks}} 个分子")
    
    for batch_idx in range(0, total_tasks, batch_size):
        batch_end = min(batch_idx + batch_size, total_tasks)
        current_batch = tasks[batch_idx:batch_end]
        batch_num = batch_idx // batch_size + 1
        
        logger.info(f"📊 批次 {{batch_num}}/{{batch_count}}: 处理 {{len(current_batch)}} 个分子")
        
        batch_start_time = time.time()
        batch_results = []
        batch_success = 0
        batch_failures = 0
        
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # 提交当前批次的任务
                future_to_task = {{}}
                for i, task in enumerate(current_batch):
                    try:
                        future = executor.submit(generate_conformers_for_mol, task)
                        future_to_task[future] = (batch_idx + i, task[1])  # (全局索引, mol_id)
                    except Exception as e:
                        logger.error(f"提交任务失败: {{e}}")
                        batch_failures += 1
                
                # 处理结果
                completed_in_batch = 0
                last_progress_update = time.time()
                
                for future in concurrent.futures.as_completed(future_to_task, timeout=1800):  # 30分钟批次超时
                    try:
                        global_idx, mol_id = future_to_task[future]
                        mol_id_result, mol_block, message = future.result(timeout=60)  # 单任务60秒超时
                        
                        if mol_block:
                            batch_results.append((mol_id_result, mol_block))
                            batch_success += 1
                        else:
                            batch_failures += 1
                        
                        completed_in_batch += 1
                        
                        # 每10秒或每100个任务更新一次进度
                        current_time = time.time()
                        if (current_time - last_progress_update > 10) or (completed_in_batch % 100 == 0):
                            current_total = processed_offset + batch_idx + completed_in_batch
                            elapsed = current_time - batch_start_time
                            rate = completed_in_batch / elapsed if elapsed > 0 else 0
                            
                            extra_info = f"批次{{batch_num}}/{{batch_count}}, 成功{{batch_success}}, 失败{{batch_failures}}, {{rate:.1f}}分子/秒"
                            update_progress(current_total, total_overall, "processing", extra_info)
                            last_progress_update = current_time
                        
                    except concurrent.futures.TimeoutError:
                        global_idx, mol_id = future_to_task.get(future, (None, 'unknown'))
                        logger.error(f"任务超时 - 分子 {{mol_id}}")
                        batch_failures += 1
                    except Exception as e:
                        global_idx, mol_id = future_to_task.get(future, (None, 'unknown'))
                        logger.error(f"处理任务时出错 - 分子 {{mol_id}}: {{e}}")
                        batch_failures += 1
                
                batch_elapsed = time.time() - batch_start_time
                batch_rate = len(current_batch) / batch_elapsed if batch_elapsed > 0 else 0
                
                logger.info(f"✅ 批次 {{batch_num}} 完成: 成功{{batch_success}}, 失败{{batch_failures}}, "
                          f"耗时{{batch_elapsed:.1f}}秒, 速率{{batch_rate:.1f}}分子/秒")
                
                all_results.extend(batch_results)
                
                # 更新整体进度
                current_total = processed_offset + batch_end
                extra_info = f"已完成{{batch_num}}/{{batch_count}}批次"
                update_progress(current_total, total_overall, "processing", extra_info)
                
        except Exception as e:
            logger.error(f"批次 {{batch_num}} 处理异常: {{e}}")
        
        # 批次间强制垃圾回收
        gc.collect()
    
    return all_results

def save_results_to_sdf(results, output_file):
    """保存结果到SDF文件，包含进度更新"""
    logger.info(f"💾 开始保存 {{len(results)}} 个成功的分子构象")
    save_start_time = time.time()
    
    mol_count = 0
    conformer_count = 0
    total_results = len(results)
    
    # 更新进度为保存阶段
    update_progress(0, total_results, "saving", "开始保存构象到SDF文件")
    
    with open(output_file, 'w') as f:
        for idx, (mol_id, mol_block) in enumerate(results):
            try:
                # 重建分子对象以写入SDF
                mol = Chem.MolFromMolBlock(mol_block, removeHs=False, sanitize=False)
                if mol:
                    mol_count += 1
                    # 写入所有构象
                    for conf_id in range(mol.GetNumConformers()):
                        f.write(Chem.MolToMolBlock(mol, confId=conf_id))
                        f.write("\\n$$$$\\n")
                        conformer_count += 1
                
                # 每500个分子更新一次进度
                if (idx + 1) % 500 == 0 or (idx + 1) == total_results:
                    extra_info = f"已保存{{mol_count}}个分子, {{conformer_count}}个构象"
                    update_progress(idx + 1, total_results, "saving", extra_info)
                    
            except Exception as e:
                logger.error(f"保存分子 {{mol_id}} 时出错: {{e}}")
    
    save_elapsed = time.time() - save_start_time
    logger.info(f"✅ 保存完成，耗时: {{save_elapsed:.2f}}秒")
    logger.info(f"📊 最终统计: {{mol_count}} 个分子，{{conformer_count}} 个构象")
    
    # 更新进度为已完成
    update_progress(total_results, total_results, "completed", f"成功保存{{mol_count}}个分子, {{conformer_count}}个构象")
    
    return mol_count, conformer_count

def main():
    """主函数"""
    logger.info("🚀 开始后台分子3D构象生成")
    logger.info(f"📋 配置参数: {{CONFIG}}")
    
    # 检查输入文件
    if not os.path.exists(CONFIG['input_file']):
        logger.error(f"❌ 输入文件不存在: {{CONFIG['input_file']}}")
        return False
    
    # 确保输出目录存在
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 记录初始系统状态
    resources = monitor_resources()
    logger.info(f"系统内存: 总计{{psutil.virtual_memory().total/1024**3:.1f}}GB, "
              f"可用{{resources['available_memory_gb']:.1f}}GB")
    logger.info(f"CPU: {{psutil.cpu_count()}}核")
    
    # 检查点文件
    checkpoint_file = os.path.join(CONFIG['output_dir'], '{script_name}.checkpoint')
    
    start_time = time.time()
    
    # 尝试加载检查点
    checkpoint_data = load_checkpoint(checkpoint_file)
    if checkpoint_data:
        logger.info(f"🔄 从检查点恢复，已处理: {{len(checkpoint_data['results'])}}")
        all_results = checkpoint_data['results']
        start_idx = checkpoint_data['metadata']['processed_count']
    else:
        all_results = []
        start_idx = 0
    
    # 加载分子数据（从断点开始）
    molecules_data, mol_ids = load_molecules_from_file(
        CONFIG['input_file'], 
        CONFIG['file_type'], 
        CONFIG['smiles_column'], 
        CONFIG['processing_limit'],
        start_idx
    )
    
    if not molecules_data:
        if start_idx == 0:
            logger.error("没有加载到有效分子")
            return False
        else:
            logger.info("✅ 所有分子已处理完成")
            # 保存最终结果
            input_filename = os.path.basename(CONFIG['input_file'])
            base_name = os.path.splitext(input_filename)[0]
            output_file = os.path.join(CONFIG['output_dir'], f'conformers_{{base_name}}_{script_name}.sdf')
            
            if all_results:
                mol_count, conformer_count = save_results_to_sdf(all_results, output_file)
                
                # 写入完成标志
                completion_file = os.path.join(CONFIG['output_dir'], '{script_name}.completed')
                with open(completion_file, 'w') as f:
                    f.write(f"{{datetime.now().isoformat()}}\\n")
                    f.write(f"SUCCESS\\n")
                    f.write(f"Total results: {{len(all_results)}}\\n")
                    f.write(f"Total molecules: {{mol_count}}\\n")
                    f.write(f"Total conformers: {{conformer_count}}\\n")
                    f.write(f"Output file: {{output_file}}\\n")
                
                logger.info(f"✅ 全部完成! 输出文件: {{output_file}}")
            return True
    
    # 准备多进程任务
    tasks = [
        (mol_data, mol_id, CONFIG['num_conformers'], CONFIG['max_attempts'], CONFIG['random_seed'])
        for mol_data, mol_id in zip(molecules_data, mol_ids)
    ]
    
    logger.info(f"🚀 准备处理 {{len(tasks)}} 个分子，使用 {{CONFIG['num_workers']}} 个进程")
    
    # 分批处理
    batch_size = max(100, min(CONFIG['batch_size'], len(tasks) // 10))
    logger.info(f"批次大小: {{batch_size}}")
    
    # 使用改进的分批处理函数（包含进度条）
    results = process_batch_with_monitoring(tasks, CONFIG['num_workers'], batch_size, start_idx)
    
    # 合并结果
    all_results.extend(results)
    
    # 定期保存检查点
    if len(all_results) > 0:
        metadata = {{
            'processed_count': start_idx + len(molecules_data),
            'success_count': len(all_results),
            'timestamp': datetime.now().isoformat()
        }}
        save_checkpoint(all_results, checkpoint_file, metadata)
    
    # 保存最终结果
    if all_results:
        input_filename = os.path.basename(CONFIG['input_file'])
        base_name = os.path.splitext(input_filename)[0]
        output_file = os.path.join(CONFIG['output_dir'], f'conformers_{{base_name}}_{script_name}.sdf')
        
        mol_count, conformer_count = save_results_to_sdf(all_results, output_file)
        
        total_time = time.time() - start_time
        
        # 写入完成标志
        completion_file = os.path.join(CONFIG['output_dir'], '{script_name}.completed')
        with open(completion_file, 'w') as f:
            f.write(f"{{datetime.now().isoformat()}}\\n")
            f.write(f"SUCCESS\\n")
            f.write(f"Total time: {{total_time:.1f}}s\\n")
            f.write(f"Processed: {{start_idx + len(molecules_data)}}\\n")
            f.write(f"Success: {{len(all_results)}}\\n")
            f.write(f"Total molecules: {{mol_count}}\\n")
            f.write(f"Total conformers: {{conformer_count}}\\n")
            f.write(f"Output file: {{output_file}}\\n")
        
        # 清理检查点文件
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
        
        logger.info(f"✅ 全部完成! 输出文件: {{output_file}}")
        logger.info(f"⏱️ 总耗时: {{total_time:.1f}}秒")
        logger.info(f"📊 成功率: {{len(all_results)/(start_idx + len(molecules_data))*100:.2f}}%")
        
        return True
    else:
        logger.error("❌ 没有成功生成任何构象")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("🎊 脚本执行成功完成")
            sys.exit(0)
        else:
            logger.error("❌ 脚本执行失败")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("🛑 用户中断执行")
        sys.exit(130)
    except Exception as e:
        logger.error(f"💥 脚本执行异常: {{e}}")
        
        # 写入错误标志
        error_file = os.path.join(CONFIG['output_dir'], '{script_name}.error')
        with open(error_file, 'w') as f:
            f.write(f"{{datetime.now().isoformat()}}\\n")
            f.write(f"ERROR: {{str(e)}}\\n")
        
        sys.exit(1)
'''

    return script_content, script_name

def run_background_conformer_generation(
    input_file,
    output_dir,
    num_conformers=10,
    max_attempts=100,
    random_seed=42,
    num_workers=None,
    processing_limit=100000,
    file_type="sdf",
    smiles_column="SMILES",
    detached=True
):
    """运行后台构象生成"""
    
    # 创建脚本
    script_content, script_name = create_background_conformer_script(
        input_file=input_file,
        output_dir=output_dir,
        num_conformers=num_conformers,
        max_attempts=max_attempts,
        random_seed=random_seed,
        num_workers=num_workers,
        processing_limit=processing_limit,
        file_type=file_type,
        smiles_column=smiles_column
    )
    
    # 保存脚本文件
    script_file = os.path.join(output_dir, f"{script_name}.py")
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # 使脚本可执行
    os.chmod(script_file, 0o755)
    
    print(f"✅ 后台构象生成脚本已创建: {script_file}")
    
    if detached:
        # 在后台运行脚本
        log_file = os.path.join(output_dir, f"{script_name}_output.log")
        
        # 使用nohup在后台运行
        cmd = f"nohup python {script_file} > {log_file} 2>&1 &"
        
        print(f"🚀 启动后台进程...")
        print(f"📝 日志文件: {log_file}")
        print(f"🔍 监控命令: tail -f {log_file}")
        print(f"⛔ 停止命令: pkill -f {script_name}")
        
        os.system(cmd)
        
        # 获取进程ID
        time.sleep(1)
        pid_cmd = f"pgrep -f {script_name}"
        pid_result = subprocess.run(pid_cmd, shell=True, capture_output=True, text=True)
        
        if pid_result.returncode == 0:
            pid = pid_result.stdout.strip()
            print(f"✅ 后台进程已启动，PID: {pid}")
            
            # 保存PID文件
            pid_file = os.path.join(output_dir, f"{script_name}.pid")
            with open(pid_file, 'w') as f:
                f.write(pid)
        else:
            print("⚠️ 无法获取进程ID，但脚本可能正在运行")
    else:
        print(f"📋 手动运行命令: python {script_file}")
    
    return script_file, script_name

def check_background_conformer_status(output_dir, script_name):
    """检查后台任务状态"""
    
    # 检查PID文件
    pid_file = os.path.join(output_dir, f"{script_name}.pid")
    if os.path.exists(pid_file):
        with open(pid_file, 'r') as f:
            pid = f.read().strip()
        
        # 检查进程是否还在运行
        try:
            import psutil
            if psutil.pid_exists(int(pid)):
                process = psutil.Process(int(pid))
                return {
                    'status': 'running',
                    'pid': pid,
                    'cpu_percent': process.cpu_percent(),
                    'memory_mb': process.memory_info().rss / 1024 / 1024
                }
        except:
            pass
    
    # 检查完成标志
    completion_file = os.path.join(output_dir, f"{script_name}.completed")
    if os.path.exists(completion_file):
        with open(completion_file, 'r') as f:
            content = f.read()
        return {'status': 'completed', 'details': content}
    
    # 检查错误标志
    error_file = os.path.join(output_dir, f"{script_name}.error")
    if os.path.exists(error_file):
        with open(error_file, 'r') as f:
            content = f.read()
        return {'status': 'error', 'details': content}
    
    return {'status': 'unknown'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="后台构象生成工具")
    parser.add_argument("input_file", help="输入文件路径")
    parser.add_argument("output_dir", help="输出目录")
    parser.add_argument("--conformers", type=int, default=10, help="每个分子的构象数")
    parser.add_argument("--attempts", type=int, default=100, help="最大嵌入尝试次数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--workers", type=int, help="工作进程数")
    parser.add_argument("--limit", type=int, default=100000, help="处理限制")
    parser.add_argument("--file-type", default="sdf", help="文件类型")
    parser.add_argument("--smiles-column", default="SMILES", help="SMILES列名")
    parser.add_argument("--no-detach", action="store_true", help="不在后台运行")
    
    args = parser.parse_args()
    
    run_background_conformer_generation(
        input_file=args.input_file,
        output_dir=args.output_dir,
        num_conformers=args.conformers,
        max_attempts=args.attempts,
        random_seed=args.seed,
        num_workers=args.workers,
        processing_limit=args.limit,
        file_type=args.file_type,
        smiles_column=args.smiles_column,
        detached=not args.no_detach
    ) 