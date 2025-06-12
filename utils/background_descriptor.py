#!/usr/bin/env python3
"""
后台3D描述符计算工具 - 独立于Streamlit运行
解决页面刷新中断问题，支持断点恢复
"""

import os
import sys
import time
import json
import signal
import argparse
import subprocess
import pickle
from datetime import datetime
from pathlib import Path

def create_background_descriptor_script(
    input_file,
    output_dir,
    processing_limit=100000,
    num_workers=None,
    include_3d=True,
    aggregation_method="mean",
    include_smiles=True,
    script_name=None
):
    """创建后台描述符计算脚本"""
    
    if num_workers is None:
        import multiprocessing
        num_workers = min(34, multiprocessing.cpu_count() * 2)
    
    if script_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = f"background_descriptor_{timestamp}"
    
    # 处理processing_limit参数
    if processing_limit == float('inf'):
        processing_limit_str = "float('inf')"
    else:
        processing_limit_str = str(processing_limit)
    
    script_content = f'''#!/usr/bin/env python3
"""
后台3D分子描述符计算脚本 - 独立运行
生成时间: {datetime.now().isoformat()}
"""

import os
import sys
import time
import pickle
import logging
import signal
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
import multiprocessing

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
    from mordred import Calculator, descriptors
except ImportError as e:
    print(f"错误: 无法导入必要库: {{e}}")
    print("请确保已激活正确的conda环境并安装了Mordred")
    sys.exit(1)

# 配置参数
CONFIG = {{
    'input_file': '{input_file}',
    'output_dir': '{output_dir}',
    'processing_limit': {processing_limit_str},
    'num_workers': {num_workers},
    'include_3d': {include_3d},
    'aggregation_method': '{aggregation_method}',
    'include_smiles': {include_smiles},
    'checkpoint_interval': 500,  # 每500个分子保存一次检查点
}}

# 设置日志
log_file = os.path.join(CONFIG['output_dir'], '{script_name}_output.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, 'w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def monitor_resources():
    """监控系统资源"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    cpu_percent = process.cpu_percent()
    
    return {{
        'memory_mb': memory_mb,
        'cpu_percent': cpu_percent,
        'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
    }}

def save_checkpoint(results_data, checkpoint_file, metadata):
    """保存检查点"""
    try:
        checkpoint_data = {{
            'results_data': results_data,
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

def create_mordred_calculator(include_3d=True):
    """创建Mordred描述符计算器"""
    calc = Calculator(descriptors, ignore_3D=not include_3d)
    return calc

def process_descriptor_value(desc_val):
    """处理单个描述符值"""
    try:
        if desc_val is None:
            return np.nan
        elif isinstance(desc_val, (int, float)):
            if np.isfinite(desc_val):
                return float(desc_val)
            else:
                return np.nan
        elif hasattr(desc_val, '__float__'):
            try:
                float_val = float(desc_val)
                return float_val if np.isfinite(float_val) else np.nan
            except (ValueError, TypeError, OverflowError):
                return np.nan
        elif isinstance(desc_val, str):
            if desc_val.lower() in ['nan', 'inf', '-inf', 'none', 'error']:
                return np.nan
            else:
                try:
                    num_val = float(desc_val)
                    return num_val if np.isfinite(num_val) else np.nan
                except ValueError:
                    return np.nan
        else:
            return np.nan
    except Exception:
        return np.nan

def calculate_molecule_descriptors_worker(args):
    """多进程worker函数"""
    mol_block, mol_id, mol_props, include_3d, aggregation_method = args
    
    try:
        # 重建分子对象
        mol = Chem.MolFromMolBlock(mol_block, removeHs=False, sanitize=False)
        if mol is None:
            return (mol_id, None, None)
        
        # 恢复分子属性（因为MolToMolBlock/MolFromMolBlock会丢失属性）
        for prop_name, prop_value in mol_props.items():
            mol.SetProp(prop_name, prop_value)
        
        # 创建计算器
        calc = create_mordred_calculator(include_3d=include_3d)
        
        # 计算所有构象的描述符
        conformer_descriptors = []
        
        if mol.GetNumConformers() == 0:
            # 没有构象，计算2D描述符
            result = calc(mol)
            desc_values = [process_descriptor_value(desc_val) for desc_val in result.values()]
            conformer_descriptors.append(desc_values)
        else:
            # 有构象，逐个计算
            for conf_id in range(mol.GetNumConformers()):
                try:
                    # 创建只包含当前构象的分子副本
                    mol_copy = Chem.Mol(mol)
                    conf = mol.GetConformer(conf_id)
                    new_conf = Chem.Conformer(mol_copy.GetNumAtoms())
                    for i in range(mol_copy.GetNumAtoms()):
                        new_conf.SetAtomPosition(i, conf.GetAtomPosition(i))
                    mol_copy.AddConformer(new_conf, assignId=True)
                    
                    # 计算描述符
                    result = calc(mol_copy)
                    desc_values = [process_descriptor_value(desc_val) for desc_val in result.values()]
                    conformer_descriptors.append(desc_values)
                except Exception as e:
                    continue
        
        if not conformer_descriptors:
            return (mol_id, None, None)
        
        # 聚合多构象描述符
        descriptors_array = np.array(conformer_descriptors, dtype=float)
        
        if aggregation_method == "mean":
            aggregated_desc = np.nanmean(descriptors_array, axis=0)
        elif aggregation_method == "max":
            aggregated_desc = np.nanmax(descriptors_array, axis=0)
        elif aggregation_method == "min":
            aggregated_desc = np.nanmin(descriptors_array, axis=0)
        elif aggregation_method == "std":
            aggregated_desc = np.nanstd(descriptors_array, axis=0)
        elif aggregation_method == "median":
            aggregated_desc = np.nanmedian(descriptors_array, axis=0)
        else:
            aggregated_desc = np.nanmean(descriptors_array, axis=0)
        
        # 计算SMILES
        smiles = None
        try:
            smiles = Chem.MolToSmiles(mol) if mol else None
        except:
            smiles = "Invalid"
        
        return (mol_id, smiles, aggregated_desc.tolist())
        
    except Exception as e:
        return (mol_id, None, None)

def main():
    """主函数"""
    logger.info("🚀 开始后台3D描述符计算")
    logger.info(f"📋 配置参数: {{CONFIG}}")
    
    # 检查输入文件
    if not os.path.exists(CONFIG['input_file']):
        logger.error(f"❌ 输入文件不存在: {{CONFIG['input_file']}}")
        return False
    
    # 确保输出目录存在
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 检查点文件
    checkpoint_file = os.path.join(CONFIG['output_dir'], '{script_name}_checkpoint.pkl')
    
    start_time = time.time()
    
    # 尝试加载检查点
    checkpoint_data = load_checkpoint(checkpoint_file)
    if checkpoint_data:
        logger.info(f"🔄 从检查点恢复，已处理: {{len(checkpoint_data['results_data'])}}")
        all_mol_ids = checkpoint_data['results_data']['mol_ids']
        all_smiles = checkpoint_data['results_data']['smiles']
        all_descriptors = checkpoint_data['results_data']['descriptors']
        start_idx = checkpoint_data['metadata']['processed_count']
    else:
        all_mol_ids = []
        all_smiles = []
        all_descriptors = []
        start_idx = 0
    
    # 读取输入文件
    logger.info(f"📂 读取输入文件: {{CONFIG['input_file']}}")
    supplier = Chem.ForwardSDMolSupplier(CONFIG['input_file'], removeHs=False, sanitize=True)
    
    # 收集分子数据 - 按分子ID分组多个构象
    molecules_by_id = {{}}
    count = 0
    
    for mol in supplier:
        if count < start_idx:
            count += 1
            continue  # 跳过已处理的分子
            
        if CONFIG['processing_limit'] != float('inf') and count >= CONFIG['processing_limit']:
            break
            
        if mol is not None:
            # 获取分子ID/名称
            mol_id = mol.GetProp('_Name') if mol.HasProp('_Name') else "mol_" + str(count)
            if not mol_id.strip():  # 如果_Name为空
                mol_id = mol.GetProp('IDNUMBER') if mol.HasProp('IDNUMBER') else "mol_" + str(count)
            
            # 提取分子属性以避免MolBlock转换时丢失
            mol_props = {{}}
            for prop_name in mol.GetPropNames():
                mol_props[prop_name] = mol.GetProp(prop_name)
            
            # 特别处理_Name属性（不会出现在GetPropNames中）
            if mol.HasProp('_Name'):
                mol_props['_Name'] = mol.GetProp('_Name')
            
            # 按分子ID分组 - 合并同一分子的多个构象
            if mol_id not in molecules_by_id:
                molecules_by_id[mol_id] = {{
                    'mol': Chem.Mol(mol),  # 创建分子副本
                    'props': mol_props,
                    'conformer_count': 0
                }}
                # 清除已有构象
                molecules_by_id[mol_id]['mol'].RemoveAllConformers()
            
            # 添加当前构象到分子对象
            for conf_id in range(mol.GetNumConformers()):
                conf = mol.GetConformer(conf_id)
                new_conf = Chem.Conformer(molecules_by_id[mol_id]['mol'].GetNumAtoms())
                for i in range(molecules_by_id[mol_id]['mol'].GetNumAtoms()):
                    new_conf.SetAtomPosition(i, conf.GetAtomPosition(i))
                molecules_by_id[mol_id]['mol'].AddConformer(new_conf, assignId=True)
                molecules_by_id[mol_id]['conformer_count'] += 1
            
            count += 1
    
    # 转换为处理数据格式
    mol_data = []
    for mol_id, mol_info in molecules_by_id.items():
        mol_block = Chem.MolToMolBlock(mol_info['mol'])
        mol_data.append((mol_block, mol_id, mol_info['props'], CONFIG['include_3d'], CONFIG['aggregation_method']))
    
    if not mol_data:
        logger.info("✅ 所有分子已处理完成")
        return True
    
    logger.info(f"📊 准备处理 {{len(mol_data)}} 个独特分子")
    total_conformers = sum(mol_info['conformer_count'] for mol_info in molecules_by_id.values())
    logger.info(f"📊 总共 {{total_conformers}} 个构象将被聚合")
    
    # 多进程计算
    num_workers = CONFIG['num_workers']
    logger.info(f"⚡ 启动多进程计算 ({{num_workers}} 个进程)")
    
    results = []
    if num_workers == 1:
        # 单进程
        for i, mol_args in enumerate(mol_data):
            result = calculate_molecule_descriptors_worker(mol_args)
            results.append(result)
            if (i + 1) % 50 == 0:
                logger.info(f"已处理: {{i + 1}}/{{len(mol_data)}}")
                
                # 定期检查点
                if (i + 1) % CONFIG['checkpoint_interval'] == 0:
                    # 整理当前结果
                    current_results = {{
                        'mol_ids': all_mol_ids + [r[0] for r in results],
                        'smiles': all_smiles + [r[1] for r in results],
                        'descriptors': all_descriptors + [r[2] for r in results]
                    }}
                    metadata = {{
                        'processed_count': start_idx + i + 1,
                        'total_molecules': len(mol_data),
                        'elapsed_time': time.time() - start_time
                    }}
                    save_checkpoint(current_results, checkpoint_file, metadata)
    else:
        # 多进程
        with Pool(processes=num_workers) as pool:
            completed = 0
            last_checkpoint_time = time.time()
            
            for result in pool.imap_unordered(calculate_molecule_descriptors_worker, mol_data):
                results.append(result)
                completed += 1
                
                if completed % 50 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    speed = completed / elapsed if elapsed > 0 else 0
                    
                    # 监控资源
                    resources = monitor_resources()
                    
                    logger.info(f"⏳ 已完成: {{start_idx + completed}}/{{CONFIG['processing_limit']}} "
                              f"(速度: {{speed:.1f}} 分子/秒, 内存: {{resources['memory_mb']:.1f}}MB)")
                
                # 定期保存检查点
                if completed % CONFIG['checkpoint_interval'] == 0:
                    current_results = {{
                        'mol_ids': all_mol_ids + [r[0] for r in results],
                        'smiles': all_smiles + [r[1] for r in results], 
                        'descriptors': all_descriptors + [r[2] for r in results]
                    }}
                    metadata = {{
                        'processed_count': start_idx + completed,
                        'total_molecules': len(mol_data),
                        'elapsed_time': time.time() - start_time
                    }}
                    save_checkpoint(current_results, checkpoint_file, metadata)
    
    # 整理最终结果
    success_count = 0
    for mol_id, smiles, desc in results:
        all_mol_ids.append(mol_id)
        if desc is not None:
            all_descriptors.append(desc)
            all_smiles.append(smiles if smiles else "Invalid")
            success_count += 1
        else:
            # 获取描述符名称以填充NaN
            calc = create_mordred_calculator(include_3d=CONFIG['include_3d'])
            descriptor_names = [str(d) for d in calc.descriptors]
            all_descriptors.append([np.nan] * len(descriptor_names))
            all_smiles.append("Invalid")
    
    elapsed_time = time.time() - start_time
    logger.info(f"🎉 计算完成! 成功: {{success_count}}/{{len(mol_data)}}, 耗时: {{elapsed_time:.1f}}秒")
    
    # 创建DataFrame
    df_data = {{}}
    
    # 添加分子ID列
    df_data['Molecule_ID'] = all_mol_ids
    
    if CONFIG['include_smiles']:
        df_data['SMILES'] = all_smiles
    
    # 获取描述符名称
    calc = create_mordred_calculator(include_3d=CONFIG['include_3d'])
    descriptor_names = [str(d) for d in calc.descriptors]
    
    descriptors_array = np.array(all_descriptors)
    for i, desc_name in enumerate(descriptor_names):
        df_data[desc_name] = descriptors_array[:, i]
    
    df = pd.DataFrame(df_data)
    
    # 保存结果
    input_filename = os.path.basename(CONFIG['input_file'])
    base_name = os.path.splitext(input_filename)[0]
    output_file = os.path.join(CONFIG['output_dir'], f'descriptors_{{base_name}}_{script_name}.csv')
    
    df.to_csv(output_file, index=False)
    
    logger.info(f"✅ 结果已保存到: {{output_file}}")
    logger.info(f"📊 输出形状: {{df.shape}}")
    
    # 统计信息
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        valid_desc = df[numeric_cols].notna().sum().sum()
        total_desc = len(df) * len(numeric_cols)
        coverage = valid_desc / total_desc * 100 if total_desc > 0 else 0
        logger.info(f"📈 有效描述符覆盖率: {{coverage:.1f}}%")
    
    total_time = time.time() - start_time
    
    # 写入完成标志
    completion_file = os.path.join(CONFIG['output_dir'], '{script_name}_completed.txt')
    with open(completion_file, 'w') as f:
        f.write(f"{{datetime.now().isoformat()}}\\n")
        f.write(f"SUCCESS\\n")
        f.write(f"Total time: {{total_time:.1f}}s\\n")
        f.write(f"Processed: {{len(all_mol_ids)}} molecules\\n")
        f.write(f"Success rate: {{success_count}}/{{len(mol_data)}}\\n")
        f.write(f"Output file: {{output_file}}\\n")
        f.write(f"Output shape: {{df.shape}}\\n")
        f.write(f"Descriptor coverage: {{coverage:.1f}}%\\n")
    
    # 清理检查点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    logger.info(f"🎊 全部完成! 输出文件: {{output_file}}")
    logger.info(f"⏱️ 总耗时: {{total_time:.1f}}秒")
    
    return True

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
        error_file = os.path.join('{output_dir}', '{script_name}_error.txt')
        with open(error_file, 'w') as f:
            f.write(f"{{datetime.now().isoformat()}}\\n")
            f.write(f"ERROR: {{str(e)}}\\n")
        
        sys.exit(1)
'''

    return script_content, script_name

def run_background_descriptor_calculation(
    input_file,
    output_dir,
    processing_limit=100000,
    num_workers=None,
    include_3d=True,
    aggregation_method="mean",
    include_smiles=True,
    detached=True
):
    """运行后台描述符计算"""
    
    # 创建脚本
    script_content, script_name = create_background_descriptor_script(
        input_file=input_file,
        output_dir=output_dir,
        processing_limit=processing_limit,
        num_workers=num_workers,
        include_3d=include_3d,
        aggregation_method=aggregation_method,
        include_smiles=include_smiles
    )
    
    # 保存脚本文件
    script_file = os.path.join(output_dir, f"{script_name}.py")
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # 使脚本可执行
    os.chmod(script_file, 0o755)
    
    print(f"✅ 后台描述符计算脚本已创建: {script_file}")
    
    if detached:
        # 在后台运行脚本
        log_file = os.path.join(output_dir, f"{script_name}_output.log")
        
        # 检测conda环境
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'base')
        python_path = sys.executable  # 使用当前Python解释器路径
        
        # 使用conda run确保在正确环境中运行
        if conda_env and conda_env != 'base':
            cmd = f"nohup conda run -n {conda_env} python {script_file} > {log_file} 2>&1 &"
        else:
            # 直接使用当前Python解释器
            cmd = f"nohup {python_path} {script_file} > {log_file} 2>&1 &"
        
        print(f"🚀 启动后台进程...")
        print(f"🐍 Python环境: {conda_env} ({python_path})")
        print(f"📝 日志文件: {log_file}")
        print(f"💻 执行命令: {cmd}")
        print(f"🔍 监控命令: tail -f {log_file}")
        print(f"⛔ 停止命令: pkill -f {script_name}")
        
        # 使用subprocess而不是os.system以便更好的错误处理
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"⚠️ 启动命令返回码: {result.returncode}")
            if result.stderr:
                print(f"❌ 错误信息: {result.stderr}")
        else:
            print("✅ 后台命令已提交")
        
        # 获取进程ID
        time.sleep(1)
        pid_cmd = f"pgrep -f {script_name}"
        pid_result = subprocess.run(pid_cmd, shell=True, capture_output=True, text=True)
        
        if pid_result.returncode == 0:
            pid = pid_result.stdout.strip()
            print(f"✅ 后台进程已启动，PID: {pid}")
            
            # 保存PID文件
            pid_file = os.path.join(output_dir, f"{script_name}_pid.txt")
            with open(pid_file, 'w') as f:
                f.write(pid)
        else:
            print("⚠️ 无法获取进程ID，但脚本可能正在运行")
    else:
        print(f"📋 手动运行命令: python {script_file}")
    
    return script_file, script_name

def check_background_descriptor_status(output_dir, script_name):
    """检查后台描述符计算任务状态"""
    
    # 检查PID文件
    pid_file = os.path.join(output_dir, f"{script_name}_pid.txt")
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
    completion_file = os.path.join(output_dir, f"{script_name}_completed.txt")
    if os.path.exists(completion_file):
        with open(completion_file, 'r') as f:
            content = f.read()
        return {'status': 'completed', 'details': content}
    
    # 检查错误标志
    error_file = os.path.join(output_dir, f"{script_name}_error.txt")
    if os.path.exists(error_file):
        with open(error_file, 'r') as f:
            content = f.read()
        return {'status': 'error', 'details': content}
    
    return {'status': 'unknown'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="后台3D描述符计算工具")
    parser.add_argument("input_file", help="输入SDF文件路径")
    parser.add_argument("output_dir", help="输出目录")
    parser.add_argument("--limit", type=int, default=100000, help="处理限制")
    parser.add_argument("--workers", type=int, help="工作进程数")
    parser.add_argument("--no-3d", action="store_true", help="不包含3D描述符")
    parser.add_argument("--aggregation", default="mean", help="聚合方法")
    parser.add_argument("--no-smiles", action="store_true", help="不包含SMILES")
    parser.add_argument("--no-detach", action="store_true", help="不在后台运行")
    
    args = parser.parse_args()
    
    run_background_descriptor_calculation(
        input_file=args.input_file,
        output_dir=args.output_dir,
        processing_limit=args.limit,
        num_workers=args.workers,
        include_3d=not args.no_3d,
        aggregation_method=args.aggregation,
        include_smiles=not args.no_smiles,
        detached=not args.no_detach
    ) 