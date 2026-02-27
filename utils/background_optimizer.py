#!/usr/bin/env python3
"""
后台构象优化工具 - 独立于Streamlit运行
解决页面刷新中断问题
"""

import os
import time
import argparse
import subprocess
from datetime import datetime

def create_background_optimizer_script(
    input_file,
    output_dir,
    processing_limit=100000,
    num_threads=None,
    optimization_steps=1000,
    script_name=None
):
    """创建后台优化脚本"""
    
    if num_threads is None:
        import multiprocessing
        num_threads = min(36, multiprocessing.cpu_count() * 2)
    
    if script_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        script_name = f"background_optimize_{timestamp}"
    
    script_content = f'''#!/usr/bin/env python3
"""
后台分子构象优化脚本 - 独立运行
生成时间: {datetime.now().isoformat()}
"""

import os
import sys
import time
import pickle
import logging
import signal
import psutil
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
    from rdkit.Chem import AllChem
except ImportError as e:
    print(f"错误: 无法导入RDKit: {{e}}")
    print("请确保已激活正确的conda环境")
    sys.exit(1)

# 配置参数
CONFIG = {{
    'input_file': '{input_file}',
    'output_dir': '{output_dir}',
    'processing_limit': {processing_limit},
    'num_threads': {num_threads},
    'optimization_steps': {optimization_steps},
    'checkpoint_interval': 1000,  # 每1000个分子保存一次检查点
}}

# 设置日志
log_file = os.path.join(CONFIG['output_dir'], '{script_name}.log')
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

def save_checkpoint(optimized_mols, checkpoint_file, metadata):
    """保存检查点"""
    try:
        checkpoint_data = {{
            'optimized_mols': optimized_mols,
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

def preprocess_molecule(mol):
    """预处理分子"""
    if mol is None:
        return None
    
    try:
        # 保存原始分子的属性
        original_props = {{}}
        for prop_name in mol.GetPropNames():
            original_props[prop_name] = mol.GetProp(prop_name)
        
        # 保存原始分子名称
        original_name = mol.GetProp('_Name') if mol.HasProp('_Name') else None
        
        # 检查是否有3D坐标 - 在处理之前检查原分子
        if mol.GetNumConformers() == 0:
            return None
            
        # 检查坐标有效性
        try:
            conf = mol.GetConformer()
            for i in range(mol.GetNumAtoms()):
                pos = conf.GetAtomPosition(i)
                if not all([abs(pos.x) < 999, abs(pos.y) < 999, abs(pos.z) < 999]):
                    return None
        except Exception:
            return None
        
        # 创建分子副本
        mol_clean = Chem.Mol(mol)
        
        # 进行基本的分子清理
        try:
            Chem.SanitizeMol(mol_clean)
        except Exception:
            return None
        
        # 验证分子结构
        if mol_clean.GetNumAtoms() == 0:
            return None
        
        # 恢复所有原始属性
        for prop_name, prop_value in original_props.items():
            mol_clean.SetProp(prop_name, prop_value)
        
        # 恢复原始名称
        if original_name is not None:
            mol_clean.SetProp('_Name', original_name)
        
        return mol_clean
    except Exception as e:
        return None

def optimize_molecule_rdkit(mol, conf_id=0, steps=1000):
    """使用RDKit MMFF94力场优化分子"""
    try:
        # 创建分子副本
        mol_copy = Chem.Mol(mol)
        
        # 保留原始分子的所有属性和ID
        for prop_name in mol.GetPropNames():
            mol_copy.SetProp(prop_name, mol.GetProp(prop_name))
        
        # 如果原始分子有名称（_Name），也要保留
        if mol.HasProp('_Name'):
            mol_copy.SetProp('_Name', mol.GetProp('_Name'))
        
        # 添加氢原子并保留3D坐标
        mol_with_h = Chem.AddHs(mol_copy, addCoords=True)
        
        # 检查是否成功添加氢原子
        if mol_with_h is None or mol_with_h.GetNumConformers() == 0:
            # 如果添加氢原子失败，尝试直接优化原分子
            mol_with_h = mol_copy
        
        # 使用MMFF94力场优化
        try:
            mp = AllChem.MMFFGetMoleculeProperties(mol_with_h)
            if mp is None:
                return {{'mol': None, 'success': False, 'message': 'MMFF94力场初始化失败'}}
            
            ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, mp, confId=conf_id)
            if ff is None:
                return {{'mol': None, 'success': False, 'message': 'MMFF94力场创建失败'}}
            
            # 执行优化
            converged = ff.Minimize(maxIts=steps)
            
            # 如果添加了氢原子，需要将坐标映射回原分子
            if mol_with_h.GetNumAtoms() != mol_copy.GetNumAtoms():
                # 移除氢原子，更新原始分子坐标
                conf_with_h = mol_with_h.GetConformer()
                conf_orig = mol_copy.GetConformer(conf_id)
                
                heavy_idx = 0
                for i in range(mol_with_h.GetNumAtoms()):
                    atom = mol_with_h.GetAtomWithIdx(i)
                    if atom.GetAtomicNum() != 1:  # 非氢原子
                        if heavy_idx < mol_copy.GetNumAtoms():
                            pos = conf_with_h.GetAtomPosition(i)
                            conf_orig.SetAtomPosition(heavy_idx, pos)
                            heavy_idx += 1
            else:
                # 没有添加氢原子，直接使用优化后的坐标
                mol_copy = mol_with_h
            
            return {{'mol': mol_copy, 'success': True, 'message': f'优化成功(收敛代码:{{converged}})'}}
        except Exception as e:
            return {{'mol': None, 'success': False, 'message': f'MMFF94优化失败: {{str(e)}}'}}
            
    except Exception as e:
        return {{'mol': None, 'success': False, 'message': f'优化异常: {{str(e)}}'}}

def optimize_single_molecule(args):
    """多进程优化单个分子的包装函数"""
    mol_data, orig_idx, steps = args
    mol, mol_props = mol_data  # 分子和属性分开传递
    
    try:
        # 恢复分子属性（因为pickle会丢失属性）
        for prop_name, prop_value in mol_props.items():
            mol.SetProp(prop_name, prop_value)
        
        result = optimize_molecule_rdkit(mol, 0, steps)
        
        # 返回时也需要包含属性信息，因为分子对象在返回时会再次丢失属性
        # 确保mol_props总是有效的字典（即使为空）
        if result['success']:
            # 如果mol_props为空，添加一个默认属性确保不会被误判为失败
            if not mol_props:
                mol_props = {{'optimization_status': 'completed'}}
            return {{
                'orig_idx': orig_idx,
                'mol': result['mol'],
                'mol_props': mol_props,
                'success': True,
                'message': result['message']
            }}
        else:
            return {{
                'orig_idx': orig_idx,
                'mol': None,
                'mol_props': None,
                'success': False,
                'message': result['message']
            }}
    except Exception as e:
        return {{
            'orig_idx': orig_idx,
            'mol': None,
            'mol_props': None,
            'success': False,
            'message': f'进程异常: {{str(e)}}'
        }}

def save_results_to_sdf(mols, output_file):
    """保存结果到SDF文件"""
    logger.info(f"保存结果到: {{output_file}}")
    with Chem.SDWriter(output_file) as writer:
        for mol in mols:
            if mol:
                writer.write(mol)

def main():
    """主函数"""
    logger.info("🚀 开始后台分子构象优化")
    logger.info(f"📋 配置参数: {{CONFIG}}")
    
    # 检查输入文件
    if not os.path.exists(CONFIG['input_file']):
        logger.error(f"❌ 输入文件不存在: {{CONFIG['input_file']}}")
        return False
    
    # 确保输出目录存在
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 检查点文件
    checkpoint_file = os.path.join(CONFIG['output_dir'], '{script_name}.checkpoint')
    
    start_time = time.time()
    
    # 尝试加载检查点
    checkpoint_data = load_checkpoint(checkpoint_file)
    if checkpoint_data:
        logger.info(f"🔄 从检查点恢复，已处理: {{len(checkpoint_data['optimized_mols'])}}")
        optimized_mols = checkpoint_data['optimized_mols']
        start_idx = checkpoint_data['metadata']['processed_count']
    else:
        optimized_mols = []
        start_idx = 0
    
    # 读取输入文件
    logger.info(f"📂 读取输入文件: {{CONFIG['input_file']}}")
    supplier = Chem.ForwardSDMolSupplier(CONFIG['input_file'], removeHs=False, sanitize=True)
    
    # 收集要处理的分子
    mols_to_optimize = []
    skipped_count = 0
    
    for i, mol in enumerate(supplier):
        if i < start_idx:
            continue  # 跳过已处理的分子
            
        if len(mols_to_optimize) >= CONFIG['processing_limit'] - start_idx:
            break
            
        if mol is not None:
            processed_mol = preprocess_molecule(mol)
            if processed_mol is not None:
                # 提取分子属性以避免pickle丢失
                mol_props = {{}}
                for prop_name in processed_mol.GetPropNames():
                    mol_props[prop_name] = processed_mol.GetProp(prop_name)
                
                # 特别处理_Name属性（不会出现在GetPropNames中）
                if processed_mol.HasProp('_Name'):
                    mol_props['_Name'] = processed_mol.GetProp('_Name')
                
                # 确保每个分子至少有一个标识属性，避免空字典被误判为失败
                if not mol_props:
                    mol_props['molecule_id'] = f'mol_{{start_idx + len(mols_to_optimize)}}'
                
                mols_to_optimize.append(((processed_mol, mol_props), start_idx + len(mols_to_optimize)))
            else:
                skipped_count += 1
    
    if not mols_to_optimize:
        logger.info("✅ 所有分子已处理完成")
        return True
    
    logger.info(f"📊 待优化分子数: {{len(mols_to_optimize)}}, 跳过: {{skipped_count}}")
    
    # 多进程优化
    success_count = 0
    num_processes = CONFIG['num_threads']
    optimization_steps = CONFIG['optimization_steps']
    
    logger.info(f"⚡ 启动多进程优化 ({{num_processes}} 个进程)")
    
    # 多进程处理
    process_args = [(mol_data[0], mol_data[1], optimization_steps) for mol_data in mols_to_optimize]
    
    with Pool(processes=num_processes) as pool:
        results = []
        completed = 0
        last_checkpoint_time = time.time()
        
        for result_data in pool.imap_unordered(optimize_single_molecule, process_args):
            results.append(result_data)
            completed += 1
            
            if result_data['success'] and result_data['mol'] is not None:
                # 恢复分子属性（因为从子进程返回时属性丢失）
                mol = result_data['mol']
                mol_props = result_data['mol_props']
                
                # 如果有属性，则恢复；空字典也是有效的（表示没有额外属性）
                if mol_props is not None:
                    for prop_name, prop_value in mol_props.items():
                        mol.SetProp(prop_name, prop_value)
                
                optimized_mols.append(mol)
                success_count += 1
            
            # 定期更新进度和检查点
            if completed % 50 == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                speed = completed / elapsed if elapsed > 0 else 0
                
                # 监控资源
                resources = monitor_resources()
                
                logger.info(f"⏳ 已完成: {{start_idx + completed}}/{{CONFIG['processing_limit']}} "
                          f"(成功: {{len(optimized_mols)}}, 速度: {{speed:.1f}} 分子/秒, "
                          f"内存: {{resources['memory_mb']:.1f}}MB)")
                
                # 定期保存检查点
                if current_time - last_checkpoint_time > 300:  # 每5分钟保存一次
                    metadata = {{
                        'processed_count': start_idx + completed,
                        'success_count': len(optimized_mols),
                        'elapsed_time': elapsed
                    }}
                    save_checkpoint(optimized_mols, checkpoint_file, metadata)
                    last_checkpoint_time = current_time
    
    optimization_time = time.time() - start_time
    logger.info(f"🎉 优化完成! 成功: {{len(optimized_mols)}}/{{len(mols_to_optimize)}}, 耗时: {{optimization_time:.1f}}秒")
    
    if not optimized_mols:
        logger.error("❌ 没有成功优化的分子")
        return False
    
    # 保存最终结果
    input_filename = os.path.basename(CONFIG['input_file'])
    base_name = os.path.splitext(input_filename)[0]
    output_file = os.path.join(CONFIG['output_dir'], f'optimized_{{base_name}}_{script_name}.sdf')
    
    save_results_to_sdf(optimized_mols, output_file)
    
    total_time = time.time() - start_time
    
    # 写入完成标志
    completion_file = os.path.join(CONFIG['output_dir'], '{script_name}.completed')
    with open(completion_file, 'w') as f:
        f.write(f"{{datetime.now().isoformat()}}\\n")
        f.write(f"SUCCESS\\n")
        f.write(f"Total time: {{total_time:.1f}}s\\n")
        f.write(f"Optimized: {{len(optimized_mols)}}/{{len(mols_to_optimize)}}\\n")
        f.write(f"Output file: {{output_file}}\\n")
        f.write(f"Output conformers: {{len(optimized_mols)}}\\n")
    
    # 清理检查点文件
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
    
    logger.info(f"✅ 全部完成! 输出文件: {{output_file}}")
    logger.info(f"⏱️ 总耗时: {{total_time:.1f}}秒")
    logger.info(f"📊 输出构象数: {{len(optimized_mols)}}")
    
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
        error_file = os.path.join('{output_dir}', '{script_name}.error')
        with open(error_file, 'w') as f:
            f.write(f"{{datetime.now().isoformat()}}\\n")
            f.write(f"ERROR: {{str(e)}}\\n")
        
        sys.exit(1)
'''

    return script_content, script_name

def run_background_optimization(
    input_file,
    output_dir,
    processing_limit=100000,
    num_threads=None,
    optimization_steps=1000,
    detached=True
):
    """运行后台优化"""
    
    # 创建脚本
    script_content, script_name = create_background_optimizer_script(
        input_file=input_file,
        output_dir=output_dir,
        processing_limit=processing_limit,
        num_threads=num_threads,
        optimization_steps=optimization_steps
    )
    
    # 保存脚本文件
    script_file = os.path.join(output_dir, f"{script_name}.py")
    with open(script_file, 'w') as f:
        f.write(script_content)
    
    # 使脚本可执行
    os.chmod(script_file, 0o755)
    
    print(f"✅ 后台优化脚本已创建: {script_file}")
    
    if detached:
        # 在后台运行脚本
        log_file = os.path.join(output_dir, f"{script_name}_output.log")
        
        # 使用nohup在后台运行
        cmd = f"nohup python {script_file} > {log_file} 2>&1 &"
        
        print("🚀 启动后台进程...")
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

def check_background_status(output_dir, script_name):
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
        except Exception:
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
    parser = argparse.ArgumentParser(description="后台构象优化工具")
    parser.add_argument("input_file", help="输入SDF文件路径")
    parser.add_argument("output_dir", help="输出目录")
    parser.add_argument("--limit", type=int, default=100000, help="处理限制")
    parser.add_argument("--threads", type=int, help="线程数")
    parser.add_argument("--steps", type=int, default=1000, help="优化步数")
    parser.add_argument("--no-detach", action="store_true", help="不在后台运行")
    
    args = parser.parse_args()
    
    run_background_optimization(
        input_file=args.input_file,
        output_dir=args.output_dir,
        processing_limit=args.limit,
        num_threads=args.threads,
        optimization_steps=args.steps,
        detached=not args.no_detach
    ) 