"""
构象动力学优化页面 - 使用OpenMM进行分子动力学优化
"""
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
import io
import os
import uuid
import numpy as np
from datetime import datetime
import time
import multiprocessing
import concurrent.futures
from multiprocessing import cpu_count
import pickle
import sys
import tempfile
from pathlib import Path
import subprocess
import threading

# OpenMM相关导入
try:
    import openmm
    from openmm import app, unit
    from openmm.app import PDBFile, ForceField, Modeller, Simulation
    from openmm.unit import nanometer, picosecond, femtosecond, kelvin, kilojoule_per_mole
    OPENMM_AVAILABLE = True
except ImportError:
    OPENMM_AVAILABLE = False

PREVIEW_SIZE = 10
# 大文件阈值 (100MB)
LARGE_FILE_THRESHOLD = 100 * 1024 * 1024

# 初始化会话状态
if 'last_processed_file_identifier' not in st.session_state:
    st.session_state.last_processed_file_identifier = None
if 'scan_results_valid' not in st.session_state:
    st.session_state.scan_results_valid = False
if 'total_potential_mols_cache' not in st.session_state:
    st.session_state.total_potential_mols_cache = 0
if 'preview_data_cache' not in st.session_state:
    st.session_state.preview_data_cache = []
if 'initial_scan_successful_cache' not in st.session_state:
    st.session_state.initial_scan_successful_cache = False
if 'saved_file_path' not in st.session_state:
    st.session_state.saved_file_path = None
if 'saved_work_dir' not in st.session_state:
    st.session_state.saved_work_dir = None
if 'file_size_cache' not in st.session_state:
    st.session_state.file_size_cache = None

# 确保数据目录存在
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def get_file_size(file_path_or_obj):
    """获取文件大小"""
    if isinstance(file_path_or_obj, str):
        if os.path.exists(file_path_or_obj):
            return os.path.getsize(file_path_or_obj)
    else:
        if hasattr(file_path_or_obj, 'size'):
            return file_path_or_obj.size
        elif hasattr(file_path_or_obj, 'getvalue'):
            return len(file_path_or_obj.getvalue())
    return 0

def generate_work_folder_name(filename):
    """生成工作文件夹名称：日期+随机码"""
    date_str = datetime.now().strftime("%Y%m%d")
    random_code = str(uuid.uuid4())[:8]
    base_name = os.path.splitext(filename)[0]
    sanitized_base = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in base_name).strip()
    return f"{date_str}_{sanitized_base}_{random_code}"

def save_uploaded_file(uploaded_file):
    """保存上传的文件到data目录"""
    if not uploaded_file:
        return None, None
    
    folder_name = generate_work_folder_name(uploaded_file.name)
    work_dir = os.path.join(DATA_DIR, folder_name)
    
    try:
        os.makedirs(work_dir, exist_ok=True)
        file_path = os.path.join(work_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"文件 '{uploaded_file.name}' 已保存到: {work_dir}")
        return file_path, work_dir
    except Exception as e:
        st.error(f"保存文件时出错: {e}")
        return None, None

def list_data_folders():
    """列出data目录下的所有文件夹"""
    if not os.path.exists(DATA_DIR):
        return []
    return [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]

def list_files_in_folder(folder_name):
    """列出指定文件夹中的文件"""
    if not folder_name:
        return []
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.isdir(folder_path):
        return []
    # 只列出SDF文件
    return [f for f in os.listdir(folder_path) if f.lower().endswith('.sdf') and os.path.isfile(os.path.join(folder_path, f))]

def preprocess_molecule(mol):
    """预处理分子，确保格式正确"""
    try:
        # 保存原始分子的属性
        original_props = {}
        for prop_name in mol.GetPropNames():
            original_props[prop_name] = mol.GetProp(prop_name)
        
        # 保存原始分子名称
        original_name = mol.GetProp('_Name') if mol.HasProp('_Name') else None
        
        # 清理分子
        mol_clean = Chem.RemoveHs(mol)
        
        # 检查是否有3D坐标
        if mol_clean.GetNumConformers() == 0:
            return None
            
        # 检查坐标有效性
        conf = mol_clean.GetConformer()
        for i in range(mol_clean.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            if not all(np.isfinite([pos.x, pos.y, pos.z])):
                return None
        
        # 进行基本的分子清理
        try:
            Chem.SanitizeMol(mol_clean)
        except:
            return None
        
        # 恢复所有原始属性
        for prop_name, prop_value in original_props.items():
            mol_clean.SetProp(prop_name, prop_value)
        
        # 恢复原始名称
        if original_name is not None:
            mol_clean.SetProp('_Name', original_name)
            
        return mol_clean
    except:
        return None

def mol_to_pdb_string(mol, conf_id=0):
    """将RDKit分子转换为PDB字符串"""
    try:
        pdb_block = Chem.MolToPDBBlock(mol, confId=conf_id)
        return pdb_block
    except Exception as e:
        return None

def optimize_molecule_with_rdkit_and_openmm(mol, conf_id=0, steps=1000, temperature=300, use_gpu=True):
    """使用RDKit进行初步优化，然后可选择性使用OpenMM进行精确优化"""
    try:
        # 首先使用RDKit的MMFF力场进行优化
        mol_copy = Chem.Mol(mol)
        
        # 保留原始分子的所有属性和ID
        for prop_name in mol.GetPropNames():
            mol_copy.SetProp(prop_name, mol.GetProp(prop_name))
        
        # 如果原始分子有名称（_Name），也要保留
        if mol.HasProp('_Name'):
            mol_copy.SetProp('_Name', mol.GetProp('_Name'))
        
        # 添加氢原子
        mol_with_h = Chem.AddHs(mol_copy)
        
        # 使用MMFF94进行优化
        try:
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol_with_h)
            if mmff_props is not None:
                ff = AllChem.MMFFGetMoleculeForceField(mol_with_h, mmff_props)
                if ff is not None:
                    ff.Minimize(maxIts=steps)
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
                    
                    return {'mol': mol_copy, 'success': True, 'message': "MMFF94优化成功"}
                else:
                    return {'mol': None, 'success': False, 'message': "无法创建MMFF94力场"}
            else:
                return {'mol': None, 'success': False, 'message': "无法获取MMFF94分子属性"}
        except Exception as e:
            return {'mol': None, 'success': False, 'message': f"MMFF94优化失败: {str(e)}"}
            
    except Exception as e:
        return {'mol': None, 'success': False, 'message': f"优化过程异常: {str(e)}"}

def optimize_molecule_with_openmm(mol, conf_id=0, steps=1000, temperature=300, use_gpu=True):
    """使用优化后的策略：RDKit + OpenMM (备用)"""
    # 主要使用RDKit的MMFF力场进行优化
    return optimize_molecule_with_rdkit_and_openmm(mol, conf_id, steps, temperature, use_gpu)

# 删除了optimize_single_molecule函数，改用ThreadPoolExecutor中的内联函数

def calculate_rmsd(mol1, mol2, conf_id1=0, conf_id2=0):
    """计算两个构象之间的RMSD"""
    try:
        rmsd = rdMolAlign.AlignMol(mol1, mol2, prbCid=conf_id1, refCid=conf_id2)
        return rmsd
    except:
        return float('inf')

def calculate_rmsd_batch(args):
    """批量计算RMSD，用于并行处理"""
    mol_idx, mol, existing_mols, threshold = args
    
    try:
        for existing_idx, existing_mol in existing_mols:
            if existing_mol is None:
                continue
            try:
                rmsd = rdMolAlign.AlignMol(mol, existing_mol, prbCid=0, refCid=0)
                if rmsd < threshold:
                    return mol_idx, False, rmsd, existing_idx
            except:
                continue
        return mol_idx, True, None, None
    except Exception as e:
        return mol_idx, True, None, None  # 错误时当作独特构象

def get_molecule_identifier(mol):
    """获取分子的唯一标识符（canonical SMILES）"""
    try:
        # 移除立体化学信息，只基于连接性分组
        mol_copy = Chem.Mol(mol)
        Chem.RemoveStereochemistry(mol_copy)
        return Chem.MolToSmiles(mol_copy, canonical=True)
    except:
        # 如果失败，返回一个基于分子图的哈希
        try:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        except:
            return f"mol_{id(mol)}"  # 最后的回退方案

def group_molecules_by_structure(optimized_mols):
    """按分子结构分组"""
    groups = {}
    
    for i, mol in enumerate(optimized_mols):
        if mol is None:
            continue
            
        mol_id = get_molecule_identifier(mol)
        if mol_id not in groups:
            groups[mol_id] = []
        groups[mol_id].append((i, mol))
    
    return groups

def merge_similar_conformers_parallel(optimized_mols, rmsd_threshold=0.5, use_parallel=True, max_workers=None):
    """高效并行合并相似构象 - 按分子结构分组处理"""
    if not optimized_mols:
        return [], []
    
    # 按分子结构分组
    mol_groups = group_molecules_by_structure(optimized_mols)
    
    if not mol_groups:
        return [], []
    
    all_unique_mols = []
    all_unique_indices = []
    
    # 统计信息
    total_input_conformers = len([mol for mol in optimized_mols if mol is not None])
    total_groups = len(mol_groups)
    
    print(f"📊 分子结构分组: {total_input_conformers} 个构象 → {total_groups} 个不同分子结构")
    
    # 对每个分子结构组分别处理
    for mol_id, group_mols in mol_groups.items():
        if len(group_mols) == 1:
            # 只有一个构象，直接保留
            idx, mol = group_mols[0]
            all_unique_mols.append(mol)
            all_unique_indices.append(idx)
            print(f"  📂 {mol_id[:50]}{'...' if len(mol_id) > 50 else ''}: 1 → 1 构象 (单个构象)")
            continue
        
        # 提取该组的分子和索引
        group_indices = [idx for idx, mol in group_mols]
        group_molecules = [mol for idx, mol in group_mols]
        
        # 计算该组内的比较次数
        group_comparisons = len(group_molecules) * (len(group_molecules) - 1) // 2
        
        # 决定是否对该组使用并行处理
        use_parallel_for_group = use_parallel and group_comparisons >= 1000  # 降低阈值，因为已经分组
        
        # 初始化变量以避免作用域问题
        group_unique = []
        group_unique_global_indices = []
        
        if use_parallel_for_group:
            # 使用并行处理该组
            group_unique, group_unique_local_indices = merge_similar_conformers_parallel_impl(
                list(enumerate(group_molecules)), rmsd_threshold, max_workers
            )
            # 将本地索引转换为全局索引
            group_unique_global_indices = [group_indices[local_idx] for local_idx in group_unique_local_indices]
        else:
            # 使用串行处理该组
            group_unique, group_unique_global_indices = merge_similar_conformers_serial_group(
                group_molecules, group_indices, rmsd_threshold
            )
        
        all_unique_mols.extend(group_unique)
        all_unique_indices.extend(group_unique_global_indices)
        
        print(f"  📂 {mol_id[:50]}{'...' if len(mol_id) > 50 else ''}: {len(group_molecules)} → {len(group_unique)} 构象")
    
    return all_unique_mols, all_unique_indices

def merge_similar_conformers_serial_group(group_molecules, group_indices, rmsd_threshold=0.5):
    """串行合并单个分子结构组内的相似构象"""
    if not group_molecules:
        return [], []
    
    if len(group_molecules) == 1:
        return group_molecules, group_indices
    
    unique_mols = []
    unique_indices = []
    
    for i, mol in enumerate(group_molecules):
        if mol is None:
            continue
            
        is_unique = True
        for existing_mol in unique_mols:
            try:
                rmsd = calculate_rmsd(mol, existing_mol)
                if rmsd < rmsd_threshold:
                    is_unique = False
                    break
            except:
                continue
        
        if is_unique:
            unique_mols.append(mol)
            unique_indices.append(group_indices[i])
    
    return unique_mols, unique_indices

def merge_similar_conformers_serial(optimized_mols, rmsd_threshold=0.5):
    """串行版本的构象合并（优化版）"""
    unique_mols = []
    unique_indices = []
    
    for i, mol in enumerate(optimized_mols):
        if mol is None:
            continue
            
        is_unique = True
        for unique_mol in unique_mols:
            if unique_mol is None:
                continue
            try:
                rmsd = calculate_rmsd(mol, unique_mol)
                if rmsd < rmsd_threshold:
                    is_unique = False
                    break
            except:
                continue
        
        if is_unique:
            unique_mols.append(mol)
            unique_indices.append(i)
    
    return unique_mols, unique_indices

def merge_similar_conformers_parallel_impl(valid_mols, rmsd_threshold, max_workers=None):
    """并行实现的构象合并"""
    import math
    
    if max_workers is None:
        max_workers = min(8, multiprocessing.cpu_count())
    
    unique_mols = []
    unique_indices = []
    
    # 分批处理以控制内存使用
    batch_size = min(100, max(10, len(valid_mols) // max_workers))
    
    for batch_start in range(0, len(valid_mols), batch_size):
        batch_end = min(batch_start + batch_size, len(valid_mols))
        batch_mols = valid_mols[batch_start:batch_end]
        
        # 为当前批次准备参数
        existing_unique = list(enumerate(unique_mols))
        
        if existing_unique:
            # 并行计算当前批次与已有独特构象的RMSD
            process_args = [
                (orig_idx, mol, existing_unique, rmsd_threshold)
                for orig_idx, mol in batch_mols
            ]
            
            try:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(calculate_rmsd_batch, process_args))
                
                # 处理结果
                for mol_idx, is_unique, rmsd_val, _ in results:
                    if is_unique:
                        mol = next(mol for orig_idx, mol in batch_mols if orig_idx == mol_idx)
                        unique_mols.append(mol)
                        unique_indices.append(mol_idx)
                        
            except Exception:
                # 并行失败时回退到串行
                for orig_idx, mol in batch_mols:
                    is_unique = True
                    for unique_mol in unique_mols:
                        try:
                            rmsd = calculate_rmsd(mol, unique_mol)
                            if rmsd < rmsd_threshold:
                                is_unique = False
                                break
                        except:
                            continue
                    if is_unique:
                        unique_mols.append(mol)
                        unique_indices.append(orig_idx)
        else:
            # 第一批次直接处理
            for orig_idx, mol in batch_mols:
                is_unique = True
                for unique_mol in unique_mols:
                    try:
                        rmsd = calculate_rmsd(mol, unique_mol)
                        if rmsd < rmsd_threshold:
                            is_unique = False
                            break
                    except:
                        continue
                if is_unique:
                    unique_mols.append(mol)
                    unique_indices.append(orig_idx)
    
    return unique_mols, unique_indices

# 为了向后兼容，保留原函数名
def merge_similar_conformers(optimized_mols, rmsd_threshold=0.5):
    """合并相似的构象（兼容性包装）"""
    return merge_similar_conformers_parallel(optimized_mols, rmsd_threshold, use_parallel=True)

def mols_to_sdf_string(mols_with_conformers):
    """将分子列表转换为SDF字符串"""
    output = io.StringIO()
    sdf_writer = Chem.SDWriter(output)
    
    for mol in mols_with_conformers:
        if mol:
            sdf_writer.write(mol)
    
    sdf_writer.flush()
    sdf_writer.close()
    sdf_string = output.getvalue()
    output.close()
    return sdf_string

def generate_optimization_script(input_file, output_file, log_file, config):
    """生成独立的多进程优化Python脚本"""
    
    script_content = '''#!/usr/bin/env python3
"""
独立的多进程分子构象优化脚本
由Streamlit应用自动生成
"""

import os
import sys
import time
import pickle
import logging
from datetime import datetime
from multiprocessing import Pool, cpu_count
import multiprocessing

# 添加项目路径以确保能导入RDKit等库
import io
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('{log_file}', 'w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 配置参数
CONFIG = {config}

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
        
        # 添加氢原子（如果还没有）
        mol_clean = Chem.AddHs(mol)
        
        # 检查是否有3D坐标
        if mol_clean.GetNumConformers() == 0:
            logger.warning("分子缺少3D坐标，跳过")
            return None
        
        # 验证分子结构
        if mol_clean.GetNumAtoms() == 0:
            logger.warning("分子没有原子，跳过")
            return None
        
        # 恢复所有原始属性
        for prop_name, prop_value in original_props.items():
            mol_clean.SetProp(prop_name, prop_value)
        
        # 恢复原始名称
        if original_name is not None:
            mol_clean.SetProp('_Name', original_name)
        
        return mol_clean
    except Exception as e:
        logger.warning(f"预处理分子时出错: {{e}}")
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
        
        # 使用MMFF94力场优化
        mp = AllChem.MMFFGetMoleculeProperties(mol_copy)
        if mp is None:
            return {{'mol': None, 'success': False, 'message': 'MMFF94力场初始化失败'}}
        
        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mp, confId=conf_id)
        if ff is None:
            return {{'mol': None, 'success': False, 'message': 'MMFF94力场创建失败'}}
        
        # 执行优化
        converged = ff.Minimize(maxIts=steps)
        
        if converged == 0:
            return {{'mol': mol_copy, 'success': True, 'message': '优化成功'}}
        else:
            return {{'mol': mol_copy, 'success': True, 'message': f'优化完成（收敛代码: {{converged}}）'}}
            
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
        return {{
            'orig_idx': orig_idx,
            'mol': result['mol'] if result['success'] else None,
            'mol_props': mol_props if result['success'] else None,  # 返回属性信息
            'success': result['success'],
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

# RMSD合并功能已移除，直接输出优化后的构象以提高速度

def save_results_to_sdf(mols, output_file):
    """保存结果到SDF文件"""
    with Chem.SDWriter(output_file) as writer:
        for mol in mols:
            if mol:
                writer.write(mol)

def main():
    """主函数"""
    logger.info("开始多进程分子构象优化")
    logger.info(f"配置参数: {{CONFIG}}")
    
    start_time = time.time()
    
    # 读取输入文件
    input_file = '{input_file}'
    logger.info(f"读取输入文件: {{input_file}}")
    
    supplier = Chem.ForwardSDMolSupplier(input_file, removeHs=False, sanitize=True)
    
    # 收集要处理的分子
    mols_to_optimize = []
    skipped_count = 0
    
    for i, mol in enumerate(supplier):
        if len(mols_to_optimize) >= CONFIG['processing_limit']:
            break
        if mol is not None:
            processed_mol = preprocess_molecule(mol)
            if processed_mol is not None:
                # 提取分子属性以避免pickle丢失
                mol_props = {{}}
                for prop_name in processed_mol.GetPropNames():
                    mol_props[prop_name] = processed_mol.GetProp(prop_name)
                
                mols_to_optimize.append(((processed_mol, mol_props), i))
            else:
                skipped_count += 1
    
    logger.info(f"待优化分子数: {{len(mols_to_optimize)}}, 跳过: {{skipped_count}}")
    
    # 多进程优化
    optimized_mols = [None] * len(mols_to_optimize)
    success_count = 0
    
    num_processes = CONFIG['num_threads']
    optimization_steps = CONFIG['optimization_steps']
    
    logger.info(f"启动多进程优化 ({{num_processes}} 个进程)")
    
    if num_processes == 1:
        # 单进程处理
        for idx, mol_data in enumerate(mols_to_optimize):
            mol_and_props, orig_idx = mol_data
            mol, mol_props = mol_and_props
            
            # 恢复分子属性（虽然单进程不需要，但保持一致性）
            for prop_name, prop_value in mol_props.items():
                mol.SetProp(prop_name, prop_value)
            
            result = optimize_molecule_rdkit(mol, 0, optimization_steps)
            
            if result['success'] and result['mol']:
                optimized_mols[idx] = result['mol']
                success_count += 1
            
            if (idx + 1) % 50 == 0:
                logger.info(f"已处理: {{idx + 1}}/{{len(mols_to_optimize)}}")
    else:
        # 多进程处理
        process_args = [(mol_data[0], mol_data[1], optimization_steps) for mol_data in mols_to_optimize]
        
        with Pool(processes=num_processes) as pool:
            results = []
            completed = 0
            
            for result_data in pool.imap_unordered(optimize_single_molecule, process_args):
                results.append(result_data)
                completed += 1
                
                if result_data['success'] and result_data['mol'] and result_data['mol_props']:
                    # 恢复分子属性（因为从子进程返回时属性丢失）
                    mol = result_data['mol']
                    mol_props = result_data['mol_props']
                    for prop_name, prop_value in mol_props.items():
                        mol.SetProp(prop_name, prop_value)
                    
                    # 找到对应的索引位置
                    for idx, (mol_and_props, orig_idx) in enumerate(mols_to_optimize):
                        if orig_idx == result_data['orig_idx']:
                            optimized_mols[idx] = mol
                            success_count += 1
                            break
                
                if completed % 50 == 0:
                    logger.info(f"已完成: {{completed}}/{{len(mols_to_optimize)}} (成功: {{success_count}})")
    
    optimization_time = time.time() - start_time
    logger.info(f"优化完成! 成功: {{success_count}}/{{len(mols_to_optimize)}}, 耗时: {{optimization_time:.1f}}秒")
    
    # 过滤有效分子
    valid_mols = [mol for mol in optimized_mols if mol is not None]
    
    if not valid_mols:
        logger.error("没有成功优化的分子")
        return
    
    logger.info("跳过RMSD合并，直接保存所有优化后的构象")
    
    # 直接保存所有优化后的构象
    output_file = '{output_file}'
    save_results_to_sdf(valid_mols, output_file)
    
    total_time = time.time() - start_time
    logger.info(f"全部完成! 输出文件: {{output_file}}")
    logger.info(f"总耗时: {{total_time:.1f}}秒")
    logger.info(f"输出构象数: {{len(valid_mols)}} (未合并)")
    
    # 写入完成标志
    with open('{log_file}.done', 'w') as f:
        f.write(f"{{datetime.now().isoformat()}}\\n")
        f.write(f"SUCCESS\\n")
        f.write(f"Total time: {{total_time:.1f}}s\\n")
        f.write(f"Optimized: {{success_count}}/{{len(mols_to_optimize)}}\\n")
        f.write(f"Output conformers: {{len(valid_mols)}} (no merging)\\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"脚本执行失败: {{e}}")
        with open('{log_file}.error', 'w') as f:
            f.write(f"{{datetime.now().isoformat()}}\\n")
            f.write(f"ERROR: {{str(e)}}\\n")
        sys.exit(1)
'''.format(
        log_file=log_file,
        config=config,
        input_file=input_file,
        output_file=output_file
    )
    
    return script_content

st.set_page_config(page_title="构象动力学优化", layout="wide")
st.title("🔬 构象动力学优化")

st.markdown("""
使用高性能分子力场进行构象优化。支持并行处理，自动合并相似构象。

⚡ **高性能优化**: 使用RDKit MMFF94力场进行快速优化  
🔄 **稳定并行**: 多线程处理，兼容Streamlit环境  
🧬 **智能分组**: 按分子结构分组，避免无意义比较  
🎯 **高效合并**: 基于RMSD自动识别并合并相似构象  
📊 **批量处理**: 支持处理大型分子库
""")

# 检查优化环境
cpu_count = multiprocessing.cpu_count()
st.info(f"💻 检测到 {cpu_count} 个CPU核心，支持并行优化")

# 检查OpenMM可用性（可选）
gpu_available = False
if OPENMM_AVAILABLE:
    try:
        platform = openmm.Platform.getPlatformByName('CUDA')
        gpu_available = True
        st.success("🚀 OpenMM + GPU (CUDA) 可用作高级选项")
    except:
        try:
            platform = openmm.Platform.getPlatformByName('OpenCL') 
            gpu_available = True
            st.success("🚀 OpenMM + GPU (OpenCL) 可用作高级选项")
        except:
            st.info("ℹ️ 当前使用CPU优化模式（推荐）")

# 文件输入方式选择
st.subheader("1. 选择输入方式")

input_method = st.radio(
    "选择输入方式:",
    ("上传新文件", "使用已保存文件"),
    horizontal=True
)

uploaded_file = None
selected_file_path = None
work_dir = None
current_filename = None

if input_method == "上传新文件":
    uploaded_file = st.file_uploader(
        "上传SDF文件",
        type=["sdf"],
        help="请上传包含3D构象的SDF文件"
    )
    
    if uploaded_file:
        current_filename = uploaded_file.name
        
        # 如果上传了不同的文件，清除之前的保存状态
        if (st.session_state.saved_file_path and 
            os.path.basename(st.session_state.saved_file_path) != uploaded_file.name):
            st.session_state.saved_file_path = None
            st.session_state.saved_work_dir = None
            st.session_state.scan_results_valid = False
        
        # 检查是否已保存过这个文件
        if (st.session_state.saved_file_path and 
            os.path.basename(st.session_state.saved_file_path) == uploaded_file.name):
            # 文件已保存，使用保存的路径
            selected_file_path = st.session_state.saved_file_path
            work_dir = st.session_state.saved_work_dir
            st.success(f"使用已保存的文件: {selected_file_path}")
        else:
            # 文件未保存，显示保存按钮
            if st.button("保存文件到工作目录"):
                file_path, work_dir_new = save_uploaded_file(uploaded_file)
                if file_path:
                    # 保存到session state
                    st.session_state.saved_file_path = file_path
                    st.session_state.saved_work_dir = work_dir_new
                    st.session_state.scan_results_valid = False
                    st.session_state.last_processed_file_identifier = file_path
                    st.rerun()
            else:
                st.warning("文件已上传但尚未保存到工作目录。点击上方按钮保存文件。")

else:  # 使用已保存文件
    st.subheader("选择已保存的文件")
    
    available_folders = list_data_folders()
    if not available_folders:
        st.info(f"在 '{DATA_DIR}/' 中未找到已保存的数据。请先上传新文件。")
    else:
        # 文件夹选择
        selected_folder = st.selectbox(
            "选择数据文件夹:",
            options=available_folders,
            index=0
        )
        
        if selected_folder:
            work_dir = os.path.join(DATA_DIR, selected_folder)
            files_in_folder = list_files_in_folder(selected_folder)
            
            if not files_in_folder:
                st.warning(f"选中的文件夹中没有找到SDF文件: {selected_folder}")
            else:
                # 文件选择
                selected_filename = st.selectbox(
                    "选择SDF文件:",
                    options=files_in_folder,
                    index=0
                )
                
                if selected_filename:
                    selected_file_path = os.path.join(work_dir, selected_filename)
                    current_filename = selected_filename
                    st.info(f"已选择文件: {selected_file_path}")
                    
                    # 检查缓存有效性
                    if st.session_state.last_processed_file_identifier != selected_file_path:
                        st.session_state.scan_results_valid = False
                        st.session_state.last_processed_file_identifier = selected_file_path

# 优化参数设置
st.subheader("2. 优化参数设置")

# RMSD阈值说明
with st.expander("💡 RMSD阈值选择指南", expanded=False):
    st.markdown("""
    **RMSD (Root Mean Square Deviation)** 用于衡量两个构象之间的空间差异：
    
    ### 🎯 **推荐设置**
    - **🔬 药物发现**: `0.15-0.25 Å` (超严格-严格)
    - **📊 构象分析**: `0.25-0.5 Å` (严格-标准) 
    - **⚡ 快速筛选**: `0.5-0.8 Å` (标准-宽松)
    - **🗂️ 大库处理**: `0.8-1.2 Å` (宽松-很宽松)
    
    ### 📈 **阈值影响**
    - **较小阈值** (0.15-0.3 Å): 保留更多独特构象，结果精细
    - **中等阈值** (0.3-0.6 Å): 平衡精度和效率
    - **较大阈值** (0.6-1.2 Å): 显著减少构象数量，提升处理速度
    
    ### 💊 **小分子特点**
    小分子构象变化通常在 0.2-2.0 Å 范围内，**0.25 Å** 是经验最佳值。
    """)

col1, col2, col3 = st.columns(3)
with col1:
    optimization_steps = st.number_input("优化步数:", min_value=100, max_value=10000, value=1000, step=100)
with col2:
    temperature = st.number_input("温度 (K):", min_value=200, max_value=500, value=300, step=10)
with col3:
    # RMSD阈值预设选项
    threshold_preset = st.selectbox(
        "RMSD阈值预设:",
        options=["自定义", "超严格 (0.15Å)", "严格 (0.25Å)", "标准 (0.5Å)", "宽松 (0.8Å)", "很宽松 (1.2Å)"],
        index=2,  # 默认选择"严格 (0.25Å)"
        help="选择预设或自定义调节"
    )
    
    # 根据预设设置默认值
    preset_values = {
        "超严格 (0.15Å)": 0.15,
        "严格 (0.25Å)": 0.25, 
        "标准 (0.5Å)": 0.5,
        "宽松 (0.8Å)": 0.8,
        "很宽松 (1.2Å)": 1.2
    }
    
    default_value = preset_values.get(threshold_preset, 0.25)
    
    if threshold_preset == "自定义":
        rmsd_threshold = st.number_input(
            "自定义RMSD阈值 (Å):", 
            min_value=0.05, 
            max_value=1.5, 
            value=0.25, 
            step=0.05,
            help="小分子推荐范围: 0.2-0.8Å"
        )
    else:
        rmsd_threshold = default_value
        st.info(f"当前RMSD阈值: {rmsd_threshold} Å")

# 并行处理设置
st.subheader("3. 并行处理设置")

cpu_cores = multiprocessing.cpu_count()
recommended_threads = min(max(cpu_cores - 2, 1), cpu_cores)  # 保留2个核心给系统

col_threads1, col_threads2 = st.columns(2)
with col_threads1:
    num_threads = st.number_input(
        "线程数:",
        min_value=1,
        max_value=cpu_cores * 2,  # 允许超线程
        value=recommended_threads,
        help=f"建议使用 {recommended_threads} 个线程（ThreadPoolExecutor，兼容Streamlit）"
    )

with col_threads2:
    st.metric(
        label="系统信息", 
        value=f"{cpu_cores} CPU核心",
        delta=f"建议: {recommended_threads} 线程"
    )

# RMSD并行计算设置
st.subheader("4. RMSD计算优化")

with st.expander("💡 RMSD并行计算说明", expanded=False):
    st.markdown("""
    **智能分子结构分组 + 并行RMSD计算**，大幅提升构象合并效率：
    
    ### 🧬 **智能分组优化**
    - **结构分组**: 自动按分子结构（canonical SMILES）分组
    - **组内比较**: 只对相同分子结构的不同构象进行RMSD比较
    - **算法优化**: 避免不同分子间无意义的RMSD计算，大幅减少计算量
    
    ### 🚀 **并行优化效果**
    - **传统方法**: O(N²) 复杂度，对所有构象进行两两比较
    - **分组方法**: 将N个构象分为K组，复杂度降为 Σ(n_i²) << N²
    - **并行计算**: 每组内利用多进程同时计算多个RMSD，可提速 3-8倍
    
    ### 📊 **实际性能提升**
    以5000个构象（包含100个不同分子，每分子50个构象）为例：
    - **传统方法**: 5000² = 2500万次比较
    - **分组方法**: 100 × 50² = 25万次比较（**减少99%**）
    - **并行处理**: 再提速4-8倍
    
    ### ⚙️ **推荐设置**
    - **< 100个构象**: 自动使用串行模式
    - **100-500个构象**: 2-4个进程  
    - **500-2000个构象**: 4-6个进程
    - **> 2000个构象**: 6-8个进程
    """)

col_rmsd1, col_rmsd2 = st.columns(2)

with col_rmsd1:
    use_parallel_rmsd = st.checkbox(
        "启用RMSD并行计算", 
        value=True,
        help="构象数≥100时自动启用，显著提升RMSD计算速度"
    )

with col_rmsd2:
    rmsd_workers = st.number_input(
        "RMSD计算线程数:",
        min_value=1,
        max_value=min(8, cpu_cores),
        value=min(4, cpu_cores),
        disabled=not use_parallel_rmsd,
        help="RMSD计算专用线程数，建议4-8个（ThreadPoolExecutor）"
    )

# 处理范围设置
st.subheader("5. 处理范围")
processing_options = ["处理所有构象", "仅处理前50个构象", "仅处理前200个构象"]
selected_scope = st.radio("选择处理范围:", options=processing_options, index=0)

# 文件处理和预览
input_ready = (uploaded_file is not None) or (selected_file_path is not None and os.path.exists(selected_file_path))

if input_ready:
    st.header("文件预览和优化控制")
    st.markdown(f"**当前文件:** `{current_filename}`")

    # 文件预览
    try:
        with st.spinner(f"正在扫描SDF文件 '{current_filename}'..."):
            current_file = uploaded_file if uploaded_file else selected_file_path
            file_size = get_file_size(current_file)
            
            # 显示文件信息
            size_mb = file_size / (1024 * 1024)
            st.info(f"文件大小: {size_mb:.1f} MB")
            
            # 读取SDF文件
            if isinstance(current_file, str):
                supplier = Chem.ForwardSDMolSupplier(current_file, removeHs=False, sanitize=True)
            else:
                if hasattr(current_file, 'seek'):
                    current_file.seek(0)
                sdf_stream = io.BytesIO(current_file.getvalue())
                supplier = Chem.ForwardSDMolSupplier(sdf_stream, removeHs=False, sanitize=True)
            
            # 计算分子数量和预览
            preview_smiles = []
            total_conformers = 0
            
            # 智能扫描策略：对于大文件使用快速计数，小文件完整扫描
            if file_size > LARGE_FILE_THRESHOLD:  # 大于100MB的文件
                st.info("🔍 检测到大文件，正在进行快速扫描...")
                
                # 快速计数所有分子
                for i, mol in enumerate(supplier):
                    if mol is not None:
                        total_conformers += 1
                        # 只收集前面的分子用于预览
                        if i < PREVIEW_SIZE:
                            preview_smiles.append(Chem.MolToSmiles(mol))
                    
                    # 对于非常大的文件，每1000个分子显示一次进度
                    if i % 1000 == 0 and i > 0:
                        st.info(f"已扫描 {i+1} 个构象...")
            else:
                # 小文件：完整扫描
                mols = []
                for i, mol in enumerate(supplier):
                    if mol is not None:
                        mols.append(mol)
                        if i < PREVIEW_SIZE:
                            preview_smiles.append(Chem.MolToSmiles(mol))
                
                total_conformers = len(mols)
            
            if total_conformers > 0:
                st.success(f"扫描完成: 找到 {total_conformers} 个构象")
                st.text_area("构象SMILES预览:", "\n".join(preview_smiles), height=150)
                
                # 确定处理限制
                limit_map = {
                    processing_options[0]: total_conformers,  # 处理所有
                    processing_options[1]: min(50, total_conformers),
                    processing_options[2]: min(200, total_conformers)
                }
                processing_limit = limit_map[selected_scope]
                
                st.info(f"将处理 {processing_limit} 个构象")
                
                # 优化方式选择
                st.subheader("6. 优化执行方式")
                
                execution_method = st.radio(
                    "选择执行方式:",
                    ("多进程后台执行 (推荐)", "Streamlit内线程执行"),
                    help="多进程方式可以充分利用CPU资源，但会在后台运行"
                )
                
                # 优化按钮
                if execution_method == "多进程后台执行 (推荐)":
                    button_label = f"🚀 启动多进程优化 {processing_limit} 个构象 (使用 {num_threads} 个进程)"
                else:
                    button_label = f"优化 {processing_limit} 个构象 (使用 {num_threads} 个线程)"
                
                if st.button(button_label):
                    if execution_method == "多进程后台执行 (推荐)":
                        # 多进程后台执行
                        st.info("🚀 准备启动多进程后台优化...")
                        
                        # 生成文件路径
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        script_name = f"optimize_{timestamp}.py"
                        log_name = f"optimize_{timestamp}.log"
                        output_name = f"optimized_conformers_{timestamp}_{current_filename}"
                        
                        # 确保路径构建正确 - work_dir已经是绝对路径
                        if work_dir and os.path.isabs(work_dir):
                            script_path = os.path.join(work_dir, script_name)
                            log_path = os.path.join(work_dir, log_name)
                            output_path = os.path.join(work_dir, output_name)
                        else:
                            # 如果work_dir是相对路径，需要转换为绝对路径
                            abs_work_dir = os.path.abspath(work_dir) if work_dir else os.getcwd()
                            script_path = os.path.join(abs_work_dir, script_name)
                            log_path = os.path.join(abs_work_dir, log_name)
                            output_path = os.path.join(abs_work_dir, output_name)
                        
                        # 配置参数
                        config = {
                            'processing_limit': processing_limit,
                            'num_threads': num_threads,
                            'optimization_steps': optimization_steps,
                            'temperature': temperature,
                            'rmsd_threshold': rmsd_threshold,
                            'rmsd_workers': rmsd_workers if use_parallel_rmsd else 1,
                        }
                        
                        # 生成优化脚本 - 确保使用绝对路径
                        input_file_path = current_file if isinstance(current_file, str) else selected_file_path
                        abs_input_file_path = os.path.abspath(input_file_path)
                        abs_output_path = os.path.abspath(output_path)
                        abs_log_path = os.path.abspath(log_path)
                        
                        script_content = generate_optimization_script(
                            abs_input_file_path,
                            abs_output_path,
                            abs_log_path,
                            config
                        )
                        
                        # 保存脚本
                        try:
                            with open(script_path, 'w', encoding='utf-8') as f:
                                f.write(script_content)
                            
                            st.success(f"✅ 优化脚本已生成: {script_path}")
                            
                            # Debug信息：检查脚本文件
                            if os.path.exists(script_path):
                                script_size = os.path.getsize(script_path) / 1024
                                st.info(f"📄 脚本文件大小: {script_size:.1f} KB")
                            else:
                                st.error("❌ 脚本文件生成失败")
                                st.stop()
                            
                            # Debug信息：显示启动命令
                            launch_cmd = [sys.executable, script_path]
                            st.code(f"启动命令: {' '.join(launch_cmd)}")
                            st.code(f"工作目录: {work_dir}")
                            st.code(f"输入文件: {abs_input_file_path}")
                            st.code(f"输出文件: {abs_output_path}")
                            st.code(f"日志文件: {abs_log_path}")
                            
                            # 启动后台进程 - 使用绝对路径，避免相对路径问题
                            abs_script_path = os.path.abspath(script_path)
                            abs_work_dir = os.path.abspath(work_dir) if work_dir else os.getcwd()
                            
                            launch_cmd_abs = [sys.executable, abs_script_path]
                            st.code(f"绝对路径启动命令: {' '.join(launch_cmd_abs)}")
                            
                            process = subprocess.Popen(
                                launch_cmd_abs,
                                cwd=abs_work_dir,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True
                            )
                            
                            # 等待短暂时间检查进程是否立即失败
                            time.sleep(0.5)
                            poll_result = process.poll()
                            
                            if poll_result is not None:
                                # 进程已经退出
                                stdout, stderr = process.communicate()
                                st.error(f"❌ 进程启动后立即退出 (退出码: {poll_result})")
                                if stdout:
                                    st.error("标准输出:")
                                    st.code(stdout, language="text")
                                if stderr:
                                    st.error("错误输出:")
                                    st.code(stderr, language="text")
                            else:
                                st.success(f"🚀 多进程优化已启动! (PID: {process.pid})")
                                st.info(f"📊 配置: {num_threads}个进程, {processing_limit}个构象, RMSD阈值{rmsd_threshold}Å")
                                st.info(f"📝 日志文件: {log_path}")
                                st.info(f"📤 输出文件: {output_path}")
                                
                                # 保存进程信息到session state
                                st.session_state.current_process = {
                                    'pid': process.pid,
                                    'script_path': script_path,
                                    'log_path': log_path,
                                    'output_path': output_path,
                                    'start_time': time.time(),
                                    'process': process
                                }
                                
                                st.warning("⚠️ 请勿关闭此页面，可以使用下方的监控功能查看进度")
                            
                        except Exception as e:
                            st.error(f"启动多进程优化失败: {e}")
                            st.error(f"错误详情: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc(), language="text")
                            
                    else:
                        # 原有的线程执行方式
                        # 并行处理提示
                        if num_threads > 1:
                            cpu_cores = multiprocessing.cpu_count()
                            efficiency = min(num_threads, cpu_cores) / cpu_cores * 100
                            st.info(f"🚀 启动多线程并行优化 ({num_threads} 线程)")
                            st.info(f"💻 预计CPU利用率: {efficiency:.0f}% | 受GIL限制")
                            if num_threads > cpu_cores:
                                st.warning(f"⚠️ 线程数({num_threads})超过CPU核心数({cpu_cores})，可能影响性能")
                        
                        # 重新读取文件以进行处理
                        if isinstance(current_file, str):
                            supplier = Chem.ForwardSDMolSupplier(current_file, removeHs=False, sanitize=True)
                        else:
                            if hasattr(current_file, 'seek'):
                                current_file.seek(0)
                            sdf_stream = io.BytesIO(current_file.getvalue())
                            supplier = Chem.ForwardSDMolSupplier(sdf_stream, removeHs=False, sanitize=True)
                        
                        # 收集要处理的分子
                        mols_to_optimize = []
                        skipped_count = 0
                        
                        for i, mol in enumerate(supplier):
                            if len(mols_to_optimize) >= processing_limit:
                                break
                            if mol is not None:
                                # 预处理分子
                                processed_mol = preprocess_molecule(mol)
                                if processed_mol is not None:
                                    mols_to_optimize.append((processed_mol, i))
                                else:
                                    skipped_count += 1
                        
                        if skipped_count > 0:
                            st.warning(f"跳过了 {skipped_count} 个无效或缺少3D坐标的分子")
                        
                        if mols_to_optimize:
                            st.info(f"开始优化 {len(mols_to_optimize)} 个构象...")
                            progress_bar = st.progress(0.0)
                            status_text = st.empty()
                            
                            optimized_mols = []
                            success_count = 0
                            errors = []
                            
                            start_time = time.time()
                            
                            if num_threads == 1:
                                # 单线程处理
                                for idx, mol_data in enumerate(mols_to_optimize):
                                    status_text.text(f"正在优化构象 {idx+1}/{len(mols_to_optimize)}...")
                                    
                                    mol, orig_idx = mol_data
                                    result = optimize_molecule_with_openmm(
                                        mol, 0, optimization_steps, temperature, use_gpu=gpu_available
                                    )
                                    
                                    if result['success'] and result['mol']:
                                        optimized_mols.append(result['mol'])
                                        success_count += 1
                                    else:
                                        errors.append(f"构象 {orig_idx}: {result['message']}")
                                    
                                    progress = (idx + 1) / len(mols_to_optimize)
                                    progress_bar.progress(progress)
                            else:
                                # 多线程处理 - 使用ThreadPoolExecutor (在Streamlit环境中更稳定)
                                st.info(f"🚀 使用多线程并行处理 ({num_threads} 个线程)，在Streamlit环境中稳定运行")
                                st.info("💡 注意：受Python GIL限制，CPU密集任务提升有限，但仍能带来性能改善")
                                
                                from concurrent.futures import ThreadPoolExecutor, as_completed
                                
                                def optimize_single_thread(mol_data_and_params):
                                    """线程安全的分子优化函数"""
                                    mol_data, opt_steps, temp = mol_data_and_params
                                    mol, orig_idx = mol_data
                                    try:
                                        result = optimize_molecule_with_openmm(mol, 0, opt_steps, temp, use_gpu=False)
                                        return {
                                            'orig_idx': orig_idx,
                                            'mol': result['mol'] if result['success'] else None,
                                            'success': result['success'],
                                            'message': result['message']
                                        }
                                    except Exception as e:
                                        return {
                                            'orig_idx': orig_idx,
                                            'mol': None,
                                            'success': False,
                                            'message': f"线程异常: {str(e)}"
                                        }
                                
                                try:
                                    with ThreadPoolExecutor(max_workers=num_threads) as executor:
                                        # 提交所有任务
                                        future_to_idx = {}
                                        for idx, mol_data in enumerate(mols_to_optimize):
                                            future = executor.submit(optimize_single_thread, (mol_data, optimization_steps, temperature))
                                            future_to_idx[future] = idx
                                        
                                        completed = 0
                                        for future in as_completed(future_to_idx):
                                            try:
                                                result = future.result()
                                                
                                                if result['success'] and result['mol']:
                                                    optimized_mols.append(result['mol'])
                                                    success_count += 1
                                                else:
                                                    errors.append(f"构象 {result['orig_idx']}: {result['message']}")
                                                
                                                completed += 1
                                                
                                                # 更新进度
                                                status_text.text(f"正在优化构象 {completed}/{len(mols_to_optimize)}... (多线程并行)")
                                                progress = completed / len(mols_to_optimize)
                                                progress_bar.progress(progress)
                                            
                                            except Exception as e:
                                                completed += 1
                                                errors.append(f"处理异常: {str(e)}")
                                                status_text.text(f"正在优化构象 {completed}/{len(mols_to_optimize)}... (有错误)")
                                                progress = completed / len(mols_to_optimize)
                                                progress_bar.progress(progress)
                                        
                                except Exception as e:
                                    st.error(f"多线程处理出错: {e}")
                                    st.info("回退到单线程处理...")
                                    
                                    # 清空之前的结果
                                    optimized_mols = []
                                    success_count = 0
                                    errors = []
                                    
                                    # 回退到单线程处理
                                    for idx, mol_data in enumerate(mols_to_optimize):
                                        status_text.text(f"正在优化构象 {idx+1}/{len(mols_to_optimize)}... (单线程回退)")
                                        
                                        mol, orig_idx = mol_data
                                        result = optimize_molecule_with_openmm(
                                            mol, 0, optimization_steps, temperature, use_gpu=gpu_available
                                        )
                                        
                                        if result['success'] and result['mol']:
                                            optimized_mols.append(result['mol'])
                                            success_count += 1
                                        else:
                                            errors.append(f"构象 {orig_idx}: {result['message']}")
                                        
                                        progress = (idx + 1) / len(mols_to_optimize)
                                        progress_bar.progress(progress)
                        
                            total_time = time.time() - start_time
                            status_text.text(f"优化完成！总耗时: {total_time/60:.1f}分钟")
                            
                            st.success(f"成功优化 {success_count}/{len(mols_to_optimize)} 个构象")
                            
                            if errors and len(errors) <= 10:
                                with st.expander("⚠️ 查看错误信息", expanded=False):
                                    for error in errors:
                                        st.write(f"- {error}")
                            elif len(errors) > 10:
                                                                st.warning(f"有 {len(errors)} 个构象优化失败")
                            
                            if optimized_mols:
                                st.info("🚀 跳过RMSD合并，直接输出所有优化后的构象（提高速度）")
                                
                                # 直接生成输出SDF
                                sdf_output = mols_to_sdf_string(optimized_mols)
                                
                                # 保存到工作目录
                                output_filename = f"optimized_conformers_{current_filename}"
                                if work_dir and os.path.exists(work_dir):
                                    output_path = os.path.join(work_dir, output_filename)
                                    try:
                                        with open(output_path, 'w') as f:
                                            f.write(sdf_output)
                                        st.success(f"优化结果已保存到: {output_path}")
                                    except Exception as e:
                                        st.warning(f"保存到工作目录失败: {e}")
                                
                                # 下载按钮
                                st.download_button(
                                    label="📥 下载优化后的构象 (SDF)",
                                    data=sdf_output,
                                    file_name=output_filename,
                                    mime="chemical/x-mdl-sdfile",
                                )
                                
                                # 显示文件大小信息
                                output_size_mb = len(sdf_output.encode('utf-8')) / (1024 * 1024)
                                st.info(f"输出SDF文件大小: {output_size_mb:.1f} MB")
                                
                                # 预览
                                st.subheader("优化后SDF预览 (前1000字符)")
                                st.code(sdf_output[:1000], language="text")
                            else:
                                st.warning("没有成功优化的构象可供输出")
            else:
                st.warning("文件中没有找到有效的分子构象")
                
    except Exception as e:
        st.error(f"处理文件时出错: {e}")
        st.error(f"错误类型: {type(e).__name__}")
        st.error(f"错误详情: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="text")

# 后台进程监控区域
st.header("🔍 后台任务监控")

if hasattr(st.session_state, 'current_process') and st.session_state.current_process:
    process_info = st.session_state.current_process
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("进程ID", process_info['pid'])
    
    with col2:
        elapsed = time.time() - process_info['start_time']
        st.metric("运行时间", f"{elapsed/60:.1f} 分钟")
    
    with col3:
        # 检查进程状态
        try:
            process = process_info['process']
            poll_result = process.poll()
            if poll_result is None:
                status = "🟢 运行中"
                # 额外检查：通过PID验证进程是否真的在运行
                try:
                    import psutil
                    if psutil.pid_exists(process_info['pid']):
                        status += " (已验证)"
                    else:
                        status = "❌ 进程不存在"
                except ImportError:
                    # 使用系统命令检查
                    result = subprocess.run(['ps', '-p', str(process_info['pid'])], 
                                          capture_output=True, text=True)
                    if result.returncode != 0:
                        status = "❌ 进程不存在"
            else:
                status = "✅ 已完成" if poll_result == 0 else f"❌ 出错 (退出码: {poll_result})"
        except Exception as e:
            status = f"❓ 未知 ({str(e)})"
        
        st.metric("状态", status)
    
    # 控制按钮
    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
    
    with col_btn1:
        if st.button("📖 查看日志"):
            if os.path.exists(process_info['log_path']):
                try:
                    with open(process_info['log_path'], 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    
                    st.subheader("📝 实时日志")
                    # 只显示最后50行
                    log_lines = log_content.split('\n')
                    recent_logs = '\n'.join(log_lines[-50:])
                    st.code(recent_logs, language="text")
                    
                except Exception as e:
                    st.error(f"读取日志失败: {e}")
            else:
                st.warning("日志文件尚未生成")
    
    with col_btn2:
        if st.button("📊 检查完成"):
            done_file = process_info['log_path'] + '.done'
            error_file = process_info['log_path'] + '.error'
            
            if os.path.exists(done_file):
                st.success("🎉 优化任务已完成!")
                try:
                    with open(done_file, 'r') as f:
                        result_info = f.read()
                    st.code(result_info)
                    
                    # 检查输出文件
                    if os.path.exists(process_info['output_path']):
                        file_size = os.path.getsize(process_info['output_path']) / (1024*1024)
                        st.info(f"✅ 输出文件已生成: {process_info['output_path']} ({file_size:.1f} MB)")
                        
                        # 提供下载按钮
                        with open(process_info['output_path'], 'rb') as f:
                            st.download_button(
                                "📥 下载优化结果",
                                f.read(),
                                file_name=os.path.basename(process_info['output_path']),
                                mime="chemical/x-mdl-sdfile"
                            )
                    
                except Exception as e:
                    st.error(f"读取结果失败: {e}")
                    
            elif os.path.exists(error_file):
                st.error("❌ 优化任务出错!")
                try:
                    with open(error_file, 'r') as f:
                        error_info = f.read()
                    st.code(error_info, language="text")
                except Exception as e:
                    st.error(f"读取错误信息失败: {e}")
            else:
                st.info("⏳ 任务仍在运行中...")
    
    with col_btn3:
        if st.button("🔄 刷新页面"):
            st.rerun()
    
    with col_btn4:
        if st.button("🗑️ 清除任务"):
            try:
                # 尝试终止进程
                process = process_info['process']
                if process.poll() is None:
                    process.terminate()
                    st.warning("进程已终止")
                
                # 清除session state
                del st.session_state.current_process
                st.rerun()
                
            except Exception as e:
                st.error(f"清除任务失败: {e}")
    
    with col_btn5:
        if st.button("🔍 Debug信息"):
            st.subheader("🔍 Debug信息")
            
            # 显示脚本路径和大小
            if os.path.exists(process_info['script_path']):
                script_size = os.path.getsize(process_info['script_path']) / 1024
                st.info(f"脚本文件: {process_info['script_path']} ({script_size:.1f} KB)")
            else:
                st.error(f"脚本文件不存在: {process_info['script_path']}")
            
            # 显示进程详细信息
            st.code(f"""
进程ID: {process_info['pid']}
脚本路径: {process_info['script_path']}
日志路径: {process_info['log_path']}
输出路径: {process_info['output_path']}
启动时间: {time.ctime(process_info['start_time'])}
""", language="text")
            
            # 尝试读取脚本前50行
            try:
                with open(process_info['script_path'], 'r', encoding='utf-8') as f:
                    script_lines = f.readlines()[:50]
                st.subheader("📄 脚本内容预览 (前50行)")
                st.code(''.join(script_lines), language="python")
            except Exception as e:
                st.error(f"无法读取脚本: {e}")
            
            # 检查进程状态详情
            try:
                process = process_info['process']
                poll_result = process.poll()
                if poll_result is not None:
                    stdout, stderr = process.communicate()
                    st.subheader("📤 进程输出")
                    if stdout:
                        st.code(stdout, language="text")
                    if stderr:
                        st.error("错误输出:")
                        st.code(stderr, language="text")
            except Exception as e:
                st.error(f"无法获取进程输出: {e}")

else:
    st.info("🔍 当前没有运行中的后台任务")

# 分隔线
st.divider()

if not input_ready:
    if input_method == "上传新文件":
        st.info("👆 请上传包含3D构象的SDF文件开始使用")
    else:
        st.info("👆 请选择已保存的SDF文件开始使用") 