"""
CADD-Toolbox - 3D分子描述符生成页面
使用Mordred库计算3D描述符，支持多构象聚合
"""

import os
import sys
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import multiprocessing
import subprocess

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D, AllChem

# Mordred imports
try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False

st.set_page_config(page_title="生成3D描述符", layout="wide")
st.title("🧬 生成3D分子描述符")

# 初始化会话状态
if 'current_descriptor_process' not in st.session_state:
    st.session_state.current_descriptor_process = None

# 自动检查后台任务完成状态
if st.session_state.current_descriptor_process:
    process_info = st.session_state.current_descriptor_process
    try:
        process = process_info['process']
        poll_result = process.poll()
        
        # 如果进程已完成且输出文件存在，显示完成提示
        if poll_result is not None and os.path.exists(process_info['output_path']):
            file_size = os.path.getsize(process_info['output_path']) / (1024*1024)
            elapsed = time.time() - process_info['start_time']
            
            st.success(f"🎉 描述符计算已自动完成！耗时 {elapsed/60:.1f} 分钟")
            st.info(f"✅ 输出文件: {os.path.basename(process_info['output_path'])} ({file_size:.1f} MB)")
            
            # 自动提供下载按钮
            with open(process_info['output_path'], 'rb') as f:
                st.download_button(
                    "📥 立即下载描述符CSV文件",
                    f.read(),
                    file_name=os.path.basename(process_info['output_path']),
                    mime="text/csv",
                    type="primary"
                )
            
            st.markdown("---")
        
        elif poll_result is not None and poll_result != 0:
            st.error(f"❌ 后台计算任务出错! (退出码: {poll_result})")
            st.info("请查看下方监控区域的Debug信息了解详情")
            st.markdown("---")
    except:
        pass

st.markdown("""
从优化后的SDF文件计算分子的3D描述符。支持多构象聚合策略，输出CSV格式结果。

💊 **Mordred描述符库**: 1800+ 种2D/3D分子描述符  
🔄 **多构象聚合**: 平均值、最大值、最小值、标准差  
📊 **批量处理**: 支持大规模分子库处理  
📝 **CSV输出**: 便于后续机器学习分析  
""")

# 检查Mordred可用性
if not MORDRED_AVAILABLE:
    st.error("❌ 未安装Mordred库！请运行：pip install mordred")
    st.code("pip install mordred", language="bash")
    st.stop()
else:
    st.success("✅ Mordred库已就绪")

# 数据目录设置
DATA_DIR = "data"

def list_data_folders():
    """列出data目录下的所有文件夹"""
    if not os.path.exists(DATA_DIR):
        return []
    return [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]

def list_sdf_files_in_folder(folder_name):
    """列出指定文件夹中的所有SDF文件"""
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.endswith('.sdf')]

def get_file_info(file_path):
    """获取文件基本信息"""
    if not os.path.exists(file_path):
        return None
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    mod_time = os.path.getmtime(file_path)
    mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
    
    return {
        'size_mb': file_size,
        'modified': mod_time_str
    }

def count_molecules_in_sdf(file_path, max_count=10000):
    """快速统计SDF文件中的分子数量"""
    try:
        count = 0
        supplier = Chem.ForwardSDMolSupplier(file_path, removeHs=False, sanitize=False)
        
        for mol in supplier:
            if mol is not None:
                count += 1
            if count >= max_count:
                return f"{max_count}+"
                
        return count
    except Exception as e:
        return f"错误: {str(e)}"

def create_mordred_calculator(include_3d=True):
    """创建Mordred描述符计算器"""
    calc = Calculator(descriptors, ignore_3D=not include_3d)
    return calc



def calculate_molecule_descriptors(mol, calc, conf_ids=None):
    """计算单个分子的描述符"""
    if mol is None:
        return None
    
    try:
        if conf_ids is None:
            # 计算所有构象
            conf_ids = list(range(mol.GetNumConformers()))
        
        if not conf_ids:
            # 没有构象，尝试计算2D描述符
            try:
                result = calc(mol)
                
                # 处理描述符值，确保数据类型正确
                desc_values = []
                for desc_val in result.values():
                    try:
                        if desc_val is None:
                            desc_values.append(np.nan)
                        elif isinstance(desc_val, (int, float)):
                            if np.isfinite(desc_val):
                                desc_values.append(float(desc_val))
                            else:
                                desc_values.append(np.nan)
                        elif hasattr(desc_val, '__float__'):
                            # 尝试转换为浮点数
                            try:
                                float_val = float(desc_val)
                                desc_values.append(float_val if np.isfinite(float_val) else np.nan)
                            except (ValueError, TypeError, OverflowError):
                                desc_values.append(np.nan)
                        elif isinstance(desc_val, str):
                            # 字符串类型，尝试转换
                            if desc_val.lower() in ['nan', 'inf', '-inf', 'none', 'error']:
                                desc_values.append(np.nan)
                            else:
                                try:
                                    num_val = float(desc_val)
                                    desc_values.append(num_val if np.isfinite(num_val) else np.nan)
                                except ValueError:
                                    desc_values.append(np.nan)
                        else:
                            # 其他类型转为NaN
                            desc_values.append(np.nan)
                    except Exception:
                        desc_values.append(np.nan)
                
                return [desc_values]
            except Exception as e:
                st.warning(f"2D描述符计算失败: {e}")
                return None
        
        conformer_descriptors = []
        for conf_id in conf_ids:
            try:
                # 创建只包含当前构象的分子副本
                mol_copy = Chem.Mol(mol)
                conf = mol.GetConformer(conf_id)
                new_conf = Chem.Conformer(mol_copy.GetNumAtoms())
                for i in range(mol_copy.GetNumAtoms()):
                    new_conf.SetAtomPosition(i, conf.GetAtomPosition(i))
                mol_copy.AddConformer(new_conf, assignId=True)
                
                # 计算该构象的描述符
                result = calc(mol_copy)
                
                # 处理描述符值，确保数据类型正确
                desc_values = []
                for desc_val in result.values():
                    try:
                        if desc_val is None:
                            desc_values.append(np.nan)
                        elif isinstance(desc_val, (int, float)):
                            if np.isfinite(desc_val):
                                desc_values.append(float(desc_val))
                            else:
                                desc_values.append(np.nan)
                        elif hasattr(desc_val, '__float__'):
                            # 尝试转换为浮点数
                            try:
                                float_val = float(desc_val)
                                desc_values.append(float_val if np.isfinite(float_val) else np.nan)
                            except (ValueError, TypeError, OverflowError):
                                desc_values.append(np.nan)
                        elif isinstance(desc_val, str):
                            # 字符串类型，尝试转换
                            if desc_val.lower() in ['nan', 'inf', '-inf', 'none', 'error']:
                                desc_values.append(np.nan)
                            else:
                                try:
                                    num_val = float(desc_val)
                                    desc_values.append(num_val if np.isfinite(num_val) else np.nan)
                                except ValueError:
                                    desc_values.append(np.nan)
                        else:
                            # 其他类型转为NaN
                            desc_values.append(np.nan)
                    except Exception:
                        desc_values.append(np.nan)
                
                conformer_descriptors.append(desc_values)
            except Exception as e:
                st.warning(f"构象 {conf_id} 描述符计算失败: {e}")
                continue
        
        return conformer_descriptors if conformer_descriptors else None
        
    except Exception as e:
        st.error(f"分子描述符计算失败: {e}")
        return None

def generate_descriptor_script(input_file, output_file, include_3d, aggregation_method, include_smiles, num_workers, processing_limit):
    """生成独立的描述符计算脚本"""
    log_file_path = output_file.replace('.csv', '.log')
    
    # 先转换变量为字符串
    include_3d_str = str(include_3d)
    include_smiles_str = str(include_smiles)  
    num_workers_str = str(num_workers)
    # 特殊处理 processing_limit，如果是 float('inf') 则直接使用
    if processing_limit == float('inf'):
        processing_limit_str = "float('inf')"
    else:
        processing_limit_str = str(processing_limit)
    
    script_content = '''#!/usr/bin/env python3
"""
独立的多进程分子描述符计算脚本
由Streamlit应用自动生成
"""

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool

# RDKit imports
from rdkit import Chem

# Mordred imports
try:
    from mordred import Calculator, descriptors
    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False
    print("错误: 未找到Mordred库")
    sys.exit(1)

# 设置日志
log_file = "{log_file_path}"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, 'w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 配置参数
CONFIG = {
    'input_file': '{input_file}',
    'output_file': '{output_file}',
    'include_3d': {include_3d_str},
    'aggregation_method': '{aggregation_method}',
    'include_smiles': {include_smiles_str},
    'num_workers': {num_workers_str},
    'processing_limit': {processing_limit_str}
}

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
    logger.info("开始多进程分子描述符计算")
    logger.info("配置参数: %s", CONFIG)
    
    start_time = time.time()
    
    # 读取输入文件
    input_file = CONFIG['input_file']
    logger.info("读取输入文件: %s", input_file)
    
    supplier = Chem.ForwardSDMolSupplier(input_file, removeHs=False, sanitize=True)
    
    # 收集分子数据 - 按分子ID分组多个构象
    molecules_by_id = {}
    count = 0
    
    for mol in supplier:
        if CONFIG['processing_limit'] != float('inf') and count >= CONFIG['processing_limit']:
            break
        if mol is not None:
            # 获取分子ID/名称
            mol_id = mol.GetProp('_Name') if mol.HasProp('_Name') else "mol_" + str(count)
            if not mol_id.strip():  # 如果_Name为空
                mol_id = mol.GetProp('IDNUMBER') if mol.HasProp('IDNUMBER') else "mol_" + str(count)
            
            # 提取分子属性以避免MolBlock转换时丢失
            mol_props = {}
            for prop_name in mol.GetPropNames():
                mol_props[prop_name] = mol.GetProp(prop_name)
            
            # 按分子ID分组 - 合并同一分子的多个构象
            if mol_id not in molecules_by_id:
                molecules_by_id[mol_id] = {
                    'mol': Chem.Mol(mol),  # 创建分子副本
                    'props': mol_props,
                    'conformer_count': 0
                }
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
    
    logger.info("准备处理 %d 个独特分子", len(mol_data))
    total_conformers = sum(mol_info['conformer_count'] for mol_info in molecules_by_id.values())
    logger.info("总共 %d 个构象将被聚合", total_conformers)
    
    # 多进程计算
    num_workers = CONFIG['num_workers']
    logger.info("启动多进程计算 (%d 个进程)", num_workers)
    
    results = []
    if num_workers == 1:
        # 单进程
        for i, mol_args in enumerate(mol_data):
            result = calculate_molecule_descriptors_worker(mol_args)
            results.append(result)
            if (i + 1) % 50 == 0:
                logger.info("已处理: %d/%d", i + 1, len(mol_data))
    else:
        # 多进程
        with Pool(processes=num_workers) as pool:
            completed = 0
            for result in pool.imap_unordered(calculate_molecule_descriptors_worker, mol_data):
                results.append(result)
                completed += 1
                if completed % 50 == 0:
                    logger.info("已完成: %d/%d", completed, len(mol_data))
    
    # 整理结果
    all_descriptors = []
    all_smiles = []
    all_mol_ids = []
    
    # 获取描述符名称
    calc = create_mordred_calculator(include_3d=CONFIG['include_3d'])
    descriptor_names = [str(d) for d in calc.descriptors]
    
    success_count = 0
    for mol_id, smiles, desc in results:
        all_mol_ids.append(mol_id)  # 保留分子ID
        if desc is not None:
            all_descriptors.append(desc)
            all_smiles.append(smiles if smiles else "Invalid")
            success_count += 1
        else:
            all_descriptors.append([np.nan] * len(descriptor_names))
            all_smiles.append("Invalid")
    
    elapsed_time = time.time() - start_time
    logger.info("计算完成! 成功: %d/%d, 耗时: %.1f秒", success_count, len(mol_data), elapsed_time)
    
    # 创建DataFrame
    df_data = {}
    
    # 添加分子ID列
    df_data['Molecule_ID'] = all_mol_ids
    
    if CONFIG['include_smiles']:
        df_data['SMILES'] = all_smiles
    
    descriptors_array = np.array(all_descriptors)
    for i, desc_name in enumerate(descriptor_names):
        df_data[desc_name] = descriptors_array[:, i]
    
    df = pd.DataFrame(df_data)
    
    # 保存结果
    output_file = CONFIG['output_file']
    df.to_csv(output_file, index=False)
    
    logger.info("结果已保存到: %s", output_file)
    logger.info("输出形状: %s", str(df.shape))
    
    # 统计信息
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        valid_desc = df[numeric_cols].notna().sum().sum()
        total_desc = len(df) * len(numeric_cols)
        coverage = valid_desc / total_desc * 100 if total_desc > 0 else 0
        logger.info("有效描述符覆盖率: %.1f%%", coverage)

if __name__ == "__main__":
    main()
'''
    
    # 替换占位符
    script_content = script_content.replace('{input_file}', input_file)
    script_content = script_content.replace('{output_file}', output_file)
    script_content = script_content.replace('{log_file_path}', log_file_path)
    script_content = script_content.replace('{include_3d_str}', include_3d_str)
    script_content = script_content.replace('{aggregation_method}', aggregation_method)
    script_content = script_content.replace('{include_smiles_str}', include_smiles_str)
    script_content = script_content.replace('{num_workers_str}', num_workers_str)
    script_content = script_content.replace('{processing_limit_str}', processing_limit_str)
    
    return script_content

def save_descriptor_results(all_descriptors, all_smiles_final, descriptor_names, include_smiles, output_path, mol_ids=None):
    """保存描述符计算结果"""
    # 创建DataFrame
    with st.spinner("组织结果数据..."):
        df_data = {}
        
        # 添加分子ID列（如果有）
        if mol_ids:
            df_data['Molecule_ID'] = mol_ids[:len(all_descriptors)]
        
        # 添加SMILES列（如果需要）
        if include_smiles:
            df_data['SMILES'] = all_smiles_final[:len(all_descriptors)]
        
        # 添加描述符列
        descriptors_array = np.array(all_descriptors)
        for i, desc_name in enumerate(descriptor_names):
            df_data[desc_name] = descriptors_array[:, i]
        
        df = pd.DataFrame(df_data)
    
    # 显示数据摘要
    st.subheader("4. 结果摘要")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("处理分子数", len(df))
    with col2:
        st.metric("描述符数量", len(descriptor_names))
    with col3:
        valid_desc = df.select_dtypes(include=[np.number]).notna().sum().sum()
        total_desc = len(df) * len(descriptor_names)
        coverage = valid_desc / total_desc * 100 if total_desc > 0 else 0
        st.metric("有效描述符覆盖率", f"{coverage:.1f}%")
    
    # 保存结果
    with st.spinner("保存结果..."):
        df.to_csv(output_path, index=False)
    
    st.success(f"✅ 结果已保存到: {output_path}")
    
    # 提供下载按钮
    csv_data = df.to_csv(index=False)
    st.download_button(
        label="📥 下载描述符CSV文件",
        data=csv_data,
        file_name=os.path.basename(output_path),
        mime="text/csv"
    )
    
    # 数据预览
    st.subheader("5. 数据预览")
    
    # 基本统计
    st.markdown("**描述符统计信息:**")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        st.dataframe(df[numeric_cols].describe())
    
    # 数据表预览
    st.markdown("**数据表预览（前10行）:**")
    st.dataframe(df.head(10))
    
    # 显示文件大小
    output_size = len(csv_data.encode('utf-8')) / (1024 * 1024)
    st.info(f"输出文件大小: {output_size:.2f} MB")

def aggregate_conformer_descriptors(conformer_descriptors, method="mean"):
    """聚合多个构象的描述符"""
    if not conformer_descriptors:
        return None
    
    try:
        # 预处理：将所有值转换为数值类型
        processed_descriptors = []
        for conf_desc in conformer_descriptors:
            processed_conf = []
            for desc_value in conf_desc:
                try:
                    # 尝试转换为浮点数
                    if desc_value is None:
                        processed_conf.append(np.nan)
                    elif isinstance(desc_value, (int, float)):
                        if np.isfinite(desc_value):
                            processed_conf.append(float(desc_value))
                        else:
                            processed_conf.append(np.nan)
                    elif isinstance(desc_value, str):
                        # 字符串类型，尝试转换
                        if desc_value.lower() in ['nan', 'inf', '-inf', 'none', 'error']:
                            processed_conf.append(np.nan)
                        else:
                            try:
                                num_val = float(desc_value)
                                processed_conf.append(num_val if np.isfinite(num_val) else np.nan)
                            except ValueError:
                                processed_conf.append(np.nan)
                    else:
                        # 其他类型，转为NaN
                        processed_conf.append(np.nan)
                except Exception:
                    processed_conf.append(np.nan)
            
            processed_descriptors.append(processed_conf)
        
        # 转换为numpy数组
        descriptors_array = np.array(processed_descriptors, dtype=float)
        
        # 聚合计算
        if method == "mean":
            result = np.nanmean(descriptors_array, axis=0)
        elif method == "max":
            result = np.nanmax(descriptors_array, axis=0)
        elif method == "min":
            result = np.nanmin(descriptors_array, axis=0)
        elif method == "std":
            result = np.nanstd(descriptors_array, axis=0)
        elif method == "median":
            result = np.nanmedian(descriptors_array, axis=0)
        else:
            result = np.nanmean(descriptors_array, axis=0)
        
        return result
        
    except Exception as e:
        st.error(f"聚合描述符时出错: {e}")
        # 返回NaN数组作为备选
        if conformer_descriptors and len(conformer_descriptors) > 0:
            return np.full(len(conformer_descriptors[0]), np.nan)
        else:
            return None

# 文件选择界面
st.subheader("1. 选择输入文件")

# 获取可用文件夹
folders = list_data_folders()

if not folders:
    st.warning("data目录下没有找到任何文件夹")
else:
    selected_folder = st.selectbox("选择数据文件夹:", folders)
    
    if selected_folder:
        # 获取该文件夹中的SDF文件
        sdf_files = list_sdf_files_in_folder(selected_folder)
        
        if not sdf_files:
            st.warning(f"文件夹 {selected_folder} 中没有SDF文件")
        else:
            selected_file = st.selectbox("选择SDF文件:", sdf_files)
            
            if selected_file:
                file_path = os.path.join(DATA_DIR, selected_folder, selected_file)
                
                # 显示文件信息
                file_info = get_file_info(file_path)
                if file_info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("文件大小", f"{file_info['size_mb']:.1f} MB")
                    with col2:
                        st.metric("修改时间", file_info['modified'])
                    with col3:
                        with st.spinner("统计分子数..."):
                            mol_count = count_molecules_in_sdf(file_path)
                        st.metric("分子数量", mol_count)
                
                # 描述符计算配置
                st.subheader("2. 描述符计算配置")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**描述符类型**")
                    include_3d = st.checkbox("包含3D描述符", value=True, 
                                           help="需要分子具有3D坐标")
                    
                    st.markdown("**处理范围**")
                    processing_option = st.selectbox(
                        "选择处理范围:",
                        ["处理所有分子", "仅处理前1000个分子（测试用）"],
                        index=0,
                        help="默认处理所有分子，支持百万级化合物库；测试选项用于快速验证"
                    )
                    
                    if processing_option == "仅处理前1000个分子（测试用）":
                        processing_limit = 1000
                        st.info(f"🧪 测试模式: 仅处理前1000个分子")
                    else:
                        processing_limit = float('inf')  # 不设限制，处理所有分子
                        st.info(f"🚀 生产模式: 处理文件中的所有分子（支持百万级）")
                
                with col2:
                    st.markdown("**多构象聚合策略**")
                    aggregation_method = st.selectbox(
                        "聚合方法:",
                        ["mean", "max", "min", "std", "median"],
                        help="如何聚合同一分子多个构象的描述符"
                    )
                    
                    include_smiles = st.checkbox("包含SMILES", value=True,
                                               help="在输出中包含分子SMILES")
                
                # 执行配置
                st.subheader("3. 执行配置")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**计算模式**")
                    execution_mode = st.selectbox(
                        "计算方式:",
                        ["多进程后台执行 (推荐)", "Streamlit内串行执行"],
                        index=0,
                        help="多进程模式将生成独立脚本运行，避免Streamlit限制"
                    )
                
                with col2:
                    st.markdown("**并行配置**")
                    if execution_mode == "多进程后台执行 (推荐)":
                        max_workers = multiprocessing.cpu_count()
                        num_workers = st.number_input(
                            "并行进程数", 
                            min_value=1, 
                            max_value=max_workers * 2,  # 允许超线程
                            value=34,
                            step=1,
                            help=f"系统有{max_workers}个CPU核心，推荐34进程"
                        )
                        st.info(f"🚀 将使用{num_workers}个进程")
                    else:
                        num_workers = 1
                        st.info("🔄 单线程串行执行")
                
                # 输出配置
                st.subheader("4. 输出配置")
                
                output_filename = st.text_input(
                    "输出文件名", 
                    value=f"descriptors_{selected_file.replace('.sdf', '')}.csv"
                )
                
                # 计算按钮
                if st.button("🚀 开始计算描述符", type="primary"):
                    # 检查是否已有正在运行的任务
                    if st.session_state.current_descriptor_process:
                        process_info = st.session_state.current_descriptor_process
                        try:
                            process = process_info['process']
                            if process.poll() is None:  # 进程仍在运行
                                st.warning(f"⚠️ 已有任务正在运行中 (PID: {process_info['pid']})！")
                                st.info("请在下方监控区域查看进度，或先清除当前任务再启动新任务。")
                                st.stop()
                        except:
                            # 如果检查失败，清除旧的进程信息
                            st.session_state.current_descriptor_process = None
                    
                    st.info(f"开始处理文件: {file_path}")
                    
                    # 创建输出路径
                    output_path = os.path.join(DATA_DIR, selected_folder, output_filename)
                    
                    try:
                        # 创建Mordred计算器
                        with st.spinner("初始化Mordred计算器..."):
                            calc = create_mordred_calculator(include_3d=include_3d)
                            descriptor_names = [str(d) for d in calc.descriptors]
                        
                        st.success(f"✅ 计算器就绪，共 {len(descriptor_names)} 个描述符")
                        
                        # 读取SDF文件并按分子ID分组构象
                        with st.spinner("读取SDF文件并按分子ID分组..."):
                            supplier = Chem.ForwardSDMolSupplier(file_path, removeHs=False, sanitize=True)
                            molecules_by_id = {}
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            count = 0
                            for mol in supplier:
                                if processing_limit != float('inf') and count >= processing_limit:
                                    break
                                
                                if mol is not None:
                                    # 获取分子ID/名称
                                    mol_id = mol.GetProp('_Name') if mol.HasProp('_Name') else f"mol_{count}"
                                    if not mol_id.strip():  # 如果_Name为空
                                        mol_id = mol.GetProp('IDNUMBER') if mol.HasProp('IDNUMBER') else f"mol_{count}"
                                    
                                    # 提取分子属性以避免MolBlock转换时丢失
                                    mol_props = {prop_name: mol.GetProp(prop_name) for prop_name in mol.GetPropNames()}
                                    
                                    # 按分子ID分组 - 合并同一分子的多个构象
                                    if mol_id not in molecules_by_id:
                                        molecules_by_id[mol_id] = {
                                            'mol': Chem.Mol(mol),  # 创建分子副本
                                            'props': mol_props,
                                            'conformer_count': 0
                                        }
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
                                    
                                    if count % 100 == 0:
                                        progress = min(count / processing_limit, 1.0) if processing_limit != float('inf') else count / 10000
                                        progress_bar.progress(min(progress, 1.0))
                                        status_text.text(f"已读取 {count} 个分子，分组为 {len(molecules_by_id)} 个独特分子...")
                            
                            # 转换为列表格式供后续处理
                            molecules = list(molecules_by_id.values())
                            smiles_list = []
                            
                            # 为每个分子生成SMILES
                            if include_smiles:
                                for mol_info in molecules:
                                    try:
                                        smiles = Chem.MolToSmiles(mol_info['mol'])
                                        smiles_list.append(smiles)
                                    except:
                                        smiles_list.append("Invalid")
                        
                        total_conformers = sum(mol_info['conformer_count'] for mol_info in molecules)
                        st.success(f"✅ 成功读取并分组：{len(molecules)} 个独特分子，共 {total_conformers} 个构象")
                        
                        # 根据选择的执行方式进行计算
                        if execution_mode == "多进程后台执行 (推荐)":
                            st.subheader("5. 多进程后台计算")
                            st.info(f"🚀 使用{num_workers}个进程并行计算{len(molecules)}个分子的描述符")
                            
                                                        # 自动启动后台计算
                            st.info("🔄 正在启动后台计算...")
                            
                            # 首先生成绝对路径
                            abs_input_path = os.path.abspath(file_path)
                            abs_output_path = os.path.abspath(output_path)
                            
                            try:
                                # 首先生成绝对路径
                                abs_input_path = os.path.abspath(file_path)
                                abs_output_path = os.path.abspath(output_path)
                                    

                                
                                # 处理processing_limit参数
                                if processing_limit == float('inf'):
                                    processing_limit_str = "float('inf')"
                                else:
                                    processing_limit_str = str(int(processing_limit))
                                
                                script_content = generate_descriptor_script(
                                    abs_input_path, abs_output_path, include_3d, 
                                    aggregation_method, include_smiles,
                                    num_workers, processing_limit_str
                                )
                                
                                # 保存脚本
                                script_path = os.path.join(DATA_DIR, selected_folder, "descriptor_calculation.py")
                                st.info(f"📄 保存脚本到: {script_path}")
                                
                                with open(script_path, 'w', encoding='utf-8') as f:
                                    f.write(script_content)
                                
                                if os.path.exists(script_path):
                                    script_size = os.path.getsize(script_path) / 1024
                                    st.success(f"✅ 脚本已保存 ({script_size:.1f} KB)")
                                else:
                                    st.error("❌ 脚本保存失败")
                                    st.stop()
                                
                                # 启动后台进程
                                import subprocess
                                # 日志文件路径要与脚本中生成的路径一致
                                log_path = abs_output_path.replace('.csv', '.log')
                                
                                # 使用绝对路径避免路径问题
                                abs_script_path = os.path.abspath(script_path)
                                # 工作目录设置为项目根目录，避免路径重复
                                abs_work_dir = os.path.abspath(".")  # 当前工作目录（项目根目录）
                                
                                cmd = ["python", abs_script_path]
                                st.info(f"🚀 启动命令: {' '.join(cmd)}")
                                st.info(f"📁 工作目录: {abs_work_dir}")
                                st.info(f"📄 脚本绝对路径: {abs_script_path}")
                                st.info(f"📂 输入文件绝对路径: {abs_input_path}")
                                st.info(f"📤 输出文件绝对路径: {abs_output_path}")
                                st.info(f"📋 日志文件绝对路径: {log_path}")
                                
                                # 确认脚本文件存在
                                if not os.path.exists(abs_script_path):
                                    st.error(f"❌ 脚本文件不存在: {abs_script_path}")
                                    # 显示目录内容用于调试
                                    script_dir = os.path.dirname(abs_script_path)
                                    if os.path.exists(script_dir):
                                        files = os.listdir(script_dir)
                                        st.info(f"目录 {script_dir} 中的文件: {files}")
                                    st.stop()
                                else:
                                    st.success(f"✅ 脚本文件存在: {abs_script_path}")
                                
                                process = subprocess.Popen(
                                    cmd, 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.STDOUT,
                                    cwd=abs_work_dir,
                                    text=True
                                )
                                
                                st.info(f"⏳ 进程已启动，PID: {process.pid}")
                                
                                # 等待短暂时间检查进程状态
                                time.sleep(1)
                                poll_result = process.poll()
                                
                                if poll_result is not None:
                                    st.error(f"❌ 进程启动后立即退出 (退出码: {poll_result})")
                                    try:
                                        stdout, stderr = process.communicate(timeout=5)
                                        if stdout:
                                            st.error("标准输出:")
                                            st.code(stdout, language="text")
                                        if stderr:
                                            st.error("错误输出:")
                                            st.code(stderr, language="text")
                                    except subprocess.TimeoutExpired:
                                        st.warning("获取进程输出超时")
                                    st.stop()
                                else:
                                    st.success(f"✅ 进程正在运行中，PID: {process.pid}")
                                
                                st.success(f"🎉 后台计算已启动！进程ID: {process.pid}")
                                st.info(f"📄 脚本已保存: {script_path}")
                                st.info(f"📋 日志文件: {log_path}")
                                
                                # 保存进程信息到session state
                                process_info = {
                                    'pid': process.pid,
                                    'script_path': script_path,
                                    'log_path': log_path,
                                    'output_path': output_path,
                                    'start_time': time.time(),
                                    'process': process,
                                    'num_workers': num_workers,
                                    'num_molecules': len(molecules)
                                }
                                
                                st.session_state.current_descriptor_process = process_info
                                
                                # Debug: 确认设置成功
                                st.info("📋 进程信息已保存到会话状态")
                                st.code(f"保存的PID: {st.session_state.current_descriptor_process['pid']}")
                                
                                st.warning("⚠️ 请勿关闭此页面，可以使用下方的监控功能查看进度")
                                
                                with st.expander("查看脚本内容"):
                                    st.code(script_content, language="python")
                            
                            except Exception as e:
                                st.error(f"❌ 启动后台计算失败: {e}")
                                st.code(f"错误详情: {str(e)}", language="text")
                                import traceback
                                st.code(traceback.format_exc(), language="text")
                        
                        else:
                            # 串行处理模式
                            st.subheader("5. 串行计算")
                            st.info("🔄 在Streamlit内进行单线程串行计算")
                            
                            # 自动开始串行计算
                            # 计算描述符
                            all_descriptors = []
                            all_smiles_final = []
                            all_mol_ids = []
                            start_time = time.time()
                            
                            st.info("🔄 串行计算模式...")
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # 提取分子ID
                            mol_ids = list(molecules_by_id.keys())
                            
                            for i, mol_info in enumerate(molecules):
                                mol = mol_info['mol']
                                mol_id = mol_ids[i] if i < len(mol_ids) else f"mol_{i}"
                                status_text.text(f"计算描述符: {i+1}/{len(molecules)} (构象数: {mol_info['conformer_count']})")
                                
                                # 恢复分子属性
                                for prop_name, prop_value in mol_info['props'].items():
                                    mol.SetProp(prop_name, prop_value)
                                
                                # 保存分子ID
                                all_mol_ids.append(mol_id)
                                
                                # 计算该分子的描述符
                                conformer_descriptors = calculate_molecule_descriptors(mol, calc)
                                
                                if conformer_descriptors:
                                    # 聚合多构象描述符
                                    aggregated_desc = aggregate_conformer_descriptors(
                                        conformer_descriptors, 
                                        method=aggregation_method
                                    )
                                    all_descriptors.append(aggregated_desc)
                                else:
                                    # 使用NaN填充
                                    all_descriptors.append([np.nan] * len(descriptor_names))
                                
                                # 处理SMILES（使用预先计算的SMILES）
                                if include_smiles:
                                    if i < len(smiles_list):
                                        all_smiles_final.append(smiles_list[i])
                                    else:
                                        try:
                                            smiles = Chem.MolToSmiles(mol) if mol else "Invalid"
                                            all_smiles_final.append(smiles)
                                        except:
                                            all_smiles_final.append("Invalid")
                                
                                # 更新进度
                                progress = (i + 1) / len(molecules)
                                progress_bar.progress(progress)
                            
                            elapsed_time = time.time() - start_time
                            st.success(f"✅ 描述符计算完成！耗时: {elapsed_time:.1f}秒")
                            
                            # 创建DataFrame并保存结果
                            save_descriptor_results(all_descriptors, all_smiles_final, descriptor_names, 
                                                   include_smiles, output_path, all_mol_ids)
                    
                    except Exception as e:
                        st.error(f"处理过程中出错: {e}")
                        import traceback
                        st.code(traceback.format_exc(), language="text")

# 后台任务监控区域
st.header("🔍 后台任务监控")

if st.session_state.current_descriptor_process:
    process_info = st.session_state.current_descriptor_process
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("进程ID", process_info['pid'])
    
    with col2:
        elapsed = time.time() - process_info['start_time']
        st.metric("运行时间", f"{elapsed/60:.1f} 分钟")
    
    with col3:
        st.metric("进程数", process_info['num_workers'])
    
    with col4:
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
    
    # 显示处理进度信息
    st.info(f"🧬 正在计算 {process_info['num_molecules']} 个分子的描述符")
    
    # 控制按钮
    col_btn1, col_btn2, col_btn3, col_btn4, col_btn5 = st.columns(5)
    
    with col_btn1:
        if st.button("📖 查看日志"):
            log_path = process_info['log_path']
            st.info(f"🔍 查找日志文件: {log_path}")
            
            if os.path.exists(log_path):
                try:
                    file_size = os.path.getsize(log_path) / 1024  # KB
                    st.success(f"✅ 找到日志文件 ({file_size:.1f} KB)")
                    
                    with open(log_path, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    
                    st.subheader("📝 实时日志")
                    # 只显示最后50行
                    log_lines = log_content.split('\n')
                    recent_logs = '\n'.join(log_lines[-50:])
                    st.code(recent_logs, language="text")
                    
                    # 提供日志文件下载
                    st.download_button(
                        "📥 下载完整日志文件",
                        log_content,
                        file_name=os.path.basename(log_path),
                        mime="text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"读取日志失败: {e}")
            else:
                st.warning(f"⚠️ 日志文件尚未生成: {log_path}")
                
                # 检查同目录下是否有其他日志文件
                log_dir = os.path.dirname(log_path)
                if os.path.exists(log_dir):
                    files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
                    if files:
                        st.info(f"📁 目录中的日志文件: {files}")
                    else:
                        st.info("📁 目录中暂无.log文件")
    
    with col_btn2:
        if st.button("📊 检查完成"):
            # 检查输出文件是否生成
            if os.path.exists(process_info['output_path']):
                file_size = os.path.getsize(process_info['output_path']) / (1024*1024)
                st.success("🎉 描述符计算已完成!")
                st.info(f"✅ 输出文件已生成: {process_info['output_path']} ({file_size:.1f} MB)")
                
                # 提供下载按钮
                with open(process_info['output_path'], 'rb') as f:
                    st.download_button(
                        "📥 下载描述符CSV文件",
                        f.read(),
                        file_name=os.path.basename(process_info['output_path']),
                        mime="text/csv"
                    )
                
                # 显示基本统计信息
                try:
                    df = pd.read_csv(process_info['output_path'])
                    st.markdown("**结果统计:**")
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("分子数", len(df))
                    with col_stat2:
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        st.metric("描述符数", len(numeric_cols))
                    with col_stat3:
                        if len(numeric_cols) > 0:
                            valid_desc = df[numeric_cols].notna().sum().sum()
                            total_desc = len(df) * len(numeric_cols)
                            coverage = valid_desc / total_desc * 100 if total_desc > 0 else 0
                            st.metric("有效覆盖率", f"{coverage:.1f}%")
                    
                    # 数据预览
                    st.markdown("**数据预览 (前5行):**")
                    st.dataframe(df.head(5))
                    
                except Exception as e:
                    st.warning(f"无法读取结果文件进行预览: {e}")
                    
            else:
                # 检查进程状态
                process = process_info['process']
                poll_result = process.poll()
                if poll_result is None:
                    st.info("⏳ 任务仍在运行中...")
                elif poll_result == 0:
                    st.warning("⚠️ 进程已完成但输出文件未找到")
                else:
                    st.error(f"❌ 计算任务出错! (退出码: {poll_result})")
                    # 显示进程输出
                    try:
                        stdout, stderr = process.communicate()
                        if stderr:
                            st.error("错误输出:")
                            st.code(stderr, language="text")
                    except:
                        pass
    
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
                st.session_state.current_descriptor_process = None
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
进程数: {process_info['num_workers']}
分子数: {process_info['num_molecules']}
""", language="text")
            
            # 尝试读取脚本前30行
            try:
                with open(process_info['script_path'], 'r', encoding='utf-8') as f:
                    script_lines = f.readlines()[:30]
                st.subheader("📄 脚本内容预览 (前30行)")
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
    st.info("🔍 当前没有运行中的后台描述符计算任务")
    
    # Debug信息
    with st.expander("🔍 Debug: Session State 调试"):
        st.markdown("**当前会话状态:**")
        st.code(f"current_descriptor_process: {st.session_state.current_descriptor_process}")
        
        st.markdown("**所有Session State键:**")
        st.json(list(st.session_state.keys()))
        
        if hasattr(st.session_state, '__dict__'):
            st.markdown("**Session State详情:**")
            state_dict = {}
            for key in st.session_state.keys():
                try:
                    value = st.session_state[key]
                    if key == 'current_descriptor_process' and value:
                        # 特殊处理进程对象
                        state_dict[key] = {
                            'pid': value.get('pid', 'N/A'),
                            'start_time': value.get('start_time', 'N/A'),
                            'script_path': value.get('script_path', 'N/A'),
                            'log_path': value.get('log_path', 'N/A'),
                            'output_path': value.get('output_path', 'N/A'),
                            'num_workers': value.get('num_workers', 'N/A'),
                            'num_molecules': value.get('num_molecules', 'N/A'),
                            'process_type': str(type(value.get('process', 'N/A')))
                        }
                    else:
                        state_dict[key] = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                except Exception as e:
                    state_dict[key] = f"Error: {e}"
            st.json(state_dict)
    
    if st.button("🧹 清理可能的僵尸进程"):
        st.info("已检查，无需清理")

# 分隔线
st.divider()

# 使用说明
with st.expander("📖 使用说明", expanded=False):
    st.markdown("""
    ### 功能说明
    
    1. **文件选择**: 从data目录中选择已优化的SDF文件
    2. **描述符配置**: 选择3D描述符和聚合策略
    3. **执行配置**: 选择计算模式和进程数
    4. **输出配置**: 设置输出文件名
    5. **描述符计算**: 使用Mordred库计算1800+种分子描述符
    6. **多构象处理**: 对同一分子的多个构象进行聚合
    7. **结果输出**: 生成CSV格式的描述符矩阵
    
    ### 执行模式说明
    
    - **多进程后台执行**: 生成独立Python脚本，使用多进程并行计算（推荐，默认34进程）
    - **Streamlit内串行执行**: 在Streamlit内单线程执行，适合小规模测试
    
    ### 处理范围说明
    
    - **处理所有分子**: 生产模式，无数量限制，支持百万级化合物库
    - **仅处理前1000个分子（测试用）**: 快速验证功能和参数设置
    
    ### 聚合策略说明
    
    - **mean**: 平均值（推荐）
    - **max**: 最大值
    - **min**: 最小值  
    - **std**: 标准差
    - **median**: 中位数
    
    ### 输出格式
    
    生成的CSV文件包含：
    - SMILES列（可选）
    - 每个描述符一列
    - 每行对应一个分子
    
    ### 后台任务监控
    
    - **实时监控**: 查看进程状态、运行时间和进度
    - **日志查看**: 实时查看计算日志，了解详细进度
    - **完成检查**: 自动检测任务完成，提供结果下载
    - **进程管理**: 支持清除和调试后台任务
    
    ### 注意事项
    
    - 确保输入的SDF文件包含3D坐标
    - **百万级化合物库**: 支持大规模处理，建议使用34进程
    - **内存使用**: 大文件可能需要较多内存，监控系统资源
    - **测试建议**: 首次使用建议先用测试模式验证
    - 多进程模式需要保持页面打开以监控进度
    - **处理时间**: 百万级数据可能需要数小时，可通过日志监控进度
    """) 