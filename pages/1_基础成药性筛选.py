"""
CADD-Toolbox - 基础成药性筛选页面
基于SMILES进行成药性筛选，包含多种常见的筛选指标
"""
import os
import pandas as pd
import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.QED import qed
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import subprocess
import time
import tempfile
import shutil

# 设置页面配置
st.set_page_config(
    page_title="基础成药性筛选",
    page_icon="💊",
    layout="wide"
)

st.title("💊 基础成药性筛选")

# 数据目录设置
DATA_DIR = os.path.abspath("data")

def list_data_folders():
    """列出data目录下的所有文件夹"""
    if not os.path.exists(DATA_DIR):
        return []
    folders = []
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return sorted(folders)

def list_csv_files_in_folder(folder_name):
    """列出指定文件夹中的所有CSV文件"""
    if not folder_name:
        return []
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        return []
    files = []
    for item in os.listdir(folder_path):
        if item.endswith('.csv') and os.path.isfile(os.path.join(folder_path, item)):
            files.append(item)
    return sorted(files)

def calculate_druglike_properties(smiles):
    """计算成药性相关属性"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = {}
    
    # 基本分子描述符
    properties['MolWt'] = Descriptors.MolWt(mol)
    properties['LogP'] = Descriptors.MolLogP(mol)
    properties['HBD'] = Descriptors.NumHDonors(mol)
    properties['HBA'] = Descriptors.NumHAcceptors(mol)
    properties['TPSA'] = Descriptors.TPSA(mol)
    properties['RotBonds'] = Descriptors.NumRotatableBonds(mol)
    properties['AromaticRings'] = Descriptors.NumAromaticRings(mol)
    properties['HeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
    if hasattr(Descriptors, 'FractionCSP3'):
        properties['FractionCsp3'] = Descriptors.FractionCSP3(mol)
    else:
        properties['FractionCsp3'] = rdMolDescriptors.CalcFractionCSP3(mol)
    properties['MolMR'] = Descriptors.MolMR(mol)
    
    # 复杂度和形状描述符
    properties['BertzCT'] = Descriptors.BertzCT(mol)
    properties['Kappa1'] = Descriptors.Kappa1(mol)
    properties['Kappa2'] = Descriptors.Kappa2(mol)
    properties['Kappa3'] = Descriptors.Kappa3(mol)
    
    # 类药性评分
    properties['QED'] = qed(mol)
    
    # 其他重要描述符
    properties['NumRings'] = Descriptors.RingCount(mol)
    properties['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
    properties['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
    properties['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
    
    # 极性和电荷相关
    properties['LabuteASA'] = Descriptors.LabuteASA(mol)
    properties['PEOE_VSA1'] = Descriptors.PEOE_VSA1(mol)
    properties['PEOE_VSA2'] = Descriptors.PEOE_VSA2(mol)
    
    return properties

def check_lipinski_rule(properties):
    """检查Lipinski规则"""
    violations = 0
    rules = {}
    
    # 分子量 ≤ 500 Da
    rules['MW_Rule'] = properties['MolWt'] <= 500
    if not rules['MW_Rule']:
        violations += 1
    
    # LogP ≤ 5
    rules['LogP_Rule'] = properties['LogP'] <= 5
    if not rules['LogP_Rule']:
        violations += 1
    
    # 氢键供体 ≤ 5
    rules['HBD_Rule'] = properties['HBD'] <= 5
    if not rules['HBD_Rule']:
        violations += 1
    
    # 氢键受体 ≤ 10
    rules['HBA_Rule'] = properties['HBA'] <= 10
    if not rules['HBA_Rule']:
        violations += 1
    
    rules['Lipinski_Violations'] = violations
    rules['Lipinski_Pass'] = violations <= 1  # 通常允许1个违规
    
    return rules

def check_veber_rule(properties):
    """检查Veber规则"""
    rules = {}
    
    # 旋转键 ≤ 10
    rules['RotBonds_Rule'] = properties['RotBonds'] <= 10
    
    # TPSA ≤ 140 Ų
    rules['TPSA_Rule'] = properties['TPSA'] <= 140
    
    rules['Veber_Pass'] = rules['RotBonds_Rule'] and rules['TPSA_Rule']
    
    return rules

def check_egan_rule(properties):
    """检查Egan规则"""
    rules = {}
    
    # LogP: -1 to 6
    rules['LogP_Egan'] = -1 <= properties['LogP'] <= 6
    
    # TPSA: 0 to 132
    rules['TPSA_Egan'] = 0 <= properties['TPSA'] <= 132
    
    rules['Egan_Pass'] = rules['LogP_Egan'] and rules['TPSA_Egan']
    
    return rules

def check_muegge_rule(properties):
    """检查Muegge规则"""
    rules = {}
    violations = 0
    
    # 分子量: 200-600 Da
    rules['MW_Muegge'] = 200 <= properties['MolWt'] <= 600
    if not rules['MW_Muegge']:
        violations += 1
    
    # LogP: -2 to 5
    rules['LogP_Muegge'] = -2 <= properties['LogP'] <= 5
    if not rules['LogP_Muegge']:
        violations += 1
    
    # TPSA ≤ 150
    rules['TPSA_Muegge'] = properties['TPSA'] <= 150
    if not rules['TPSA_Muegge']:
        violations += 1
    
    # 环数: 0-7
    rules['Rings_Muegge'] = 0 <= properties['NumRings'] <= 7
    if not rules['Rings_Muegge']:
        violations += 1
    
    # 重原子数: 4-15
    rules['HeavyAtoms_Muegge'] = 4 <= properties['HeavyAtoms'] <= 15
    if not rules['HeavyAtoms_Muegge']:
        violations += 1
    
    # 旋转键 ≤ 15
    rules['RotBonds_Muegge'] = properties['RotBonds'] <= 15
    if not rules['RotBonds_Muegge']:
        violations += 1
    
    # HBD ≤ 5
    rules['HBD_Muegge'] = properties['HBD'] <= 5
    if not rules['HBD_Muegge']:
        violations += 1
    
    # HBA ≤ 10
    rules['HBA_Muegge'] = properties['HBA'] <= 10
    if not rules['HBA_Muegge']:
        violations += 1
    
    rules['Muegge_Violations'] = violations
    rules['Muegge_Pass'] = violations == 0
    
    return rules

def apply_custom_filters(df, filters):
    """应用自定义筛选条件"""
    mask = pd.Series([True] * len(df))
    
    for prop, (min_val, max_val) in filters.items():
        if prop in df.columns:
            if min_val is not None:
                mask &= (df[prop] >= min_val)
            if max_val is not None:
                mask &= (df[prop] <= max_val)
    
    return df[mask]

def create_property_distribution_plot(df, property_name):
    """创建属性分布图"""
    if property_name not in df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 直方图
    ax.hist(df[property_name].dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_xlabel(property_name)
    ax.set_ylabel('频数')
    ax.set_title(f'{property_name} 分布')
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    mean_val = df[property_name].mean()
    median_val = df[property_name].median()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.2f}')
    ax.axvline(median_val, color='green', linestyle='--', label=f'中位数: {median_val:.2f}')
    ax.legend()
    
    plt.tight_layout()
    return fig

# ====================== 分块并行处理函数 ======================

def generate_druglike_script():
    """生成成药性计算的Python脚本模板"""
    script_content = '''
import os
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem.QED import qed

def calculate_druglike_properties(smiles):
    """计算成药性相关属性"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    properties = {}
    
    # 基本分子描述符
    properties['MolWt'] = Descriptors.MolWt(mol)
    properties['LogP'] = Descriptors.MolLogP(mol)
    properties['HBD'] = Descriptors.NumHDonors(mol)
    properties['HBA'] = Descriptors.NumHAcceptors(mol)
    properties['TPSA'] = Descriptors.TPSA(mol)
    properties['RotBonds'] = Descriptors.NumRotatableBonds(mol)
    properties['AromaticRings'] = Descriptors.NumAromaticRings(mol)
    properties['HeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
    if hasattr(Descriptors, 'FractionCSP3'):
        properties['FractionCsp3'] = Descriptors.FractionCSP3(mol)
    else:
        properties['FractionCsp3'] = rdMolDescriptors.CalcFractionCSP3(mol)
    properties['MolMR'] = Descriptors.MolMR(mol)
    
    # 复杂度和形状描述符
    properties['BertzCT'] = Descriptors.BertzCT(mol)
    properties['Kappa1'] = Descriptors.Kappa1(mol)
    properties['Kappa2'] = Descriptors.Kappa2(mol)
    properties['Kappa3'] = Descriptors.Kappa3(mol)
    
    # 类药性评分
    properties['QED'] = qed(mol)
    
    # 其他重要描述符
    properties['NumRings'] = Descriptors.RingCount(mol)
    properties['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
    properties['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
    properties['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
    
    # 极性和电荷相关
    properties['LabuteASA'] = Descriptors.LabuteASA(mol)
    properties['PEOE_VSA1'] = Descriptors.PEOE_VSA1(mol)
    properties['PEOE_VSA2'] = Descriptors.PEOE_VSA2(mol)
    
    return properties

def check_lipinski_rule(properties):
    """检查Lipinski规则"""
    violations = 0
    rules = {}
    
    rules['MW_Rule'] = properties['MolWt'] <= 500
    if not rules['MW_Rule']:
        violations += 1
    
    rules['LogP_Rule'] = properties['LogP'] <= 5
    if not rules['LogP_Rule']:
        violations += 1
    
    rules['HBD_Rule'] = properties['HBD'] <= 5
    if not rules['HBD_Rule']:
        violations += 1
    
    rules['HBA_Rule'] = properties['HBA'] <= 10
    if not rules['HBA_Rule']:
        violations += 1
    
    rules['Lipinski_Violations'] = violations
    rules['Lipinski_Pass'] = violations <= 1
    
    return rules

def check_veber_rule(properties):
    """检查Veber规则"""
    rules = {}
    rules['RotBonds_Rule'] = properties['RotBonds'] <= 10
    rules['TPSA_Rule'] = properties['TPSA'] <= 140
    rules['Veber_Pass'] = rules['RotBonds_Rule'] and rules['TPSA_Rule']
    return rules

def check_egan_rule(properties):
    """检查Egan规则"""
    rules = {}
    rules['LogP_Egan'] = -1 <= properties['LogP'] <= 6
    rules['TPSA_Egan'] = 0 <= properties['TPSA'] <= 132
    rules['Egan_Pass'] = rules['LogP_Egan'] and rules['TPSA_Egan']
    return rules

def check_muegge_rule(properties):
    """检查Muegge规则"""
    rules = {}
    violations = 0
    
    rules['MW_Muegge'] = 200 <= properties['MolWt'] <= 600
    if not rules['MW_Muegge']: violations += 1
    
    rules['LogP_Muegge'] = -2 <= properties['LogP'] <= 5
    if not rules['LogP_Muegge']: violations += 1
    
    rules['TPSA_Muegge'] = properties['TPSA'] <= 150
    if not rules['TPSA_Muegge']: violations += 1
    
    rules['Rings_Muegge'] = 0 <= properties['NumRings'] <= 7
    if not rules['Rings_Muegge']: violations += 1
    
    rules['HeavyAtoms_Muegge'] = 4 <= properties['HeavyAtoms'] <= 15
    if not rules['HeavyAtoms_Muegge']: violations += 1
    
    rules['RotBonds_Muegge'] = properties['RotBonds'] <= 15
    if not rules['RotBonds_Muegge']: violations += 1
    
    rules['HBD_Muegge'] = properties['HBD'] <= 5
    if not rules['HBD_Muegge']: violations += 1
    
    rules['HBA_Muegge'] = properties['HBA'] <= 10
    if not rules['HBA_Muegge']: violations += 1
    
    rules['Muegge_Violations'] = violations
    rules['Muegge_Pass'] = violations == 0
    
    return rules

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py input_chunk.csv output_chunk.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    print(f"处理文件: {input_file}")
    
    # 读取输入文件
    df = pd.read_csv(input_file)
    
    properties_list = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        smiles = row['SMILES']
        props = calculate_druglike_properties(smiles)
        
        if props is not None:
            # 规则检查
            lipinski_rules = check_lipinski_rule(props)
            veber_rules = check_veber_rule(props)
            egan_rules = check_egan_rule(props)
            muegge_rules = check_muegge_rule(props)
            
            all_props = {**props, **lipinski_rules, **veber_rules, **egan_rules, **muegge_rules}
            properties_list.append(all_props)
            valid_indices.append(idx)
    
    if properties_list:
        # 创建结果DataFrame
        prop_df = pd.DataFrame(properties_list)
        valid_df = df.iloc[valid_indices].reset_index(drop=True)
        result_df = pd.concat([valid_df, prop_df], axis=1)
        
        # 保存结果
        result_df.to_csv(output_file, index=False)
        print(f"完成处理: {len(result_df)} 个有效分子")
    else:
        # 创建空的结果文件
        df.iloc[:0].to_csv(output_file, index=False)
        print("没有有效分子")
'''
    return script_content

def split_dataframe_to_chunks(df, num_chunks, temp_dir):
    """将DataFrame分割成多个块文件"""
    chunk_files = []
    chunk_size = len(df) // num_chunks
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        if i == num_chunks - 1:  # 最后一块包含剩余所有数据
            chunk_df = df.iloc[start_idx:]
        else:
            chunk_df = df.iloc[start_idx:start_idx + chunk_size]
        
        chunk_file = os.path.join(temp_dir, f"chunk_{i}.csv")
        chunk_df.to_csv(chunk_file, index=False)
        chunk_files.append(chunk_file)
    
    return chunk_files

def merge_result_files(result_files, output_file):
    """合并多个结果文件"""
    all_results = []
    
    for file in result_files:
        if os.path.exists(file) and os.path.getsize(file) > 0:
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    all_results.append(df)
            except Exception as e:
                print(f"读取文件 {file} 时出错: {e}")
    
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_file, index=False)
        return len(final_df)
    else:
        return 0

# 主界面
st.markdown("---")

# 文件选择部分
col1, col2 = st.columns(2)

with col1:
    st.header("📂 文件选择")
    
    # 文件夹选择
    folders = list_data_folders()
    if folders:
        selected_folder = st.selectbox("选择工作目录", options=[""] + folders)
    else:
        selected_folder = ""
        st.warning("暂无数据文件夹，请先在数据预处理页面创建")

with col2:
    st.header("📄 CSV文件选择")
    
    if selected_folder:
        csv_files = list_csv_files_in_folder(selected_folder)
        if csv_files:
            selected_file = st.selectbox("选择CSV文件", options=[""] + csv_files)
        else:
            selected_file = ""
            st.info("该文件夹中暂无CSV文件")
    else:
        selected_file = ""
        st.info("请先选择工作目录")

# 数据处理部分
if selected_folder and selected_file:
    st.markdown("---")
    
    file_path = os.path.join(DATA_DIR, selected_folder, selected_file)
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        st.success(f"成功读取文件: {selected_file} (包含 {len(df)} 条记录)")
        
        # 检查必需列
        if 'SMILES' not in df.columns:
            st.error("CSV文件中必须包含 'SMILES' 列")
            st.stop()
        
        # 显示文件预览
        with st.expander("文件预览", expanded=False):
            st.dataframe(df.head())
        
        # 成药性计算
        st.header("🧮 成药性属性计算")
        
        # 并行处理参数设置
        st.subheader("⚙️ 并行处理设置")
        
        col1, col2 = st.columns(2)
        with col1:
            max_procs = os.cpu_count() or 4
            num_chunks = st.slider("分块数量", 1, max_procs, min(4, max_procs))
        
        with col2:
            processing_method = st.selectbox(
                "处理方式",
                ["分块并行处理", "单进程处理"],
                index=0
            )

        if st.button("开始计算成药性属性", type="primary"):
            if processing_method == "分块并行处理":
                with st.spinner("正在进行分块并行计算..."):
                    # 创建临时目录
                    temp_dir = tempfile.mkdtemp(prefix="druglike_")
                    
                    try:
                        # 生成处理脚本
                        script_content = generate_druglike_script()
                        script_file = os.path.join(temp_dir, "druglike_processor.py")
                        with open(script_file, 'w', encoding='utf-8') as f:
                            f.write(script_content)
                        
                        # 分割数据文件
                        st.info(f"正在将数据分割为 {num_chunks} 个块...")
                        chunk_files = split_dataframe_to_chunks(df, num_chunks, temp_dir)
                        
                        # 创建输出文件列表
                        result_files = []
                        processes = []
                        
                        # 启动并行进程
                        st.info("启动并行计算进程...")
                        for i, chunk_file in enumerate(chunk_files):
                            result_file = os.path.join(temp_dir, f"result_{i}.csv")
                            result_files.append(result_file)
                            
                            # 启动子进程
                            cmd = ["python", script_file, chunk_file, result_file]
                            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            processes.append(process)
                        
                        # 监控进程进度
                        progress_bar = st.progress(0.0)
                        completed = 0
                        
                        while completed < len(processes):
                            time.sleep(1)  # 等待1秒
                            completed = sum(1 for p in processes if p.poll() is not None)
                            progress_bar.progress(completed / len(processes))
                        
                        # 检查所有进程是否成功完成
                        failed_processes = []
                        for i, process in enumerate(processes):
                            if process.returncode != 0:
                                stderr = process.stderr.read().decode()
                                failed_processes.append((i, stderr))
                        
                        if failed_processes:
                            st.warning(f"有 {len(failed_processes)} 个进程执行失败")
                            for i, error in failed_processes:
                                st.error(f"进程 {i} 错误: {error}")
                        
                        # 合并结果文件
                        st.info("合并计算结果...")
                        output_file = os.path.join(temp_dir, "final_result.csv")
                        total_molecules = merge_result_files(result_files, output_file)
                        
                        if total_molecules > 0:
                            # 读取最终结果
                            result_df = pd.read_csv(output_file)
                            st.session_state['druglike_df'] = result_df
                            st.session_state['original_count'] = len(df)
                            st.session_state['valid_count'] = len(result_df)
                            st.session_state['invalid_count'] = len(df) - len(result_df)
                            
                            st.success(f"并行计算完成! 有效分子: {len(result_df)}, 无效分子: {len(df) - len(result_df)}")
                        else:
                            st.error("没有有效的分子可以计算属性")
                    
                    except Exception as e:
                        st.error(f"并行处理出错: {str(e)}")
                    
                    finally:
                        # 清理临时文件
                        try:
                            shutil.rmtree(temp_dir)
                        except:
                            pass
            
            else:  # 单进程处理
                with st.spinner("正在单进程计算成药性属性..."):
                    properties_list = []
                    valid_smiles = []
                    invalid_count = 0
                    
                    progress_bar = st.progress(0)
                    
                    for i, smiles in enumerate(df['SMILES']):
                        progress_bar.progress((i + 1) / len(df))
                        
                        props = calculate_druglike_properties(smiles)
                        if props is not None:
                            # 添加规则检查
                            lipinski_rules = check_lipinski_rule(props)
                            veber_rules = check_veber_rule(props)
                            egan_rules = check_egan_rule(props)
                            muegge_rules = check_muegge_rule(props)
                            
                            # 合并所有属性
                            all_props = {**props, **lipinski_rules, **veber_rules, **egan_rules, **muegge_rules}
                            properties_list.append(all_props)
                            valid_smiles.append(smiles)
                        else:
                            invalid_count += 1
                    
                    if properties_list:
                        # 创建属性DataFrame
                        prop_df = pd.DataFrame(properties_list)
                        
                        # 合并原始数据和属性数据
                        valid_df = df[df['SMILES'].isin(valid_smiles)].reset_index(drop=True)
                        result_df = pd.concat([valid_df, prop_df], axis=1)
                        
                        st.success(f"计算完成! 有效分子: {len(result_df)}, 无效分子: {invalid_count}")
                        
                        # 保存计算结果到session state
                        st.session_state['druglike_df'] = result_df
                        st.session_state['original_count'] = len(df)
                        st.session_state['valid_count'] = len(result_df)
                        st.session_state['invalid_count'] = invalid_count
                    else:
                        st.error("没有有效的分子可以计算属性")
                        st.stop()
        
        # 如果已经计算了属性，显示筛选界面
        if 'druglike_df' in st.session_state:
            result_df = st.session_state['druglike_df']
            
            st.markdown("---")
            st.header("📊 数据统计")
            
            # 统计信息
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("原始分子数", st.session_state['original_count'])
            with col2:
                st.metric("有效分子数", st.session_state['valid_count'])
            with col3:
                st.metric("无效分子数", st.session_state['invalid_count'])
            with col4:
                st.metric("有效率", f"{st.session_state['valid_count']/st.session_state['original_count']*100:.1f}%")
            
            # 规则统计
            st.subheader("🎯 成药性规则统计")
            
            rule_cols = st.columns(4)
            with rule_cols[0]:
                lipinski_pass = result_df['Lipinski_Pass'].sum()
                st.metric("Lipinski规则通过", f"{lipinski_pass}/{len(result_df)}", 
                         f"{lipinski_pass/len(result_df)*100:.1f}%")
            
            with rule_cols[1]:
                veber_pass = result_df['Veber_Pass'].sum()
                st.metric("Veber规则通过", f"{veber_pass}/{len(result_df)}", 
                         f"{veber_pass/len(result_df)*100:.1f}%")
            
            with rule_cols[2]:
                egan_pass = result_df['Egan_Pass'].sum()
                st.metric("Egan规则通过", f"{egan_pass}/{len(result_df)}", 
                         f"{egan_pass/len(result_df)*100:.1f}%")
            
            with rule_cols[3]:
                muegge_pass = result_df['Muegge_Pass'].sum()
                st.metric("Muegge规则通过", f"{muegge_pass}/{len(result_df)}", 
                         f"{muegge_pass/len(result_df)*100:.1f}%")
            
            # 属性分布可视化
            st.subheader("📈 属性分布")
            
            # 选择要可视化的属性
            viz_properties = ['MolWt', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'QED']
            selected_viz_prop = st.selectbox("选择要可视化的属性", viz_properties)
            
            if selected_viz_prop in result_df.columns:
                fig = create_property_distribution_plot(result_df, selected_viz_prop)
                if fig:
                    st.pyplot(fig)
                    plt.close()
            
            # 筛选条件设置
            st.markdown("---")
            st.header("🔍 筛选条件设置")
            
            # 预设规则筛选
            st.subheader("📋 预设规则筛选")
            
            rule_filter_cols = st.columns(4)
            with rule_filter_cols[0]:
                use_lipinski = st.checkbox("应用Lipinski规则", value=True)
            with rule_filter_cols[1]:
                use_veber = st.checkbox("应用Veber规则", value=False)
            with rule_filter_cols[2]:
                use_egan = st.checkbox("应用Egan规则", value=False)
            with rule_filter_cols[3]:
                use_muegge = st.checkbox("应用Muegge规则", value=False)
            
            # 自定义范围筛选
            st.subheader("⚙️ 自定义范围筛选")
            
            # 创建筛选条件
            filter_conditions = {}
            
            # 主要属性筛选
            main_props = ['MolWt', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds']
            
            for i, prop in enumerate(main_props):
                if i % 3 == 0:
                    filter_cols = st.columns(3)
                
                with filter_cols[i % 3]:
                    st.write(f"**{prop}**")
                    min_val = result_df[prop].min()
                    max_val = result_df[prop].max()
                    
                    use_filter = st.checkbox(f"筛选{prop}", key=f"use_{prop}")
                    if use_filter:
                        range_val = st.slider(
                            f"{prop}范围",
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=(float(min_val), float(max_val)),
                            key=f"range_{prop}"
                        )
                        filter_conditions[prop] = range_val
            
            # 其他属性筛选
            with st.expander("其他属性筛选"):
                other_props = ['QED', 'AromaticRings', 'NumRings', 'HeavyAtoms', 'FractionCsp3']
                
                for i, prop in enumerate(other_props):
                    if i % 2 == 0:
                        other_cols = st.columns(2)
                    
                    with other_cols[i % 2]:
                        st.write(f"**{prop}**")
                        min_val = result_df[prop].min()
                        max_val = result_df[prop].max()
                        
                        use_filter = st.checkbox(f"筛选{prop}", key=f"use_{prop}")
                        if use_filter:
                            range_val = st.slider(
                                f"{prop}范围",
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=(float(min_val), float(max_val)),
                                key=f"range_{prop}"
                            )
                            filter_conditions[prop] = range_val
            
            # 应用筛选
            st.markdown("---")
            st.header("🎯 应用筛选")
            
            if st.button("应用筛选条件", type="primary"):
                filtered_df = result_df.copy()
                
                # 应用预设规则
                if use_lipinski:
                    filtered_df = filtered_df[filtered_df['Lipinski_Pass'] == True]
                if use_veber:
                    filtered_df = filtered_df[filtered_df['Veber_Pass'] == True]
                if use_egan:
                    filtered_df = filtered_df[filtered_df['Egan_Pass'] == True]
                if use_muegge:
                    filtered_df = filtered_df[filtered_df['Muegge_Pass'] == True]
                
                # 应用自定义范围筛选
                if filter_conditions:
                    filtered_df = apply_custom_filters(filtered_df, filter_conditions)
                
                st.session_state['filtered_df'] = filtered_df
                st.success(f"筛选完成! 筛选后分子数: {len(filtered_df)}/{len(result_df)} ({len(filtered_df)/len(result_df)*100:.1f}%)")
            
            # 显示筛选结果
            if 'filtered_df' in st.session_state:
                filtered_df = st.session_state['filtered_df']
                
                st.markdown("---")
                st.header("📋 筛选结果")
                
                # 筛选统计
                st.subheader("📊 筛选统计")
                filter_stats_cols = st.columns(3)
                
                with filter_stats_cols[0]:
                    st.metric("筛选前", len(result_df))
                with filter_stats_cols[1]:
                    st.metric("筛选后", len(filtered_df))
                with filter_stats_cols[2]:
                    retention_rate = len(filtered_df) / len(result_df) * 100
                    st.metric("保留率", f"{retention_rate:.1f}%")
                
                # 显示筛选后的数据
                st.subheader("📄 筛选后数据预览")
                
                # 选择要显示的列
                display_cols = ['ID', 'SMILES', 'MolWt', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'QED']
                available_display_cols = [col for col in display_cols if col in filtered_df.columns]
                
                st.dataframe(filtered_df[available_display_cols].head(20))
                
                # 保存筛选结果
                st.subheader("💾 保存筛选结果")
                
                # 生成输出文件名
                base_name = os.path.splitext(selected_file)[0]
                output_filename = f"filtered_{base_name}.csv"
                output_path = os.path.join(DATA_DIR, selected_folder, output_filename)
                
                st.text(f"输出文件名: {output_filename}")
                
                # 保存选项
                save_cols = st.columns(2)
                with save_cols[0]:
                    save_all_cols = st.checkbox("保存所有计算属性", value=False)
                with save_cols[1]:
                    save_original_only = st.checkbox("仅保存原始列", value=True)
                
                if st.button("💾 保存筛选结果", type="primary"):
                    try:
                        # 确定要保存的列
                        if save_original_only:
                            # 只保存原始CSV文件的列
                            original_cols = df.columns.tolist()
                            save_df = filtered_df[original_cols]
                        elif save_all_cols:
                            # 保存所有列
                            save_df = filtered_df
                        else:
                            # 保存关键列
                            key_cols = ['ID', 'SMILES'] + [col for col in ['MolWt', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'QED'] if col in filtered_df.columns]
                            # 添加其他原始列
                            other_original_cols = [col for col in df.columns if col not in key_cols and col in filtered_df.columns]
                            save_df = filtered_df[key_cols + other_original_cols]
                        
                        # 保存文件
                        save_df.to_csv(output_path, index=False)
                        
                        st.success(f"✅ 筛选结果已保存: {output_filename}")
                        st.info(f"保存了 {len(save_df)} 行数据, {len(save_df.columns)} 列")
                        
                        # 提供下载链接
                        with open(output_path, 'rb') as f:
                            st.download_button(
                                label="📥 下载筛选结果",
                                data=f.read(),
                                file_name=output_filename,
                                mime="text/csv"
                            )
                    
                    except Exception as e:
                        st.error(f"保存文件时出错: {str(e)}")
    
    except Exception as e:
        st.error(f"读取文件时出错: {str(e)}")

# 侧边栏状态信息
if selected_folder:
    with st.sidebar:
        st.subheader("📊 当前状态")
        st.text(f"工作目录: {selected_folder}")
        if selected_file:
            st.text(f"选择文件: {selected_file}")
        
        # 显示文件信息
        if selected_file:
            file_path = os.path.join(DATA_DIR, selected_folder, selected_file)
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # KB
                st.text(f"文件大小: {file_size:.1f}KB")
        
        # 显示计算状态
        if 'druglike_df' in st.session_state:
            st.text("✅ 属性已计算")
            st.text(f"有效分子: {st.session_state['valid_count']}")
        
        if 'filtered_df' in st.session_state:
            st.text("✅ 筛选已完成")
            st.text(f"筛选后: {len(st.session_state['filtered_df'])}")
        
        # 成药性规则说明
        st.subheader("📚 成药性规则说明")
        with st.expander("Lipinski规则"):
            st.text("• 分子量 ≤ 500 Da")
            st.text("• LogP ≤ 5")
            st.text("• 氢键供体 ≤ 5")
            st.text("• 氢键受体 ≤ 10")
            st.text("• 违规 ≤ 1个")
        
        with st.expander("Veber规则"):
            st.text("• 旋转键 ≤ 10")
            st.text("• TPSA ≤ 140 Ų")
        
        with st.expander("Egan规则"):
            st.text("• LogP: -1 to 6")
            st.text("• TPSA: 0 to 132")
        
        with st.expander("Muegge规则"):
            st.text("• 分子量: 200-600 Da")
            st.text("• LogP: -2 to 5")
            st.text("• TPSA ≤ 150")
            st.text("• 环数: 0-7")
            st.text("• 重原子: 4-15")
            st.text("• 旋转键 ≤ 15")
            st.text("• HBD ≤ 5, HBA ≤ 10")
