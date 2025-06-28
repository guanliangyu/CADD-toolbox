"""
CADD-Toolbox - 2D分子描述符生成页面
基于SMILES生成多种2D分子描述符和指纹
"""
import os
import pandas as pd
import streamlit as st
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, rdFingerprintGenerator
from rdkit.Chem.QED import qed
from rdkit.Chem.Descriptors import rdMolDescriptors as rdMolDesc
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.Chem import AllChem
from rdkit.Avalon import pyAvalonTools
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from tqdm import tqdm

# 设置页面配置
st.set_page_config(
    page_title="生成2D分子描述符",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 生成2D分子描述符")

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
    """列出指定文件夹中的CSV文件"""
    if not folder_name:
        return []
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.isdir(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.lower().endswith('.csv') and os.path.isfile(os.path.join(folder_path, f))]

def validate_smiles(smiles):
    """验证SMILES字符串"""
    if pd.isna(smiles):
        return False
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol is not None
    except:
        return False

def generate_molecular_descriptors(smiles_series, progress_callback=None):
    """生成分子描述符"""
    descriptors = []
    
    descriptor_functions = {
        'MolWt': Descriptors.MolWt,
        'LogP': Descriptors.MolLogP,
        'NumHDonors': Descriptors.NumHDonors,
        'NumHAcceptors': Descriptors.NumHAcceptors,
        'TPSA': Descriptors.TPSA,
        'NumRotatableBonds': Descriptors.NumRotatableBonds,
        'NumAromaticRings': Descriptors.NumAromaticRings,
        'NumAliphaticRings': Descriptors.NumAliphaticRings,
        'NumHeavyAtoms': Descriptors.HeavyAtomCount,
        'NumHeteroatoms': Descriptors.NumHeteroatoms,
        'FractionCsp3': Descriptors.FractionCsp3,
        'Chi0v': Descriptors.Chi0v,
        'Chi1v': Descriptors.Chi1v,
        'Chi2v': Descriptors.Chi2v,
        'Chi3v': Descriptors.Chi3v,
        'Chi4v': Descriptors.Chi4v,
        'Kappa1': Descriptors.Kappa1,
        'Kappa2': Descriptors.Kappa2,
        'Kappa3': Descriptors.Kappa3,
        'BertzCT': Descriptors.BertzCT,
        'SlogP_VSA1': Descriptors.SlogP_VSA1,
        'SlogP_VSA2': Descriptors.SlogP_VSA2,
        'SMR_VSA1': Descriptors.SMR_VSA1,
        'SMR_VSA2': Descriptors.SMR_VSA2,
        'LabuteASA': Descriptors.LabuteASA,
        'PEOE_VSA1': Descriptors.PEOE_VSA1,
        'PEOE_VSA2': Descriptors.PEOE_VSA2,
        'QED': qed,
        'MaxAbsPartialCharge': Descriptors.MaxAbsPartialCharge,
        'MinAbsPartialCharge': Descriptors.MinAbsPartialCharge,
        'MaxPartialCharge': Descriptors.MaxPartialCharge,
        'MinPartialCharge': Descriptors.MinPartialCharge,
        'MolMR': Descriptors.MolMR,
        'BalabanJ': Descriptors.BalabanJ,
        'HallKierAlpha': Descriptors.HallKierAlpha,
        'NumSaturatedCarbocycles': Descriptors.NumSaturatedCarbocycles,
        'NumSaturatedHeterocycles': Descriptors.NumSaturatedHeterocycles,
        'NumAliphaticCarbocycles': Descriptors.NumAliphaticCarbocycles,
        'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles,
        'NumAromaticCarbocycles': Descriptors.NumAromaticCarbocycles,
        'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles,
        'RingCount': Descriptors.RingCount,
        'FpDensityMorgan1': Descriptors.FpDensityMorgan1,
        'FpDensityMorgan2': Descriptors.FpDensityMorgan2,
        'BCUT2D_MWHI': Descriptors.BCUT2D_MWHI,
        'BCUT2D_MWLOW': Descriptors.BCUT2D_MWLOW,
        'BCUT2D_CHGHI': Descriptors.BCUT2D_CHGHI,
        'BCUT2D_CHGLO': Descriptors.BCUT2D_CHGLO,
        'BCUT2D_LOGPHI': Descriptors.BCUT2D_LOGPHI,
        'BCUT2D_LOGPLOW': Descriptors.BCUT2D_LOGPLOW,
        'BCUT2D_MRHI': Descriptors.BCUT2D_MRHI,
        'BCUT2D_MRLOW': Descriptors.BCUT2D_MRLOW,
    }
    
    for i, smiles in enumerate(smiles_series):
        row_descriptors = {}
        
        if pd.isna(smiles):
            # 填充NaN值
            for desc_name in descriptor_functions:
                row_descriptors[desc_name] = np.nan
        else:
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    for desc_name, desc_func in descriptor_functions.items():
                        try:
                            value = desc_func(mol)
                            row_descriptors[desc_name] = value
                        except:
                            row_descriptors[desc_name] = np.nan
                else:
                    # 无效SMILES
                    for desc_name in descriptor_functions:
                        row_descriptors[desc_name] = np.nan
            except:
                # SMILES解析失败
                for desc_name in descriptor_functions:
                    row_descriptors[desc_name] = np.nan
        
        descriptors.append(row_descriptors)
        
        if progress_callback and (i + 1) % 100 == 0:
            progress_callback(i + 1, len(smiles_series))
    
    return pd.DataFrame(descriptors)

def generate_fingerprints(smiles_series, fp_type='morgan', radius=2, nbits=1024, progress_callback=None):
    """生成分子指纹"""
    fingerprints = []
    
    for i, smiles in enumerate(smiles_series):
        if pd.isna(smiles):
            # 填充零向量
            fingerprints.append([0] * nbits)
        else:
            try:
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is not None:
                    if fp_type == 'morgan':
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
                    elif fp_type == 'rdkit':
                        fp = Chem.RDKFingerprint(mol, fpSize=nbits)
                    elif fp_type == 'maccs':
                        fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
                        # MACCS键固定为167位
                        fingerprints.append(list(fp))
                        if progress_callback and (i + 1) % 100 == 0:
                            progress_callback(i + 1, len(smiles_series))
                        continue
                    elif fp_type == 'avalon':
                        fp = pyAvalonTools.GetAvalonFP(mol, nbits)
                    elif fp_type == 'atompairs':
                        fp = Pairs.GetAtomPairFingerprintAsBitVect(mol, nBits=nbits)
                    elif fp_type == 'torsions':
                        fp = Torsions.GetTopologicalTorsionFingerprintAsBitVect(mol, nBits=nbits)
                    else:
                        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
                    
                    fingerprints.append(list(fp))
                else:
                    # 无效分子
                    fingerprints.append([0] * nbits)
            except:
                # 处理失败
                fingerprints.append([0] * nbits)
        
        if progress_callback and (i + 1) % 100 == 0:
            progress_callback(i + 1, len(smiles_series))
    
    # 为指纹位创建列名
    if fp_type == 'maccs':
        fp_columns = [f'MACCS_{i}' for i in range(167)]
    else:
        fp_columns = [f'{fp_type.upper()}_FP_{i}' for i in range(nbits)]
    
    return pd.DataFrame(fingerprints, columns=fp_columns)

def generate_2d_descriptors_batch(smiles_batch, descriptor_types, fp_config):
    """批量生成2D描述符"""
    batch_results = {}
    
    # 生成分子描述符
    if 'molecular_descriptors' in descriptor_types:
        batch_results['molecular_descriptors'] = generate_molecular_descriptors(smiles_batch)
    
    # 生成指纹
    for fp_type in ['morgan', 'rdkit', 'maccs', 'avalon', 'atompairs', 'torsions']:
        if fp_type in descriptor_types:
            if fp_type in fp_config:
                config = fp_config[fp_type]
                if fp_type == 'morgan':
                    fp_df = generate_fingerprints(smiles_batch, fp_type, 
                                                radius=config['radius'], 
                                                nbits=config['nbits'])
                elif fp_type == 'maccs':
                    fp_df = generate_fingerprints(smiles_batch, fp_type)
                elif fp_type in ['rdkit', 'avalon', 'atompairs', 'torsions']:
                    fp_df = generate_fingerprints(smiles_batch, fp_type, 
                                                nbits=config['nbits'])
                
                batch_results[fp_type] = fp_df
    
    return batch_results

st.markdown("""
基于SMILES字符串生成多种2D分子描述符和指纹，用于机器学习和QSAR建模。

🧬 **分子描述符**: 分子量、LogP、TPSA、拓扑指数等50+个描述符  
🔗 **分子指纹**: Morgan、RDKit、MACCS、Avalon等多种指纹  
⚡ **并行处理**: 支持多线程加速处理大数据集  
📊 **批量生成**: 支持多种描述符类型同时生成  
📈 **质量控制**: 自动验证SMILES有效性和处理异常
""")

# 文件选择
st.header("1. 📂 选择输入文件")

available_folders = list_data_folders()
if not available_folders:
    st.error(f"在 '{DATA_DIR}' 目录中未找到数据文件夹。请先运行基础成药性筛选页面。")
    st.stop()

# 文件夹选择
selected_folder = st.selectbox(
    "选择数据文件夹:",
    options=available_folders,
    index=0,
    help="选择包含CSV文件的文件夹"
)

if selected_folder:
    csv_files = list_csv_files_in_folder(selected_folder)
    
    if not csv_files:
        st.warning(f"选中的文件夹中没有找到CSV文件: {selected_folder}")
        st.stop()
    
    # 文件选择
    selected_file = st.selectbox(
        "选择CSV文件:",
        options=csv_files,
        index=0,
        help="选择包含SMILES列的CSV文件"
    )
    
    if selected_file:
        file_path = os.path.join(DATA_DIR, selected_folder, selected_file)
        st.success(f"已选择文件: {file_path}")
        
        # 显示文件信息
        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            st.info(f"📁 文件大小: {file_size:.2f} MB")
        except:
            pass

# 读取和预览数据
if 'selected_file' in locals() and selected_file:
    st.header("2. 📊 数据预览")
    
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        st.success(f"成功读取数据: {len(df)} 行 × {len(df.columns)} 列")
        
        # 显示基本信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总分子数", len(df))
        with col2:
            st.metric("列数", len(df.columns))
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("内存使用", f"{memory_usage:.1f} MB")
        
        # 显示前几行
        st.subheader("数据预览")
        st.dataframe(df.head(10))
        
        # 检查SMILES列
        smiles_candidates = [col for col in df.columns if 'smiles' in col.lower()]
        
        if not smiles_candidates:
            st.error("未找到SMILES列！请确保CSV文件包含名为'SMILES'的列。")
            st.stop()
        
        # SMILES列选择
        smiles_column = st.selectbox(
            "选择SMILES列:",
            options=smiles_candidates,
            index=0,
            help="选择包含SMILES字符串的列"
        )
        
        # 验证SMILES有效性
        if smiles_column:
            smiles_series = df[smiles_column]
            valid_smiles = smiles_series.apply(validate_smiles)
            valid_count = valid_smiles.sum()
            invalid_count = len(smiles_series) - valid_count
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("有效SMILES", valid_count, delta=f"{valid_count/len(smiles_series)*100:.1f}%")
            with col2:
                st.metric("无效SMILES", invalid_count, delta=f"{invalid_count/len(smiles_series)*100:.1f}%")
            with col3:
                st.metric("有效率", f"{valid_count/len(smiles_series)*100:.1f}%")
            
            if invalid_count > 0:
                st.warning(f"发现 {invalid_count} 个无效SMILES，将在生成描述符时用NaN填充")
                
                # 显示一些无效的SMILES例子
                invalid_smiles = smiles_series[~valid_smiles].dropna().head(5)
                if len(invalid_smiles) > 0:
                    with st.expander("查看无效SMILES示例"):
                        for idx, smiles in invalid_smiles.items():
                            st.code(f"行 {idx}: {smiles}")
        
    except Exception as e:
        st.error(f"读取CSV文件失败: {e}")
        st.stop()

# 描述符类型选择
if 'df' in locals():
    st.header("3. 🧬 选择描述符类型")
    
    # 分子描述符选择
    st.subheader("分子描述符")
    include_molecular_descriptors = st.checkbox(
        "包含分子描述符",
        value=True,
        help="包含50+个常用分子描述符：分子量、LogP、TPSA、拓扑指数等"
    )
    
    if include_molecular_descriptors:
        st.info("将生成50+个分子描述符，包括：分子量、LogP、氢键供体/受体、TPSA、旋转键数、芳香环数、拓扑指数、BCUT描述符等")
    
    # 分子指纹选择
    st.subheader("分子指纹")
    
    fingerprint_types = []
    fp_config = {}
    
    # Morgan指纹
    include_morgan = st.checkbox(
        "Morgan指纹 (ECFP)",
        value=True,
        help="基于原子环境的圆形指纹，最常用的分子指纹"
    )
    
    if include_morgan:
        fingerprint_types.append('morgan')
        col1, col2 = st.columns(2)
        with col1:
            morgan_radius = st.selectbox(
                "Morgan半径:",
                options=[1, 2, 3, 4],
                index=1,
                help="指纹半径，影响原子环境的大小"
            )
        with col2:
            morgan_bits = st.selectbox(
                "Morgan位数:",
                options=[512, 1024, 2048, 4096],
                index=1,
                help="指纹向量的长度"
            )
        fp_config['morgan'] = {'radius': morgan_radius, 'nbits': morgan_bits}
    
    # RDKit指纹
    include_rdkit = st.checkbox(
        "RDKit指纹",
        value=False,
        help="基于路径的分子指纹"
    )
    
    if include_rdkit:
        fingerprint_types.append('rdkit')
        rdkit_bits = st.selectbox(
            "RDKit位数:",
            options=[512, 1024, 2048, 4096],
            index=1,
            key="rdkit_bits"
        )
        fp_config['rdkit'] = {'nbits': rdkit_bits}
    
    # MACCS指纹
    include_maccs = st.checkbox(
        "MACCS指纹",
        value=False,
        help="基于药物化学子结构的166位指纹"
    )
    
    if include_maccs:
        fingerprint_types.append('maccs')
        st.info("MACCS指纹固定为167位，基于药物化学相关的子结构")
    
    # Avalon指纹
    include_avalon = st.checkbox(
        "Avalon指纹",
        value=False,
        help="基于特征路径的分子指纹"
    )
    
    if include_avalon:
        fingerprint_types.append('avalon')
        avalon_bits = st.selectbox(
            "Avalon位数:",
            options=[512, 1024, 2048, 4096],
            index=1,
            key="avalon_bits"
        )
        fp_config['avalon'] = {'nbits': avalon_bits}
    
    # 原子对指纹
    include_atompairs = st.checkbox(
        "原子对指纹",
        value=False,
        help="基于原子对的分子指纹"
    )
    
    if include_atompairs:
        fingerprint_types.append('atompairs')
        ap_bits = st.selectbox(
            "原子对位数:",
            options=[512, 1024, 2048, 4096],
            index=1,
            key="ap_bits"
        )
        fp_config['atompairs'] = {'nbits': ap_bits}
    
    # 扭转指纹
    include_torsions = st.checkbox(
        "拓扑扭转指纹",
        value=False,
        help="基于拓扑扭转的分子指纹"
    )
    
    if include_torsions:
        fingerprint_types.append('torsions')
        torsion_bits = st.selectbox(
            "扭转指纹位数:",
            options=[512, 1024, 2048, 4096],
            index=1,
            key="torsion_bits"
        )
        fp_config['torsions'] = {'nbits': torsion_bits}
    
    # 汇总选择的描述符类型
    descriptor_types = []
    if include_molecular_descriptors:
        descriptor_types.append('molecular_descriptors')
    descriptor_types.extend(fingerprint_types)
    
    if not descriptor_types:
        st.warning("请至少选择一种描述符类型！")
        st.stop()
    
    # 显示选择摘要
    st.subheader("选择摘要")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**已选择的描述符类型:**")
        if include_molecular_descriptors:
            st.write("- 🧬 分子描述符 (50+个)")
        for fp_type in fingerprint_types:
            if fp_type == 'morgan':
                st.write(f"- 🔗 Morgan指纹 (半径{morgan_radius}, {morgan_bits}位)")
            elif fp_type == 'rdkit':
                st.write(f"- 🔗 RDKit指纹 ({rdkit_bits}位)")
            elif fp_type == 'maccs':
                st.write("- 🔗 MACCS指纹 (167位)")
            elif fp_type == 'avalon':
                st.write(f"- 🔗 Avalon指纹 ({avalon_bits}位)")
            elif fp_type == 'atompairs':
                st.write(f"- 🔗 原子对指纹 ({ap_bits}位)")
            elif fp_type == 'torsions':
                st.write(f"- 🔗 扭转指纹 ({torsion_bits}位)")
    
    with col2:
        # 估算总列数
        total_columns = len(df.columns)  # 原有列
        if include_molecular_descriptors:
            total_columns += 50  # 分子描述符
        for fp_type in fingerprint_types:
            if fp_type == 'maccs':
                total_columns += 167
            elif fp_type in fp_config:
                total_columns += fp_config[fp_type]['nbits']
        
        st.metric("预计总列数", total_columns)
        
        # 估算内存使用
        estimated_memory = len(df) * total_columns * 8 / (1024 * 1024)  # 假设每个值8字节
        st.metric("预计内存", f"{estimated_memory:.1f} MB")
        
        if estimated_memory > 1000:
            st.warning("⚠️ 预计内存使用较大，建议减少指纹位数或分批处理")

# 处理选项
st.header("4. ⚙️ 处理选项")

col1, col2, col3 = st.columns(3)

with col1:
    # 处理范围
    processing_options = [
        "处理所有分子",
        "仅处理前100个分子",
        "仅处理前500个分子",
        "仅处理前1000个分子"
    ]
    selected_scope = st.selectbox(
        "处理范围:",
        options=processing_options,
        index=0,
        help="选择要处理的分子数量"
    )

with col2:
    # 并行处理
    cpu_cores = multiprocessing.cpu_count()
    num_threads = st.number_input(
        "并行线程数:",
        min_value=1,
        max_value=cpu_cores,
        value=min(4, cpu_cores),
        help=f"系统有{cpu_cores}个CPU核心"
    )

with col3:
    # 批处理大小
    batch_size = st.number_input(
        "批处理大小:",
        min_value=50,
        max_value=1000,
        value=200,
        help="每批处理的分子数量"
    )

# 确定处理数量
if 'df' in locals():
    total_molecules = len(df)
    if selected_scope == "处理所有分子":
        molecules_to_process = total_molecules
    elif selected_scope == "仅处理前100个分子":
        molecules_to_process = min(100, total_molecules)
    elif selected_scope == "仅处理前500个分子":
        molecules_to_process = min(500, total_molecules)
    elif selected_scope == "仅处理前1000个分子":
        molecules_to_process = min(1000, total_molecules)
    
    st.info(f"将处理 {molecules_to_process} 个分子")

# 生成描述符
if 'df' in locals() and 'smiles_column' in locals() and descriptor_types:
    st.header("5. 🚀 生成2D描述符")
    
    if st.button("开始生成2D描述符", type="primary"):
        start_time = time.time()
        
        # 准备数据
        df_subset = df.head(molecules_to_process).copy()
        smiles_series = df_subset[smiles_column]
        
        st.info(f"开始生成2D描述符，共 {len(df_subset)} 个分子...")
        
        # 创建进度条
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        # 生成描述符
        all_descriptors = []
        
        try:
            # 分子描述符
            if 'molecular_descriptors' in descriptor_types:
                status_text.text("正在生成分子描述符...")
                
                def update_progress(current, total):
                    progress = current / total * 0.3  # 分子描述符占30%
                    progress_bar.progress(progress)
                    status_text.text(f"正在生成分子描述符... {current}/{total}")
                
                mol_desc_df = generate_molecular_descriptors(smiles_series, update_progress)
                all_descriptors.append(mol_desc_df)
                st.success(f"✅ 分子描述符生成完成: {len(mol_desc_df.columns)} 个描述符")
            
            # 分子指纹
            fingerprint_progress_start = 0.3 if 'molecular_descriptors' in descriptor_types else 0.0
            fingerprint_progress_per_type = (1.0 - fingerprint_progress_start) / len(fingerprint_types) if fingerprint_types else 0
            
            for i, fp_type in enumerate(fingerprint_types):
                status_text.text(f"正在生成{fp_type.upper()}指纹...")
                
                def update_fp_progress(current, total):
                    base_progress = fingerprint_progress_start + i * fingerprint_progress_per_type
                    type_progress = (current / total) * fingerprint_progress_per_type
                    progress_bar.progress(base_progress + type_progress)
                    status_text.text(f"正在生成{fp_type.upper()}指纹... {current}/{total}")
                
                if fp_type == 'morgan':
                    fp_df = generate_fingerprints(smiles_series, fp_type, 
                                                radius=fp_config[fp_type]['radius'],
                                                nbits=fp_config[fp_type]['nbits'],
                                                progress_callback=update_fp_progress)
                elif fp_type == 'maccs':
                    fp_df = generate_fingerprints(smiles_series, fp_type, 
                                                progress_callback=update_fp_progress)
                elif fp_type in ['rdkit', 'avalon', 'atompairs', 'torsions']:
                    fp_df = generate_fingerprints(smiles_series, fp_type,
                                                nbits=fp_config[fp_type]['nbits'],
                                                progress_callback=update_fp_progress)
                
                all_descriptors.append(fp_df)
                st.success(f"✅ {fp_type.upper()}指纹生成完成: {len(fp_df.columns)} 个特征")
            
            # 合并所有描述符
            status_text.text("正在合并描述符...")
            progress_bar.progress(0.95)
            
            # 合并原始数据和描述符
            final_df = df_subset.copy()
            for desc_df in all_descriptors:
                final_df = pd.concat([final_df, desc_df], axis=1)
            
            progress_bar.progress(1.0)
            total_time = time.time() - start_time
            
            st.success(f"🎉 2D描述符生成完成！")
            st.info(f"总耗时: {total_time:.1f}秒 | 最终数据: {len(final_df)} 行 × {len(final_df.columns)} 列")
            
            # 显示结果统计
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("原始列数", len(df_subset.columns))
            with col2:
                st.metric("新增列数", len(final_df.columns) - len(df_subset.columns))
            with col3:
                st.metric("最终列数", len(final_df.columns))
            
            # 预览结果
            st.subheader("结果预览")
            
            # 显示新增的描述符列
            original_columns = set(df_subset.columns)
            new_columns = [col for col in final_df.columns if col not in original_columns]
            
            if new_columns:
                st.info(f"新增了 {len(new_columns)} 个描述符特征")
                with st.expander("查看新增的描述符列"):
                    # 按类型分组显示
                    if 'molecular_descriptors' in descriptor_types:
                        mol_desc_cols = [col for col in new_columns if not any(fp in col for fp in ['MORGAN', 'RDKIT', 'MACCS', 'AVALON', 'ATOMPAIRS', 'TORSIONS'])]
                        if mol_desc_cols:
                            st.write("**分子描述符:**")
                            st.write(", ".join(mol_desc_cols))
                    
                    for fp_type in fingerprint_types:
                        fp_cols = [col for col in new_columns if fp_type.upper() in col]
                        if fp_cols:
                            st.write(f"**{fp_type.upper()}指纹特征:**")
                            st.write(f"{len(fp_cols)} 个二进制特征 ({fp_cols[0]} ... {fp_cols[-1]})")
            
            # 显示前几行
            st.dataframe(final_df.head(5))
            
            # 保存结果
            output_filename = f"2d_fingerprint_{selected_file}"
            output_path = os.path.join(DATA_DIR, selected_folder, output_filename)
            
            try:
                final_df.to_csv(output_path, index=False)
                st.success(f"✅ 结果已保存到: {output_path}")
                
                # 显示文件大小
                output_size = os.path.getsize(output_path) / (1024 * 1024)
                st.info(f"📁 输出文件大小: {output_size:.2f} MB")
                
            except Exception as e:
                st.error(f"保存文件失败: {e}")
            
            # 提供下载
            csv_data = final_df.to_csv(index=False)
            st.download_button(
                label="📥 下载2D描述符CSV文件",
                data=csv_data,
                file_name=output_filename,
                mime="text/csv",
                help="下载包含2D描述符的完整CSV文件"
            )
            
            # 显示描述符统计
            st.subheader("描述符统计")
            
            # 只对数值列进行统计
            numeric_columns = final_df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                stats_df = final_df[numeric_columns].describe()
                st.dataframe(stats_df)
                
                # 检查缺失值
                missing_stats = final_df[numeric_columns].isnull().sum()
                missing_stats = missing_stats[missing_stats > 0]
                
                if len(missing_stats) > 0:
                    st.warning("⚠️ 缺失值统计:")
                    for col, missing_count in missing_stats.items():
                        st.write(f"- {col}: {missing_count} 个缺失值 ({missing_count/len(final_df)*100:.1f}%)")
                else:
                    st.success("✅ 所有描述符都已成功计算，无缺失值")
            
        except Exception as e:
            st.error(f"生成2D描述符时出错: {e}")
            import traceback
            st.code(traceback.format_exc(), language="text")

# 帮助信息
with st.sidebar:
    st.header("📚 帮助信息")
    
    with st.expander("🧬 分子描述符说明"):
        st.markdown("""
        **常用分子描述符：**
        - **MolWt**: 分子量
        - **LogP**: 脂水分配系数
        - **TPSA**: 拓扑极性表面积
        - **NumHDonors/NumHAcceptors**: 氢键供体/受体数
        - **NumRotatableBonds**: 可旋转键数
        - **FractionCsp3**: sp3碳原子分数
        - **Chi0v-Chi4v**: 价连接性指数
        - **Kappa1-Kappa3**: 分子形状指数
        - **BertzCT**: 分子复杂度
        - **QED**: 类药性评分
        """)
    
    with st.expander("🔗 分子指纹说明"):
        st.markdown("""
        **指纹类型对比：**
        
        **Morgan (ECFP)**:
        - 最常用的圆形指纹
        - 基于原子环境
        - 适合相似性搜索
        
        **RDKit**:
        - 基于路径的指纹
        - 考虑分子路径信息
        
        **MACCS**:
        - 167位固定指纹
        - 基于药物化学子结构
        - 适合药物分子
        
        **Avalon**:
        - 基于特征路径
        - 良好的多样性表现
        
        **原子对**:
        - 基于原子对距离
        - 适合骨架跃迁
        
        **扭转**:
        - 基于拓扑扭转
        - 捕获3D特征
        """)
    
    with st.expander("⚙️ 参数建议"):
        st.markdown("""
        **指纹参数建议：**
        
        **Morgan半径：**
        - 半径2: 标准设置，适合大多数应用
        - 半径3: 更大的原子环境，适合复杂分子
        
        **指纹位数：**
        - 1024位: 标准设置，平衡性能和精度
        - 2048位: 更高精度，适合大数据集
        
        **并行处理：**
        - 线程数 = CPU核心数的50-75%
        - 批处理大小: 100-500个分子
        """)
    
    with st.expander("🎯 使用建议"):
        st.markdown("""
        **应用场景：**
        
        **机器学习：**
        - 分子描述符 + Morgan指纹
        - 考虑使用特征选择
        
        **相似性搜索：**
        - Morgan指纹 (半径2, 1024位)
        - 或MACCS指纹
        
        **QSAR建模：**
        - 分子描述符为主
        - 结合Morgan指纹
        
        **虚拟筛选：**
        - 多种指纹组合
        - 考虑计算成本
        """)

st.divider()
st.markdown("🧬 **CADD-Toolbox** - 2D分子描述符生成工具")
