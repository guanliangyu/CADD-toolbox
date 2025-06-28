"""
CADD-Toolbox - 数据预处理页面
提供文件管理和数据标准化功能
"""
import os
import sys
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors import MolWt  # type: ignore[attr-defined]
import tempfile
import shutil

# 设置页面配置
st.set_page_config(
    page_title="数据预处理",
    page_icon="📁",
    layout="wide"
)

st.title("📁 数据预处理")

# 数据目录设置
DATA_DIR = os.path.abspath("data")

def ensure_data_dir():
    """确保data目录存在"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        st.info(f"已创建数据目录: {DATA_DIR}")

def list_data_folders():
    """列出data目录下的所有文件夹"""
    ensure_data_dir()
    folders = []
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return sorted(folders)

def list_files_in_folder(folder_name):
    """列出指定文件夹中的所有文件"""
    if not folder_name:
        return []
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        return []
    files = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isfile(item_path):
            files.append(item)
    return sorted(files)

def create_new_folder(folder_name):
    """在data目录下创建新文件夹"""
    if folder_name:
        folder_path = os.path.join(DATA_DIR, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            return True, f"成功创建文件夹: {folder_name}"
        else:
            return False, f"文件夹已存在: {folder_name}"
    return False, "请输入有效的文件夹名称"

def read_sdf_file(file_path):
    """一次遍历读取 SDF 文件，减少 I/O 开销"""
    data = []
    all_props = set()

    # 单遍历收集行数据及属性集合
    for i, mol in enumerate(Chem.SDMolSupplier(file_path)):
        if mol is None:
            continue

        row = {
            "Index": i + 1,
            "SMILES": Chem.MolToSmiles(mol),
            "MolWt": MolWt(mol)
        }

        for prop in mol.GetPropNames():
            row[prop] = mol.GetProp(prop)
            all_props.add(prop)

        data.append(row)

    # 补齐缺失列，避免后续 KeyError
    for row in data:
        for prop in all_props:
            row.setdefault(prop, None)

    df = pd.DataFrame(data)
    return df, list(df.columns)

def read_csv_file(file_path):
    """读取CSV文件"""
    try:
        df = pd.read_csv(file_path)
        return df, list(df.columns)
    except Exception as e:
        st.error(f"读取CSV文件时出错: {str(e)}")
        return None, []

def validate_smiles(smiles_series):
    """验证SMILES列的有效性"""
    valid_count = 0
    total_count = len(smiles_series)
    
    for smiles in smiles_series.dropna():
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            valid_count += 1
    
    return valid_count, total_count

def calculate_molecular_weight(smiles_series):
    """根据SMILES计算分子量"""
    mol_weights = []
    for smiles in smiles_series:
        if pd.isna(smiles):
            mol_weights.append(None)
        else:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                mol_weights.append(MolWt(mol))
            else:
                mol_weights.append(None)
    return mol_weights

def create_standardized_output(df, id_col, smiles_col, molwt_col, other_cols, output_path):
    """创建标准化输出"""
    # 创建新的DataFrame
    output_df = pd.DataFrame()
    
    # 标准列
    output_df['ID'] = df[id_col] if id_col else range(1, len(df) + 1)
    output_df['SMILES'] = df[smiles_col]
    
    # 分子量处理
    if molwt_col and molwt_col in df.columns:
        output_df['MolWt'] = df[molwt_col]
    else:
        # 根据SMILES计算分子量
        st.info("正在根据SMILES计算分子量...")
        output_df['MolWt'] = calculate_molecular_weight(df[smiles_col])
    
    # 其他选择的列
    for col in other_cols:
        if col in df.columns:
            output_df[col] = df[col]
    
    # 保存文件
    output_df.to_csv(output_path, index=False)

    return output_df

# --------------------------------------------------
# ⚡️ 缓存辅助工具：必须位于首次调用之前
# --------------------------------------------------


def file_mtime(path: str) -> float:
    """获取文件最后修改时间（秒级）"""
    try:
        return os.path.getmtime(path)
    except FileNotFoundError:
        return 0.0


@st.cache_data(show_spinner="⏳ 正在读取数据 ...")
def load_data(file_path: str, file_ext: str, mtime: float):
    """加载数据文件并缓存，mtime 变化自动失效"""
    if file_ext == '.sdf':
        return read_sdf_file(file_path)
    elif file_ext == '.csv':
        return read_csv_file(file_path)
    else:
        return None, []

# 主界面
st.markdown("---")

# 创建三列布局
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.header("📂 文件夹管理")
    
    # 创建新文件夹
    st.subheader("创建新文件夹")
    new_folder_name = st.text_input("文件夹名称", placeholder="输入新文件夹名称")
    if st.button("创建文件夹"):
        if new_folder_name:
            success, message = create_new_folder(new_folder_name)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
    
    # 文件夹选择
    st.subheader("选择工作目录")
    folders = list_data_folders()
    if folders:
        selected_folder = st.selectbox("选择文件夹", options=[""] + folders)
    else:
        selected_folder = ""
        st.info("暂无文件夹，请先创建")

with col2:
    st.header("📤 文件上传")
    
    if selected_folder:
        st.info(f"当前工作目录: {selected_folder}")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "上传文件",
            type=['csv', 'sdf'],
            help="支持CSV和SDF格式文件"
        )
        
        if uploaded_file is not None:
            # 保存上传的文件
            folder_path = os.path.join(DATA_DIR, selected_folder)
            file_path = os.path.join(folder_path, uploaded_file.name)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"文件已上传: {uploaded_file.name}")
            st.rerun()
    else:
        st.warning("请先选择工作目录")

with col3:
    st.header("📄 文件选择")
    
    if selected_folder:
        files = list_files_in_folder(selected_folder)
        if files:
            selected_file = st.selectbox("选择文件", options=[""] + files)
        else:
            selected_file = ""
            st.info("该文件夹中暂无文件")
    else:
        selected_file = ""
        st.info("请先选择工作目录")

# 文件处理部分
if selected_folder and selected_file:
    st.markdown("---")
    st.header("🔧 数据处理")
    
    file_path = os.path.join(DATA_DIR, selected_folder, selected_file)
    file_ext = os.path.splitext(selected_file)[1].lower()
    
    # 使用缓存读取数据
    st.info("正在读取数据...")
    df, available_columns = load_data(file_path, file_ext, file_mtime(file_path))

    if df is not None:
        if file_ext == '.sdf':
            st.success(f"成功读取SDF文件，包含 {len(df)} 个分子")
        elif file_ext == '.csv':
            st.success(f"成功读取CSV文件，包含 {len(df)} 行数据")
        
        # 显示文件预览
        with st.expander("文件预览", expanded=True):
            st.dataframe(df.head(10))
            st.info(f"总计 {len(df)} 行，{len(df.columns)} 列")
        
        # 字段选择
        st.subheader("🏷️ 字段映射")
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("**必需字段**")
            
            # ID列选择
            id_options = ["自动生成序号"] + available_columns
            id_col_idx = st.selectbox("ID列", options=range(len(id_options)), 
                                     format_func=lambda x: id_options[x])
            id_col = available_columns[id_col_idx-1] if id_col_idx > 0 else None
            
            # SMILES列选择
            smiles_options = [col for col in available_columns if 'smil' in col.lower() or 'structure' in col.lower()]
            if not smiles_options:
                smiles_options = available_columns
                
            smiles_col = st.selectbox("SMILES/结构列", options=available_columns,
                                    index=available_columns.index(smiles_options[0]) if smiles_options else 0)
            
            # 分子量列选择
            molwt_options = ["根据SMILES计算"] + [col for col in available_columns if 'molwt' in col.lower() or 'weight' in col.lower() or 'mw' in col.lower()]
            molwt_col_idx = st.selectbox("分子量列", options=range(len(molwt_options)),
                                       format_func=lambda x: molwt_options[x])
            molwt_col = available_columns[molwt_col_idx-1] if molwt_col_idx > 0 and molwt_col_idx <= len(available_columns) else None
        
        with col_right:
            st.markdown("**其他属性列**")
            
            # 其他列选择
            other_columns = [col for col in available_columns if col not in [id_col, smiles_col, molwt_col]]
            selected_other_cols = st.multiselect(
                "选择要保留的其他属性",
                options=other_columns,
                help="选择需要在输出文件中保留的其他列"
            )
        
        # 数据验证
        if smiles_col:
            with st.expander("数据验证"):
                st.info("正在验证SMILES数据...")
                valid_count, total_count = validate_smiles(df[smiles_col])
                st.metric("有效SMILES", f"{valid_count}/{total_count}", 
                         f"{valid_count/total_count*100:.1f}%" if total_count > 0 else "0%")
                
                if valid_count < total_count:
                    st.warning(f"发现 {total_count - valid_count} 个无效SMILES，处理时将跳过")
        
        # 输出设置
        st.subheader("💾 输出设置")
        
        # 生成输出文件名
        base_name = os.path.splitext(selected_file)[0]
        output_filename = f"prepared_{base_name}.csv"
        output_path = os.path.join(DATA_DIR, selected_folder, output_filename)
        
        st.text(f"输出文件名: {output_filename}")
        
        # 处理按钮
        if st.button("🚀 生成标准化文件", type="primary"):
            try:
                with st.spinner("正在处理数据..."):
                    output_df = create_standardized_output(
                        df, id_col, smiles_col, molwt_col, 
                        selected_other_cols, output_path
                    )
                
                st.success(f"✅ 成功生成标准化文件: {output_filename}")
                
                # 显示输出预览
                with st.expander("输出预览", expanded=True):
                    st.dataframe(output_df.head(10))
                    st.info(f"输出文件包含 {len(output_df)} 行，{len(output_df.columns)} 列")
                
                # 提供下载链接
                with open(output_path, 'rb') as f:
                    st.download_button(
                        label="📥 下载标准化文件",
                        data=f.read(),
                        file_name=output_filename,
                        mime="text/csv"
                    )
                    
            except Exception as e:
                st.error(f"处理数据时出错: {str(e)}")

# 显示当前状态
if selected_folder:
    with st.sidebar:
        st.subheader("📊 当前状态")
        st.text(f"工作目录: {selected_folder}")
        if selected_file:
            st.text(f"选择文件: {selected_file}")
        
        files = list_files_in_folder(selected_folder)
        st.text(f"文件数量: {len(files)}")
        
        # 显示文件列表
        if files:
            st.subheader("📁 文件列表")
            for file in files:
                file_path = os.path.join(DATA_DIR, selected_folder, file)
                file_size = os.path.getsize(file_path) / 1024  # KB
                st.text(f"• {file} ({file_size:.1f}KB)")