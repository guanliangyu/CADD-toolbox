"""
生成3D构象页面 - 从SMILES字符串或SDF文件生成3D构象
"""
import streamlit as st
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import io
import concurrent.futures
import multiprocessing
import os
import uuid
from datetime import datetime
import shutil
import time

PREVIEW_SIZE = 10
# 大文件阈值 (100MB)
LARGE_FILE_THRESHOLD = 100 * 1024 * 1024
# 超大文件阈值 (1GB)
HUGE_FILE_THRESHOLD = 1024 * 1024 * 1024

# 初始化会话状态
if 'last_processed_file_identifier' not in st.session_state:
    st.session_state.last_processed_file_identifier = None
if 'scan_results_valid' not in st.session_state:
    st.session_state.scan_results_valid = False 
if 'total_potential_mols_cache' not in st.session_state:
    st.session_state.total_potential_mols_cache = 0
if 'preview_data_cache' not in st.session_state:
    st.session_state.preview_data_cache = []
if 'df_preview_cache' not in st.session_state:
    st.session_state.df_preview_cache = None
if 'initial_scan_successful_cache' not in st.session_state:
    st.session_state.initial_scan_successful_cache = False
if 'saved_file_path' not in st.session_state:
    st.session_state.saved_file_path = None
if 'saved_work_dir' not in st.session_state:
    st.session_state.saved_work_dir = None
if 'file_size_cache' not in st.session_state:
    st.session_state.file_size_cache = None
if 'total_count_computing' not in st.session_state:
    st.session_state.total_count_computing = False
if 'file_hash_cache' not in st.session_state:
    st.session_state.file_hash_cache = None
if 'scan_timestamp_cache' not in st.session_state:
    st.session_state.scan_timestamp_cache = None
if 'is_estimated_cache' not in st.session_state:
    st.session_state.is_estimated_cache = False

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
        # UploadedFile对象
        if hasattr(file_path_or_obj, 'size'):
            return file_path_or_obj.size
        elif hasattr(file_path_or_obj, 'getvalue'):
            return len(file_path_or_obj.getvalue())
    return 0

def get_file_hash(file_path_or_obj):
    """获取文件的简单哈希值用于缓存验证"""
    import hashlib
    
    if isinstance(file_path_or_obj, str):
        if os.path.exists(file_path_or_obj):
            # 对于文件路径，使用文件修改时间+大小作为哈希
            stat = os.stat(file_path_or_obj)
            hash_input = f"{file_path_or_obj}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(hash_input.encode()).hexdigest()
    else:
        # 对于上传文件，使用名称+大小作为哈希
        if hasattr(file_path_or_obj, 'name') and hasattr(file_path_or_obj, 'size'):
            hash_input = f"{file_path_or_obj.name}_{file_path_or_obj.size}"
            return hashlib.md5(hash_input.encode()).hexdigest()
    return None

def is_cache_valid(file_path_or_obj):
    """检查缓存是否有效"""
    current_hash = get_file_hash(file_path_or_obj)
    if not current_hash:
        return False
    
    # 检查是否有缓存的哈希值
    cached_hash = st.session_state.file_hash_cache
    cached_identifier = st.session_state.last_processed_file_identifier
    
    # 当前文件标识符
    if isinstance(file_path_or_obj, str):
        current_identifier = file_path_or_obj
    else:
        current_identifier = file_path_or_obj.name if hasattr(file_path_or_obj, 'name') else str(file_path_or_obj)
    
    return (current_hash == cached_hash and 
            current_identifier == cached_identifier and 
            st.session_state.scan_results_valid)

def fast_line_count(file_path):
    """快速计算文件行数 - 适用于大文件"""
    count = 0
    with open(file_path, 'rb') as f:
        buffer_size = 1024 * 1024  # 1MB buffer
        while True:
            buffer = f.read(buffer_size)
            if not buffer:
                break
            count += buffer.count(b'\n')
    return count

def sample_lines_from_file(file_path, sample_size=10, total_lines=None):
    """从文件中采样行 - 用于大文件预览"""
    if total_lines is None:
        total_lines = fast_line_count(file_path)
    
    if total_lines <= sample_size:
        # 文件很小，读取所有行
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()], total_lines
    
    # 计算采样间隔
    step = max(1, total_lines // sample_size)
    sampled_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if len(sampled_lines) >= sample_size:
                break
            if i % step == 0:
                line = line.strip()
                if line:
                    sampled_lines.append(line)
    
    return sampled_lines, total_lines

def estimate_sdf_molecule_count(file_path, sample_size=1000):
    """估算SDF文件中的分子数量"""
    file_size = os.path.getsize(file_path)
    
    # 采样前sample_size字节来估算平均分子大小
    with open(file_path, 'rb') as f:
        sample_data = f.read(min(sample_size * 1000, file_size))  # 采样
    
    # 计算$$$$的数量来估算分子数
    molecule_separators = sample_data.count(b'$$$$')
    if molecule_separators == 0:
        return None  # 无法估算
    
    sample_bytes = len(sample_data)
    avg_mol_size = sample_bytes / molecule_separators
    estimated_count = int(file_size / avg_mol_size)
    
    return estimated_count

def generate_work_folder_name(filename):
    """生成工作文件夹名称：日期+随机码"""
    date_str = datetime.now().strftime("%Y%m%d")
    random_code = str(uuid.uuid4())[:8]
    base_name = os.path.splitext(filename)[0]
    sanitized_base = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in base_name).strip()
    return f"{date_str}_{sanitized_base}_{random_code}"

def save_uploaded_file(uploaded_file):
    """保存上传的文件到data目录，并清理upload缓存"""
    if not uploaded_file:
        return None, None
    
    # 生成工作文件夹
    folder_name = generate_work_folder_name(uploaded_file.name)
    work_dir = os.path.join(DATA_DIR, folder_name)
    
    try:
        os.makedirs(work_dir, exist_ok=True)
        file_path = os.path.join(work_dir, uploaded_file.name)
        
        # 保存文件
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
    return [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

def mols_to_sdf_string(mols_with_conformers):
    """将含有构象的分子列表转换为SDF字符串"""
    output = io.StringIO()
    sdf_writer = Chem.SDWriter(output)
    
    for mol in mols_with_conformers:
        if mol:
            for conf_id in range(mol.GetNumConformers()):
                sdf_writer.write(mol, confId=conf_id)
    
    sdf_writer.flush()
    sdf_writer.close()
    sdf_string = output.getvalue()
    output.close()
    return sdf_string

def generate_conformers_for_mol(mol, num_confs, max_attempts, random_seed):
    """为单个分子生成3D构象"""
    if not mol:
        return {'mol': None, 'success': False, 'message': "输入分子为空"}

    original_smiles = Chem.MolToSmiles(mol)

    try:
        mol_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        # 新版RDKit使用maxIterations而不是maxAttempts
        if hasattr(params, 'maxIterations'):
            params.maxIterations = max_attempts
        params.numThreads = 0
        
        cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params)
        
        if not cids or len(cids) == 0:
            return {'mol': None, 'success': False, 'message': f"RDKit构象生成失败: {original_smiles}"}
        
        return {'mol': mol_h, 'success': True, 'message': None}
    except Exception as e:
        return {'mol': None, 'success': False, 'message': f"异常: {original_smiles}: {str(e)}"}

def smart_file_scan(file_to_scan, current_filename, smiles_column):
    """智能文件扫描 - 根据文件大小选择不同策略"""
    file_size = get_file_size(file_to_scan)
    file_ext = current_filename.lower().split('.')[-1]
    
    # 存储文件大小到缓存
    st.session_state.file_size_cache = file_size
    
    # 显示文件大小信息
    size_mb = file_size / (1024 * 1024)
    if size_mb < 1:
        st.info(f"文件大小: {file_size/1024:.1f} KB")
    else:
        st.info(f"文件大小: {size_mb:.1f} MB")
    
    total_mols = 0
    preview_data = []
    df_preview = None
    scan_successful = False
    is_estimated = False
    
    if file_ext == "csv":
        try:
            # CSV文件使用pandas的优化读取
            df_preview = pd.read_csv(file_to_scan, nrows=PREVIEW_SIZE)
            if hasattr(file_to_scan, 'seek'):
                file_to_scan.seek(0)
            
            if smiles_column in df_preview.columns:
                st.dataframe(df_preview, height=200)
                st.caption(f"显示前 {len(df_preview)} 行的预览")
                scan_successful = True
                
                # 对于大CSV文件，延迟计算总行数
                if file_size > LARGE_FILE_THRESHOLD:
                    total_mols = -1  # 延迟计算
                    st.info("⚡ 大文件检测：总行数将在处理时计算以提高响应速度")
                else:
                    # 小文件直接计算
                    if isinstance(file_to_scan, str):
                        total_mols = sum(1 for _ in open(file_to_scan, 'r')) - 1  # 减去标题行
                    else:
                        total_mols = -1  # UploadedFile延迟计算
            else:
                st.error(f"在CSV中未找到SMILES列 '{smiles_column}'")
        except Exception as e:
            st.error(f"读取CSV预览时出错: {e}")
    
    elif file_ext in ["txt", "smi"]:
        if isinstance(file_to_scan, str):
            # 根据文件大小选择不同策略
            if file_size > HUGE_FILE_THRESHOLD:
                # 超大文件：仅采样预览，不计算总数
                st.warning("⚡ 超大文件检测：使用采样预览模式")
                preview_data, estimated_total = sample_lines_from_file(file_to_scan, PREVIEW_SIZE)
                total_mols = estimated_total
                is_estimated = True
                scan_successful = True
                
            elif file_size > LARGE_FILE_THRESHOLD:
                # 大文件：快速计算总数
                st.info("⚡ 大文件检测：使用优化扫描模式")
                start_time = time.time()
                total_mols = fast_line_count(file_to_scan)
                scan_time = time.time() - start_time
                st.success(f"快速扫描完成 ({scan_time:.2f}秒)")
                
                # 读取前几行作为预览
                with open(file_to_scan, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= PREVIEW_SIZE:
                            break
                        line = line.strip()
                        if line:
                            preview_data.append(line)
                scan_successful = True
                
            else:
                # 小文件：常规扫描
                lines = []
                with open(file_to_scan, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        line = line.strip()
                        if line:
                            lines.append(line)
                            if i < PREVIEW_SIZE:
                                preview_data.append(line)
                total_mols = len(lines)
                scan_successful = True
        else:
            # UploadedFile对象
            if hasattr(file_to_scan, 'seek'):
                file_to_scan.seek(0)
            
            lines = []
            with io.TextIOWrapper(file_to_scan, encoding="utf-8") as text_reader:
                for i, line in enumerate(text_reader):
                    line = line.strip()
                    if line:
                        lines.append(line)
                        if i < PREVIEW_SIZE:
                            preview_data.append(line)
            
            total_mols = len(lines)
            scan_successful = True
            
            if hasattr(file_to_scan, 'seek'):
                file_to_scan.seek(0)
        
        if scan_successful and preview_data:
            st.text_area("SMILES预览:", "\n".join(preview_data), height=150)
    
    elif file_ext == "sdf":
        if isinstance(file_to_scan, str):
            if file_size > HUGE_FILE_THRESHOLD:
                # 超大SDF文件：仅估算
                estimated_count = estimate_sdf_molecule_count(file_to_scan)
                if estimated_count:
                    st.warning(f"⚡ 超大SDF文件检测：估算约 {estimated_count:,} 个分子")
                    total_mols = estimated_count
                    is_estimated = True
                    
                    # 读取前几个分子作为预览
                    supplier = Chem.ForwardSDMolSupplier(file_to_scan, removeHs=False, sanitize=True)
                    temp_preview_mols = []
                    for i, mol in enumerate(supplier):
                        if i >= PREVIEW_SIZE:
                            break
                        if mol is not None:
                            temp_preview_mols.append(mol)
                    
                    preview_data = [Chem.MolToSmiles(m) if m else "无效分子" for m in temp_preview_mols]
                    scan_successful = True
                else:
                    st.warning("无法估算SDF文件大小，将使用完整扫描")
            
            if not scan_successful:  # 如果估算失败，使用常规方法
                if file_size > LARGE_FILE_THRESHOLD:
                    st.info("⚡ 大SDF文件检测：正在计算分子数量...")
                
                supplier = Chem.ForwardSDMolSupplier(file_to_scan, removeHs=False, sanitize=True)
                temp_preview_mols = []
                count = 0
                
                start_time = time.time()
                for i, mol in enumerate(supplier):
                    if mol is not None:
                        count += 1
                        if i < PREVIEW_SIZE:
                            temp_preview_mols.append(mol)
                    
                    # 每1000个分子更新一次进度（仅对大文件）
                    if file_size > LARGE_FILE_THRESHOLD and i % 1000 == 0 and i > 0:
                        elapsed = time.time() - start_time
                        # 使用更优雅的进度显示方式
                        if not hasattr(st.session_state, 'scan_status_placeholder'):
                            st.session_state.scan_status_placeholder = st.empty()
                        st.session_state.scan_status_placeholder.info(f"⏳ 扫描进度: {i:,} 个条目 ({elapsed:.1f}秒)")
                
                total_mols = count
                preview_data = [Chem.MolToSmiles(m) if m else "无效分子" for m in temp_preview_mols]
                scan_successful = True
        else:
            # UploadedFile SDF
            if hasattr(file_to_scan, 'seek'):
                file_to_scan.seek(0)
            sdf_stream = io.BytesIO(file_to_scan.getvalue())
            supplier = Chem.ForwardSDMolSupplier(sdf_stream, removeHs=False, sanitize=True)
            
            temp_preview_mols = []
            count = 0
            for i, mol in enumerate(supplier):
                if mol is not None:
                    count += 1
                    if i < PREVIEW_SIZE:
                        temp_preview_mols.append(mol)
            
            total_mols = count
            preview_data = [Chem.MolToSmiles(m) if m else "无效分子" for m in temp_preview_mols]
            scan_successful = True
            
            if hasattr(file_to_scan, 'seek'):
                file_to_scan.seek(0)
        
        if scan_successful and preview_data:
            st.text_area("SDF中的SMILES预览:", "\n".join(preview_data), height=150)
            
            if is_estimated:
                st.warning(f"📊 估算约 {total_mols:,} 个分子（基于文件采样）")
            else:
                st.info(f"在SDF文件中找到 {total_mols:,} 个分子")
    
    return total_mols, preview_data, df_preview, scan_successful, is_estimated

st.set_page_config(page_title="3D构象生成", layout="wide")
st.title("🧬 3D构象生成")

st.markdown("""
从分子的SMILES字符串或SDF文件生成3D构象。支持CSV、TXT、SMI和SDF格式。
使用RDKit的ETKDGv3算法生成构象。

🚀 **性能优化**: 自动检测大文件并使用优化策略处理百万级分子库。
⚡ **智能缓存**: 避免重复扫描，提升用户体验。
""")

# 缓存管理区域
if st.session_state.scan_results_valid:
    with st.expander("🗂️ 缓存管理", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.scan_timestamp_cache:
                cache_time = time.strftime("%Y-%m-%d %H:%M:%S", 
                                         time.localtime(st.session_state.scan_timestamp_cache))
                st.info(f"缓存时间: {cache_time}")
        with col2:
            if st.button("🔄 清除缓存", help="强制重新扫描文件"):
                # 清除所有相关缓存
                st.session_state.scan_results_valid = False
                st.session_state.file_hash_cache = None
                st.session_state.last_processed_file_identifier = None
                st.session_state.scan_timestamp_cache = None
                st.session_state.total_potential_mols_cache = 0
                st.session_state.preview_data_cache = []
                st.session_state.df_preview_cache = None
                st.session_state.initial_scan_successful_cache = False
                st.session_state.is_estimated_cache = False
                # 清除状态占位符
                if hasattr(st.session_state, 'scan_status_placeholder'):
                    delattr(st.session_state, 'scan_status_placeholder')
                if hasattr(st.session_state, 'parse_status_placeholder'):
                    delattr(st.session_state, 'parse_status_placeholder')
                st.success("✅ 缓存已清除！")
                st.rerun()

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
        "上传SMILES文件(CSV、TXT、SMI)或SDF文件",
        type=["csv", "txt", "smi", "sdf"],
        help="最大文件大小约4GB。支持百万级分子库的高效处理。"
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
                st.warning(f"选中的文件夹中没有找到文件: {selected_folder}")
            else:
                # 文件选择
                selected_filename = st.selectbox(
                    "选择输入文件:",
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

# CSV文件的SMILES列名设置
smiles_column = "SMILES"
if current_filename and current_filename.lower().endswith(".csv"):
    smiles_column = st.text_input("输入包含SMILES的列名:", "SMILES")

# 构象生成设置
st.subheader("2. 构象生成设置")
col1, col2, col3 = st.columns(3)
with col1:
    num_conformers = st.number_input("每个分子的构象数:", min_value=1, max_value=100, value=10)
with col2:
    max_attempts = st.number_input("最大嵌入尝试次数:", min_value=1, max_value=500, value=100)
with col3:
    random_seed = st.number_input("随机种子:", min_value=-1, value=42, help="-1表示不指定种子")

# 并行处理设置
st.subheader("3. 并行处理设置")
num_threads = st.number_input(
    "工作线程数:",
    min_value=1,
    max_value=multiprocessing.cpu_count() * 2,
    value=min(36, multiprocessing.cpu_count() * 2),
    help="用于并行处理分子的线程数"
)

# 处理范围设置
st.subheader("4. 处理范围")
processing_options = ["处理所有分子", "仅处理前500个分子", "仅处理前10000个分子", "仅处理前100000个分子"]
selected_scope = st.radio("选择处理范围:", options=processing_options, index=0)

# 文件处理和预览
input_ready = (uploaded_file is not None) or (selected_file_path is not None and os.path.exists(selected_file_path))

if input_ready:
    st.header("文件预览和生成控制")
    st.markdown(f"**当前文件:** `{current_filename}`")

    # 使用改进的缓存机制
    file_to_scan = uploaded_file if uploaded_file else selected_file_path
    cache_valid = is_cache_valid(file_to_scan)
    
    if cache_valid:
        st.success("✅ 使用缓存的扫描结果 (文件未变更)")
        total_mols = st.session_state.total_potential_mols_cache
        preview_data = st.session_state.preview_data_cache
        df_preview = st.session_state.df_preview_cache
        scan_successful = st.session_state.initial_scan_successful_cache
        is_estimated = getattr(st.session_state, 'is_estimated_cache', False)

        # 显示缓存的预览
        if df_preview is not None:
            st.dataframe(df_preview, height=200)
            st.caption(f"显示前 {len(df_preview)} 行的预览")
        elif preview_data:
            file_ext = current_filename.lower().split('.')[-1]
            if file_ext == "sdf":
                st.text_area("SDF中的SMILES预览:", "\n".join(preview_data), height=150)
            else:
                st.text_area("SMILES预览:", "\n".join(preview_data), height=150)
        
        # 显示文件大小
        if st.session_state.file_size_cache:
            size_mb = st.session_state.file_size_cache / (1024 * 1024)
            if size_mb < 1:
                st.info(f"文件大小: {st.session_state.file_size_cache/1024:.1f} KB")
            else:
                st.info(f"文件大小: {size_mb:.1f} MB")

    else:  # 执行新的智能扫描
        try:
            with st.spinner(f"正在智能扫描文件 '{current_filename}'..."):
                total_mols, preview_data, df_preview, scan_successful, is_estimated = smart_file_scan(
                    file_to_scan, current_filename, smiles_column
                )

            # 更新缓存，包括新的哈希值
            if scan_successful:
                current_hash = get_file_hash(file_to_scan)
                current_identifier = file_to_scan if isinstance(file_to_scan, str) else getattr(file_to_scan, 'name', str(file_to_scan))
                
                st.session_state.total_potential_mols_cache = total_mols
                st.session_state.preview_data_cache = preview_data
                st.session_state.df_preview_cache = df_preview
                st.session_state.initial_scan_successful_cache = True
                st.session_state.scan_results_valid = True
                st.session_state.is_estimated_cache = is_estimated
                st.session_state.file_hash_cache = current_hash
                st.session_state.last_processed_file_identifier = current_identifier
                st.session_state.scan_timestamp_cache = time.time()
            else:
                st.session_state.scan_results_valid = False

        except Exception as e:
            st.error(f"扫描文件时出错: {e}")
            st.session_state.scan_results_valid = False
            scan_successful = False
            is_estimated = False

    # 显示扫描状态
    if scan_successful:
        file_ext = current_filename.lower().split('.')[-1]
        if total_mols == -1:
            st.info("CSV文件预览完成。开始生成时将计算总条目数。")
        elif is_estimated:
            st.success(f"智能扫描完成: 估算约 {total_mols:,} 个条目")
        elif file_ext != "sdf":
            st.success(f"扫描完成: 在 '{current_filename}' 中找到 {total_mols:,} 个条目")

    # 生成按钮和处理逻辑
    if scan_successful and (total_mols > 0 or total_mols == -1):
        st.markdown("---")
        
        # 确定处理限制
        limit_map = {
            processing_options[0]: float('inf'),  # 处理所有
            processing_options[1]: 500,
            processing_options[2]: 10000,
            processing_options[3]: 100000
        }
        processing_limit = limit_map[selected_scope]
        
        # 确定按钮标签
        if total_mols == -1:
            if processing_limit == float('inf'):
                mol_count_label = "CSV中的所有分子"
            else:
                mol_count_label = f"前{int(processing_limit)}个(如果CSV中有足够数据)"
        else:
            if processing_limit == float('inf'):
                mol_count_label = f"{total_mols:,}"
            else:
                actual_count = min(total_mols, int(processing_limit))
                mol_count_label = f"{actual_count:,}"

        # 显示范围信息
        if processing_limit != float('inf'):
            if total_mols > processing_limit:
                st.info(f"范围选择: 将处理 {total_mols:,} 个条目中的前{int(processing_limit):,}个")
            elif total_mols == -1:
                st.info(f"范围选择: 将尝试处理CSV中的前{int(processing_limit):,}个条目")
        
        # 大文件处理建议
        if st.session_state.file_size_cache and st.session_state.file_size_cache > LARGE_FILE_THRESHOLD:
            st.warning("⚡ **大文件建议**: 对于百万级分子库，建议选择较小的处理范围以获得更好的性能")

        button_label = f"为 {mol_count_label} 个分子生成 {num_conformers} 个构象 (使用 {num_threads} 个线程)"
        
        if st.button(button_label):
            # 解析和加载分子
            input_mols = []
            source_ids = []
            limit = int(processing_limit) if processing_limit != float('inf') else float('inf')

            with st.spinner(f"正在从 '{current_filename}' 加载和解析分子..."):
                try:
                    current_file = uploaded_file if uploaded_file else selected_file_path
                    if hasattr(current_file, 'seek'):
                        current_file.seek(0)
                    
                    file_ext = current_filename.lower().split('.')[-1]

                    if file_ext == "csv":
                        # 使用分块读取处理大CSV文件
                        if st.session_state.file_size_cache > LARGE_FILE_THRESHOLD:
                            st.info("⚡ 使用分块读取模式处理大CSV文件...")
                            chunk_size = 10000
                            total_processed = 0
                            
                            for chunk in pd.read_csv(current_file, chunksize=chunk_size):
                                if smiles_column in chunk.columns:
                                    for idx, row in chunk.iterrows():
                                        if len(input_mols) >= limit:
                                            break
                                        smi = str(row[smiles_column])
                                        mol = Chem.MolFromSmiles(smi)
                                        if mol:
                                            input_mols.append(mol)
                                            source_ids.append(f"行 {total_processed + idx + 1}: {smi}")
                                    total_processed += len(chunk)
                                    if len(input_mols) >= limit:
                                        break
                                    if total_processed % 50000 == 0:
                                        # 使用状态占位符而不是频繁输出
                                        if not hasattr(st.session_state, 'parse_status_placeholder'):
                                            st.session_state.parse_status_placeholder = st.empty()
                                        st.session_state.parse_status_placeholder.info(f"🔄 解析进度: {total_processed:,} 行，{len(input_mols):,} 个有效分子")
                        else:
                            df_full = pd.read_csv(current_file)
                            csv_row_count = len(df_full)
                            st.info(f"CSV完全读取完成。包含 {csv_row_count:,} 行数据。开始解析...")

                        if smiles_column in df_full.columns:
                            for idx, row in df_full.iterrows():
                                if len(input_mols) >= limit:
                                    break
                                smi = str(row[smiles_column])
                                mol = Chem.MolFromSmiles(smi)
                                if mol:
                                        input_mols.append(mol)
                                        source_ids.append(f"行 {idx+1}: {smi}")
                            else:
                                st.error(f"SMILES列 '{smiles_column}' 消失了？请重新检查。")
                            st.stop()

                    elif file_ext in ["txt", "smi"]:
                        if isinstance(current_file, str):
                            with open(current_file, "r", encoding="utf-8") as f:
                                for i, line in enumerate(f):
                                    if len(input_mols) >= limit:
                                        break
                                    smi = line.strip()
                                    if smi:
                                        mol = Chem.MolFromSmiles(smi)
                                        if mol:
                                            input_mols.append(mol)
                                            source_ids.append(f"行 {i+1}: {smi}")
                                    
                                    # 进度更新（仅对大文件）
                                    if st.session_state.file_size_cache > LARGE_FILE_THRESHOLD and i % 10000 == 0 and i > 0:
                                        if not hasattr(st.session_state, 'parse_status_placeholder'):
                                            st.session_state.parse_status_placeholder = st.empty()
                                        st.session_state.parse_status_placeholder.info(f"🔄 解析进度: {i:,} 行，{len(input_mols):,} 个有效分子")
                        else:
                            with io.TextIOWrapper(current_file, encoding="utf-8") as text_reader:
                                for i, line in enumerate(text_reader):
                                    if len(input_mols) >= limit:
                                        break
                                    smi = line.strip()
                                    if smi:
                                        mol = Chem.MolFromSmiles(smi)
                                        if mol:
                                            input_mols.append(mol)
                                            source_ids.append(f"行 {i+1}: {smi}")

                    elif file_ext == "sdf":
                        if isinstance(current_file, str):
                            supplier = Chem.ForwardSDMolSupplier(current_file, removeHs=False, sanitize=True)
                        else:
                            sdf_stream = io.BytesIO(current_file.getvalue())
                            supplier = Chem.ForwardSDMolSupplier(sdf_stream, removeHs=False, sanitize=True)
                        
                        for i, mol in enumerate(supplier):
                            if len(input_mols) >= limit:
                                break
                            if mol is not None:
                                input_mols.append(mol)
                                source_ids.append(f"SDF分子 {i+1} ({Chem.MolToSmiles(mol)})")
                            
                            # 进度更新（仅对大文件）
                            if st.session_state.file_size_cache > LARGE_FILE_THRESHOLD and i % 1000 == 0 and i > 0:
                                if not hasattr(st.session_state, 'parse_status_placeholder'):
                                    st.session_state.parse_status_placeholder = st.empty()
                                st.session_state.parse_status_placeholder.info(f"🔄 解析进度: {i:,} 个SDF条目，{len(input_mols):,} 个有效分子")
                    
                    if not input_mols:
                        st.warning("基于当前范围，没有解析到有效分子用于生成。")
                        st.stop()
                        
                except Exception as e:
                    st.error(f"完整解析 '{current_filename}' 时出错: {e}")
                    st.stop()
            
            # 构象生成处理
            if input_mols:
                st.info(f"🚀 开始为 {len(input_mols):,} 个分子生成构象...")
                
                # 创建进度条和状态显示区域
                progress_container = st.container()
                with progress_container:
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    stats_col1, stats_col2, stats_col3 = st.columns(3)
                    with stats_col1:
                        completed_metric = st.empty()
                    with stats_col2:
                        success_metric = st.empty()
                    with stats_col3:
                        speed_metric = st.empty()
                
                processed_mols = []
                success_count = 0
                errors = []
                
                tasks = [(mol, num_conformers, max_attempts, random_seed) for mol in input_mols]

                completed = 0
                start_time = time.time()
                last_update_time = start_time
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                    future_to_idx = {executor.submit(generate_conformers_for_mol, *task): i
                    for i, task in enumerate(tasks)}
                    
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        identifier = source_ids[idx]
                        
                        try:
                            result = future.result()
                            if result['success'] and result['mol']:
                                processed_mols.append(result['mol'])
                                success_count += 1
                            else:
                                errors.append(f"警告 {identifier}: {result['message']}")
                        except Exception as exc:
                            errors.append(f"处理错误 {identifier}: {exc}")
                        
                        completed += 1
                        progress = completed / len(input_mols)
                        current_time = time.time()
                        
                        # 限制更新频率，每0.5秒或每完成100个分子更新一次
                        if (current_time - last_update_time > 0.5) or (completed % 100 == 0) or (completed == len(input_mols)):
                            elapsed_time = current_time - start_time
                            
                            # 更新进度条
                            progress_bar.progress(progress)
                            
                            # 更新指标
                            completed_metric.metric("已完成", f"{completed:,}/{len(input_mols):,}")
                            success_metric.metric("成功率", f"{success_count/completed*100:.1f}%" if completed > 0 else "0%")
                            
                            if completed > 10:  # 避免初期估算不准
                                avg_time_per_mol = elapsed_time / completed
                                remaining_time = avg_time_per_mol * (len(input_mols) - completed)
                                speed = completed / elapsed_time
                                speed_metric.metric("处理速度", f"{speed:.1f} 分子/秒")
                                status_text.info(f"⏱️ 预计剩余时间: {remaining_time/60:.1f} 分钟")
                            else:
                                speed_metric.metric("处理速度", "计算中...")
                                status_text.info(f"🔄 处理中: {completed:,}/{len(input_mols):,} 分子完成")
                            
                            last_update_time = current_time

                total_time = time.time() - start_time
                status_text.success(f"✅ 构象生成完成！总耗时: {total_time/60:.1f} 分钟")
                st.success(f"🎉 成功为 {success_count:,}/{len(input_mols):,} 个分子生成了构象！")

                if errors:
                    with st.expander("⚠️ 查看错误和警告", expanded=False):
                        st.markdown("\n".join([f"- {msg}" for msg in errors[:100]]))  # 只显示前100个错误
                        if len(errors) > 100:
                            st.warning(f"还有 {len(errors)-100} 个错误未显示...")

                if processed_mols:
                    sdf_output = mols_to_sdf_string(processed_mols)
                    
                    # 保存到工作目录（如果有的话）
                    output_filename = f"generated_conformers_{current_filename}.sdf"
                    if work_dir and os.path.exists(work_dir):
                        output_path = os.path.join(work_dir, output_filename)
                        try:
                            with open(output_path, 'w') as f:
                                f.write(sdf_output)
                            st.success(f"结果已保存到工作目录: {output_path}")
                        except Exception as e:
                            st.warning(f"保存到工作目录失败: {e}")
                    
                        st.download_button(
                        label="📥 下载生成的构象 (SDF)",
                        data=sdf_output,
                        file_name=output_filename,
                            mime="chemical/x-mdl-sdfile",
                        )
                    
                    # 显示文件大小信息
                    output_size_mb = len(sdf_output.encode('utf-8')) / (1024 * 1024)
                    st.info(f"生成的SDF文件大小: {output_size_mb:.1f} MB")
                    
                    st.subheader("生成的SDF预览 (前1000字符)")
                    st.code(sdf_output[:1000], language="text")
                else:
                    st.warning("没有成功生成构象来创建SDF文件。")

else:
    if input_method == "上传新文件":
        st.info("👆 请在上方上传SMILES或SDF文件开始使用。")
    else:
        st.info("👆 请选择已保存的数据文件夹和文件开始使用。") 