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
import time
import subprocess

from utils.background_conformer_generator import run_background_conformer_generation

# 简化的后台运行功能，现在不依赖外部工具
BACKGROUND_CONFORMER_AVAILABLE = True

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
# 简化的后台任务管理
if 'simple_background_tasks' not in st.session_state:
    st.session_state.simple_background_tasks = []

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
                scan_successful = True
                
                if file_size > LARGE_FILE_THRESHOLD:
                    total_mols = -1
                    st.info("⚡ 大文件检测：总行数将在处理时计算以提高响应速度")
                else:
                    if isinstance(file_to_scan, str):
                        total_mols = sum(1 for _ in open(file_to_scan, 'r')) - 1
                    else:
                        total_mols = -1
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
        
        if scan_successful and is_estimated:
            st.warning(f"📊 估算约 {total_mols:,} 个分子（基于文件采样）")
        elif scan_successful:
            st.info(f"在SDF文件中找到 {total_mols:,} 个分子")
    
    return total_mols, preview_data, df_preview, scan_successful, is_estimated


def render_file_preview_section(input_ready, input_method, uploaded_file, selected_file_path,
                                current_filename, smiles_column):
    """渲染数据预览区域，并返回扫描的关键信息"""
    total_mols = 0
    scan_successful = False
    is_estimated = False
    df_preview = None
    preview_data = []

    if not input_ready:
        if input_method == "上传新文件":
            st.info("👆 请先上传并保存文件到工作目录。")
        else:
            st.info("👆 请先从工作目录中选择需要处理的文件。")
        return total_mols, scan_successful, is_estimated

    file_to_scan = uploaded_file if uploaded_file else selected_file_path
    cache_valid = is_cache_valid(file_to_scan)

    st.markdown(f"**当前文件:** `{current_filename}`")

    if cache_valid:
        st.success("✅ 使用缓存的扫描结果 (文件未变更)")
        total_mols = st.session_state.total_potential_mols_cache
        preview_data = st.session_state.preview_data_cache
        df_preview = st.session_state.df_preview_cache
        scan_successful = st.session_state.initial_scan_successful_cache
        is_estimated = getattr(st.session_state, 'is_estimated_cache', False)

        if df_preview is not None:
            st.dataframe(df_preview, height=200)
            st.caption(f"显示前 {len(df_preview)} 行的预览")
        elif preview_data:
            file_ext = current_filename.lower().split('.')[-1]
            label = "SDF中的SMILES预览:" if file_ext == "sdf" else "SMILES预览:"
            st.text_area(label, "\n".join(preview_data), height=150)

        if st.session_state.file_size_cache:
            size_mb = st.session_state.file_size_cache / (1024 * 1024)
            if size_mb < 1:
                st.info(f"文件大小: {st.session_state.file_size_cache/1024:.1f} KB")
            else:
                st.info(f"文件大小: {size_mb:.1f} MB")
    else:
        try:
            with st.spinner(f"正在智能扫描文件 '{current_filename}'..."):
                total_mols, preview_data, df_preview, scan_successful, is_estimated = smart_file_scan(
                    file_to_scan, current_filename, smiles_column
                )

            if scan_successful:
                current_hash = get_file_hash(file_to_scan)
                current_identifier = file_to_scan if isinstance(file_to_scan, str) else getattr(
                    file_to_scan, 'name', str(file_to_scan)
                )

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

        if scan_successful:
            file_ext = current_filename.lower().split('.')[-1]
            if df_preview is not None:
                st.dataframe(df_preview, height=200)
                st.caption(f"显示前 {len(df_preview)} 行的预览")
            elif preview_data:
                label = "SDF中的SMILES预览:" if file_ext == "sdf" else "SMILES预览:"
                st.text_area(label, "\n".join(preview_data), height=150)

    if scan_successful:
        file_ext = current_filename.lower().split('.')[-1]
        if total_mols == -1:
            st.info("CSV文件预览完成。开始生成时将计算总条目数。")
        elif is_estimated:
            st.success(f"智能扫描完成: 估算约 {total_mols:,} 个条目")
        elif file_ext != "sdf":
            st.success(f"扫描完成: 在 '{current_filename}' 中找到 {total_mols:,} 个条目")
    else:
        if input_method == "上传新文件":
            st.info("👆 请确保文件已上传并保存到工作目录。")
        else:
            st.info("👆 请确认目标文件可访问。")

    return total_mols, scan_successful, is_estimated

def create_simple_run_script(work_dir, script_name):
    """创建简单的后台运行脚本"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_script_name = f"run_conformer_{timestamp}"
    
    run_script_content = f'''#!/bin/bash
# 简单的构象生成后台运行脚本
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SCRIPT_NAME="{run_script_name}"
WORK_DIR="{work_dir}"
MAIN_SCRIPT="conformer_generation.py"

echo "=================================================="
echo "启动构象生成任务: $SCRIPT_NAME"
echo "工作目录: $WORK_DIR"
echo "开始时间: $(date)"
echo "=================================================="

# 切换到工作目录
cd "$WORK_DIR"

# 检查主脚本是否存在
if [ ! -f "$MAIN_SCRIPT" ]; then
    echo "❌ 错误: 找不到 $MAIN_SCRIPT"
    echo "请先点击'生成构象脚本'按钮生成脚本文件"
    exit 1
fi

# 创建日志文件
LOG_FILE="${{SCRIPT_NAME}}.log"
PROGRESS_FILE="${{SCRIPT_NAME}}.progress"
PID_FILE="${{SCRIPT_NAME}}.pid"

# 保存当前进程ID
echo $$ > "$PID_FILE"

# 开始运行主脚本
echo "🚀 开始运行构象生成..."
echo "📝 日志文件: $LOG_FILE"

# 运行主脚本并记录日志
python "$MAIN_SCRIPT" 2>&1 | tee "$LOG_FILE"

# 检查运行结果
EXIT_CODE=${{PIPESTATUS[0]}}
END_TIME=$(date)

echo "=================================================="
echo "任务完成时间: $END_TIME"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 构象生成成功完成！"
    
    # 创建完成标志文件
    echo "$END_TIME" > "${{SCRIPT_NAME}}.completed"
    echo "SUCCESS" >> "${{SCRIPT_NAME}}.completed"
    echo "Exit code: $EXIT_CODE" >> "${{SCRIPT_NAME}}.completed"
    
    # 查找输出文件
    OUTPUT_FILES=$(find . -name "generated_conformers_*.sdf" -o -name "conformers_*.sdf" 2>/dev/null)
    if [ -n "$OUTPUT_FILES" ]; then
        echo "输出文件:" >> "${{SCRIPT_NAME}}.completed"
        echo "$OUTPUT_FILES" >> "${{SCRIPT_NAME}}.completed"
    fi
    
else
    echo "❌ 构象生成失败，退出码: $EXIT_CODE"
    
    # 创建错误标志文件
    echo "$END_TIME" > "${{SCRIPT_NAME}}.error"
    echo "FAILED" >> "${{SCRIPT_NAME}}.error"
    echo "Exit code: $EXIT_CODE" >> "${{SCRIPT_NAME}}.error"
fi

# 清理PID文件
rm -f "$PID_FILE"

exit $EXIT_CODE
'''
    
    return run_script_content, run_script_name

def simple_run_background_conformer(work_dir):
    """简单的后台运行功能"""
    # 创建运行脚本
    run_script_content, run_script_name = create_simple_run_script(work_dir, "conformer")
    
    # 保存运行脚本
    run_script_path = os.path.join(work_dir, f"{run_script_name}.sh")
    with open(run_script_path, 'w') as f:
        f.write(run_script_content)
    
    # 设置执行权限
    import stat
    os.chmod(run_script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    
    # 在后台运行脚本
    cmd = f"cd {work_dir} && nohup bash {run_script_name}.sh > /dev/null 2>&1 &"
    os.system(cmd)
    
    return run_script_path, run_script_name

def simple_check_background_status(work_dir, script_name):
    """简单的后台任务状态检查"""
    # 检查PID文件
    pid_file = os.path.join(work_dir, f"{script_name}.pid")
    if os.path.exists(pid_file):
        try:
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
        except Exception:
            pass
    
    # 检查完成标志
    completion_file = os.path.join(work_dir, f"{script_name}.completed")
    if os.path.exists(completion_file):
        try:
            with open(completion_file, 'r') as f:
                content = f.read()
            return {'status': 'completed', 'details': content}
        except Exception:
            return {'status': 'completed', 'details': '任务已完成'}
    
    # 检查错误标志
    error_file = os.path.join(work_dir, f"{script_name}.error")
    if os.path.exists(error_file):
        try:
            with open(error_file, 'r') as f:
                content = f.read()
            return {'status': 'error', 'details': content}
        except Exception:
            return {'status': 'error', 'details': '任务执行出错'}
    
    return {'status': 'unknown'}

def generate_conformer_script(input_file, output_file, num_conformers, max_attempts, random_seed, num_workers, processing_limit, file_type, smiles_column="SMILES"):
    """生成独立的构象生成脚本，并写入输出目录"""
    # 使用相对路径的文件名，因为脚本会在工作目录中运行
    input_filename = os.path.basename(input_file)
    output_filename = os.path.basename(output_file)
    log_filename = output_filename.replace('.sdf', '.log')
    output_dir = os.path.dirname(output_file)
    
    # 转换变量为字符串
    num_conformers_str = str(num_conformers)
    max_attempts_str = str(max_attempts)
    random_seed_str = str(random_seed)
    num_workers_str = str(num_workers)
    
    # 特殊处理 processing_limit
    if processing_limit == float('inf'):
        processing_limit_str = "float('inf')"
    else:
        processing_limit_str = str(processing_limit)
    
    script_content = f'''#!/usr/bin/env python3
"""
独立的多进程3D构象生成脚本 - Debug增强版
由Streamlit应用自动生成
增加了详细的调试信息和问题诊断功能

注意：此脚本需要在包含输入文件的工作目录中运行
输入文件: {input_filename}
输出文件: {output_filename}
"""

import os
import sys
import time
import logging
import pandas as pd
import io
from datetime import datetime
from multiprocessing import Pool
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
import traceback
import signal
import psutil
import gc
import resource

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem

# 设置日志
log_file = "{log_filename}"
logging.basicConfig(
    level=logging.DEBUG,  # 更详细的日志级别
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, 'w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 配置参数
CONFIG = {{
    'input_file': '{input_filename}',
    'output_file': '{output_filename}',
    'num_conformers': {num_conformers_str},
    'max_attempts': {max_attempts_str},
    'random_seed': {random_seed_str},
    'num_workers': {num_workers_str},
    'processing_limit': {processing_limit_str},
    'file_type': '{file_type}',
    'smiles_column': '{smiles_column}'
}}

# 全局计数器和锁
import threading
task_counter_lock = threading.Lock()
completed_tasks = 0
failed_tasks = 0

def signal_handler(signum, frame):
    """信号处理器，用于优雅关闭"""
    logger.warning(f"收到信号 {{signum}}，正在优雅关闭...")
    sys.exit(1)

# 注册信号处理器
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

def log_system_info():
    """记录系统资源信息"""
    try:
        # 内存信息
        memory = psutil.virtual_memory()
        logger.info(f"系统内存: 总计{{memory.total/1024**3:.1f}}GB, 可用{{memory.available/1024**3:.1f}}GB, 使用率{{memory.percent}}%")
        
        # CPU信息
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"CPU: {{cpu_count}}核, 当前使用率{{cpu_percent}}%")
        
        # 进程信息
        current_process = psutil.Process()
        proc_memory = current_process.memory_info()
        logger.info(f"当前进程内存: RSS={{proc_memory.rss/1024**2:.1f}}MB, VMS={{proc_memory.vms/1024**2:.1f}}MB")
        
    except Exception as e:
        logger.warning(f"获取系统信息失败: {{e}}")

def generate_conformers_for_mol(args):
    """多进程worker函数 - 为单个分子生成3D构象 - Debug增强版"""
    global completed_tasks, failed_tasks
    
    mol_data, mol_id, num_confs, max_attempts, random_seed = args
    
    start_time = time.time()
    process_pid = os.getpid()
    
    try:
        # 详细日志：开始处理
        logger.debug(f"[PID{{process_pid}}] 开始处理分子 {{mol_id}}")
        
        # 重建分子对象 - 智能判断数据类型
        if isinstance(mol_data, str):
            # 判断是SMILES还是MolBlock
            if "\\n" in mol_data and ("M  END" in mol_data or "V2000" in mol_data):
                # MolBlock格式
                logger.debug(f"[PID{{process_pid}}] 解析MolBlock，长度: {{len(mol_data)}}")
                mol = Chem.MolFromMolBlock(mol_data, removeHs=False, sanitize=False)
                if not mol:
                    logger.warning(f"[PID{{process_pid}}] 分子 {{mol_id}} MolBlock解析失败")
                    with task_counter_lock:
                        failed_tasks += 1
                    return (mol_id, None, "MolBlock解析失败")
            else:
                # SMILES字符串
                logger.debug(f"[PID{{process_pid}}] 解析SMILES: {{mol_data[:50]}}")
                mol = Chem.MolFromSmiles(mol_data)
                if not mol:
                    logger.warning(f"[PID{{process_pid}}] 分子 {{mol_id}} SMILES解析失败: {{mol_data}}")
                    with task_counter_lock:
                        failed_tasks += 1
                    return (mol_id, None, "SMILES解析失败")
        else:
            # 其他类型（不应该出现）
            logger.error(f"[PID{{process_pid}}] 分子 {{mol_id}} 未知数据类型: {{type(mol_data)}}")
            with task_counter_lock:
                failed_tasks += 1
            return (mol_id, None, "未知数据类型")
        
        logger.debug(f"[PID{{process_pid}}] 分子 {{mol_id}} 解析成功，原子数: {{mol.GetNumAtoms()}}")
        
        original_smiles = Chem.MolToSmiles(mol)
        logger.debug(f"[PID{{process_pid}}] 分子 {{mol_id}} SMILES: {{original_smiles}}")

        try:
            # 确保分子被正确sanitize（特别是对于从SDF加载的分子）
            logger.debug(f"[PID{{process_pid}}] 为分子 {{mol_id}} 执行sanitize")
            try:
                Chem.SanitizeMol(mol)
            except Exception as sanitize_error:
                logger.warning(f"[PID{{process_pid}}] 分子 {{mol_id}} sanitize警告: {{sanitize_error}}")
                # 尝试部分sanitize
                try:
                    Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL^Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                except Exception:
                    pass  # 继续尝试处理
            
            # 添加氢原子
            logger.debug(f"[PID{{process_pid}}] 为分子 {{mol_id}} 添加氢原子")
            mol_h = Chem.AddHs(mol)
            logger.debug(f"[PID{{process_pid}}] 分子 {{mol_id}} 添加氢后原子数: {{mol_h.GetNumAtoms()}}")
            
            # 设置ETKDG参数
            logger.debug(f"[PID{{process_pid}}] 设置ETKDG参数")
            params = AllChem.ETKDGv3()
            params.randomSeed = random_seed
            if hasattr(params, 'maxIterations'):
                params.maxIterations = max_attempts
            params.numThreads = 0  # 重要：每个进程内部不使用多线程
            
            logger.debug(f"[PID{{process_pid}}] 分子 {{mol_id}} 开始嵌入构象，目标数量: {{num_confs}}")
            
            # 生成构象 - 增加超时保护
            embed_start = time.time()
            cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=num_confs, params=params)
            embed_time = time.time() - embed_start
            
            if not cids or len(cids) == 0:
                logger.warning(f"[PID{{process_pid}}] 分子 {{mol_id}} 构象嵌入失败 (耗时{{embed_time:.2f}}s): {{original_smiles}}")
                with task_counter_lock:
                    failed_tasks += 1
                return (mol_id, None, f"RDKit构象生成失败: {{original_smiles}}")
            
            logger.debug(f"[PID{{process_pid}}] 分子 {{mol_id}} 成功生成 {{len(cids)}} 个构象 (耗时{{embed_time:.2f}}s)")
            
            # 转换为MolBlock以便传输
            logger.debug(f"[PID{{process_pid}}] 转换分子 {{mol_id}} 为MolBlock")
            mol_block = Chem.MolToMolBlock(mol_h)
            
            elapsed = time.time() - start_time
            logger.debug(f"[PID{{process_pid}}] 分子 {{mol_id}} 处理完成，总耗时: {{elapsed:.2f}}秒")
            
            with task_counter_lock:
                completed_tasks += 1
            
            return (mol_id, mol_block, f"成功生成{{len(cids)}}个构象")
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"异常: {{original_smiles}}: {{str(e)}}"
            logger.error(f"[PID{{process_pid}}] 分子 {{mol_id}} 构象生成异常 (耗时{{elapsed:.2f}}s): {{error_msg}}")
            logger.error(f"[PID{{process_pid}}] 分子 {{mol_id}} 异常堆栈: {{traceback.format_exc()}}")
            with task_counter_lock:
                failed_tasks += 1
            return (mol_id, None, error_msg)
            
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"Worker异常: {{str(e)}}"
        logger.error(f"[PID{{process_pid}}] 分子 {{mol_id}} Worker异常 (耗时{{elapsed:.2f}}s): {{error_msg}}")
        logger.error(f"[PID{{process_pid}}] Worker堆栈: {{traceback.format_exc()}}")
        with task_counter_lock:
            failed_tasks += 1
        return (mol_id, None, error_msg)

def load_molecules_from_file(file_path, file_type, smiles_column, processing_limit):
    """从文件加载分子数据"""
    molecules_data = []
    mol_ids = []
    
    logger.info(f"开始从文件加载分子: {{file_path}}")
    logger.info(f"文件类型: {{file_type}}, 处理限制: {{processing_limit}}")
    
    count = 0
    load_start_time = time.time()
    
    if file_type == "csv":
        df = pd.read_csv(file_path)
        logger.info(f"CSV文件包含 {{len(df)}} 行")
        logger.info(f"CSV列名: {{list(df.columns)}}")
        
        if smiles_column not in df.columns:
            raise ValueError(f"CSV文件中未找到SMILES列 '{{smiles_column}}'")
        
        logger.info(f"开始检查SMILES列: {{smiles_column}}")
        for idx, row in df.iterrows():
            if processing_limit != float('inf') and count >= processing_limit:
                break
            
            smi = str(row[smiles_column])
            mol = Chem.MolFromSmiles(smi)
            if mol:
                molecules_data.append(smi)  # 传递SMILES字符串
                mol_ids.append(f"row_{{idx+1}}")
                count += 1
                
                if count % 10000 == 0:
                    elapsed = time.time() - load_start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    logger.info(f"已加载 {{count}} 个有效分子 (速率: {{rate:.1f}} 分子/秒)")
    
    elif file_type in ["txt", "smi"]:
        logger.info(f"开始检查文本文件...")
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if processing_limit != float('inf') and count >= processing_limit:
                    break
                
                smi = line.strip()
                if smi:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        molecules_data.append(smi)  # 传递SMILES字符串
                        mol_ids.append(f"line_{{i+1}}")
                        count += 1
                        
                        if count % 10000 == 0:
                            elapsed = time.time() - load_start_time
                            rate = count / elapsed if elapsed > 0 else 0
                            logger.info(f"已加载 {{count}} 个有效分子 (速率: {{rate:.1f}} 分子/秒)")
    
    elif file_type == "sdf":
        logger.info(f"开始检查SDF文件...")
        supplier = Chem.ForwardSDMolSupplier(file_path, removeHs=False, sanitize=True)
        for i, mol in enumerate(supplier):
            if processing_limit != float('inf') and count >= processing_limit:
                break
            
            if mol is not None:
                mol_block = Chem.MolToMolBlock(mol)
                molecules_data.append(mol_block)  # 传递MolBlock
                mol_id = mol.GetProp('_Name') if mol.HasProp('_Name') else f"sdf_mol_{{i+1}}"
                mol_ids.append(mol_id)
                count += 1
                
                if count % 1000 == 0:
                    elapsed = time.time() - load_start_time
                    rate = count / elapsed if elapsed > 0 else 0
                    logger.info(f"已加载 {{count}} 个有效分子 (速率: {{rate:.1f}} 分子/秒)")
    
    load_elapsed = time.time() - load_start_time
    logger.info(f"总共加载了 {{len(molecules_data)}} 个有效分子，耗时: {{load_elapsed:.2f}}秒")
    return molecules_data, mol_ids

def process_batch_with_monitoring(tasks, num_workers, batch_size=1000):
    """分批处理任务，增加监控和错误恢复"""
    all_results = []
    total_tasks = len(tasks)
    batch_count = (total_tasks + batch_size - 1) // batch_size
    
    logger.info(f"将 {{total_tasks}} 个任务分为 {{batch_count}} 批，每批 {{batch_size}} 个")
    
    for batch_idx in range(0, total_tasks, batch_size):
        batch_end = min(batch_idx + batch_size, total_tasks)
        current_batch = tasks[batch_idx:batch_end]
        batch_num = batch_idx // batch_size + 1
        
        logger.info(f"开始处理第 {{batch_num}}/{{batch_count}} 批 ({{len(current_batch)}} 个任务)")
        
        batch_start_time = time.time()
        batch_results = []
        batch_success = 0
        batch_failures = 0
        
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                logger.debug(f"批次 {{batch_num}}: ProcessPoolExecutor已启动")
                
                # 提交当前批次的任务
                future_to_task = {{}}
                for i, task in enumerate(current_batch):
                    try:
                        future = executor.submit(generate_conformers_for_mol, task)
                        future_to_task[future] = (batch_idx + i, task[1])  # (全局索引, mol_id)
                    except Exception as e:
                        logger.error(f"提交任务失败: {{e}}")
                        batch_failures += 1
                
                logger.debug(f"批次 {{batch_num}}: 已提交 {{len(future_to_task)}} 个任务")
                
                # 处理结果，设置较短的超时
                completed_in_batch = 0
                for future in concurrent.futures.as_completed(future_to_task, timeout=1800):  # 30分钟批次超时
                    try:
                        global_idx, mol_id = future_to_task[future]
                        mol_id_result, mol_block, message = future.result(timeout=60)  # 单任务60秒超时
                        
                        if mol_block:
                            batch_results.append((mol_id_result, mol_block))
                            batch_success += 1
                        else:
                            logger.warning(f"批次 {{batch_num}}: 分子 {{mol_id}} 失败: {{message}}")
                            batch_failures += 1
                        
                        completed_in_batch += 1
                        
                        # 每50个任务报告一次进度
                        if completed_in_batch % 50 == 0:
                            batch_progress = completed_in_batch / len(current_batch) * 100
                            elapsed = time.time() - batch_start_time
                            rate = completed_in_batch / elapsed if elapsed > 0 else 0
                            logger.info(f"批次 {{batch_num}} 进度: {{completed_in_batch}}/{{len(current_batch)}} ({{batch_progress:.1f}}%), "
                                      f"成功: {{batch_success}}, 失败: {{batch_failures}}, 速率: {{rate:.1f}} 任务/秒")
                        
                    except concurrent.futures.TimeoutError:
                        global_idx, mol_id = future_to_task.get(future, (None, 'unknown'))
                        logger.error(f"批次 {{batch_num}}: 任务超时 - 分子 {{mol_id}}")
                        batch_failures += 1
                    except Exception as e:
                        global_idx, mol_id = future_to_task.get(future, (None, 'unknown'))
                        logger.error(f"批次 {{batch_num}}: 处理任务时出错 - 分子 {{mol_id}}: {{e}}")
                        batch_failures += 1
                
                batch_elapsed = time.time() - batch_start_time
                batch_rate = len(current_batch) / batch_elapsed if batch_elapsed > 0 else 0
                
                logger.info(f"批次 {{batch_num}} 完成: 成功 {{batch_success}}, 失败 {{batch_failures}}, "
                          f"耗时 {{batch_elapsed:.1f}}秒, 速率 {{batch_rate:.1f}} 任务/秒")
                
                all_results.extend(batch_results)
                
        except Exception as e:
            logger.error(f"批次 {{batch_num}} 处理异常: {{e}}")
            logger.error(f"异常堆栈: {{traceback.format_exc()}}")
        
        # 批次间强制垃圾回收
        gc.collect()
        
        # 记录系统状态
        if batch_num % 5 == 0:  # 每5批记录一次系统状态
            log_system_info()
    
    return all_results

def main():
    """主函数"""
    logger.info("开始多进程3D构象生成 - Debug增强版")
    logger.info("配置参数: %s", CONFIG)
    
    # 记录初始系统状态
    log_system_info()
    
    total_start_time = time.time()
    
    try:
        # 加载分子数据
        molecules_data, mol_ids = load_molecules_from_file(
            CONFIG['input_file'], 
            CONFIG['file_type'], 
            CONFIG['smiles_column'], 
            CONFIG['processing_limit']
        )
        
        if not molecules_data:
            logger.error("没有加载到有效分子")
            return
        
        # 准备多进程任务
        tasks = [
            (mol_data, mol_id, CONFIG['num_conformers'], CONFIG['max_attempts'], CONFIG['random_seed'])
            for mol_data, mol_id in zip(molecules_data, mol_ids)
        ]
        
        logger.info(f"准备处理 {{len(tasks)}} 个分子，使用 {{CONFIG['num_workers']}} 个进程")
        
        # 分批处理，降低内存压力
        batch_size = max(100, min(1000, len(tasks) // 10))  # 动态批次大小
        logger.info(f"使用批次大小: {{batch_size}}")
        
        # 使用改进的分批处理函数
        results = process_batch_with_monitoring(tasks, CONFIG['num_workers'], batch_size)
        
        # 保存结果
        if results:
            logger.info(f"开始保存 {{len(results)}} 个成功的分子构象")
            save_start_time = time.time()
            
            mol_count = 0
            conformer_count = 0
            
            with open(CONFIG['output_file'], 'w') as f:
                for mol_id, mol_block in results:
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
                        
                        if mol_count % 1000 == 0:
                            logger.info(f"已保存 {{mol_count}} 个分子，{{conformer_count}} 个构象")
                            
                    except Exception as e:
                        logger.error(f"保存分子 {{mol_id}} 时出错: {{e}}")
            
            save_elapsed = time.time() - save_start_time
            logger.info(f"保存完成，耗时: {{save_elapsed:.2f}}秒")
            logger.info(f"最终统计: {{mol_count}} 个分子，{{conformer_count}} 个构象")
            
            logger.info(f"构象生成完成！成功处理 {{len(results)}}/{{len(tasks)}} 个分子")
            logger.info(f"输出文件: {{CONFIG['output_file']}}")
        else:
            logger.error("没有成功生成任何构象")
        
        total_elapsed = time.time() - total_start_time
        logger.info(f"总耗时: {{total_elapsed/60:.1f}} 分钟")
        
        if len(tasks) > 0:
            success_rate = len(results) / len(tasks) * 100
            logger.info(f"最终成功率: {{success_rate:.2f}}%")
        
        # 最终系统状态
        log_system_info()
        
    except Exception as e:
        logger.error(f"脚本执行出错: {{e}}")
        logger.error(f"错误堆栈: {{traceback.format_exc()}}")
        raise

if __name__ == "__main__":
    main()
'''
    
    # 写入脚本文件
    os.makedirs(output_dir, exist_ok=True)
    script_name = "conformer_generation"
    script_path = os.path.join(output_dir, f"{script_name}.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    return script_path, script_name, script_content

st.set_page_config(page_title="3D构象生成", layout="wide")
st.title("🧬 3D构象生成")

st.markdown("""
从分子的SMILES字符串或SDF文件生成3D构象。支持CSV、TXT、SMI和SDF格式。
使用RDKit的ETKDGv3算法生成构象。

🚀 **简化的后台执行**: 先生成构象脚本，再通过简单的运行脚本启动后台处理  
⚡ **智能缓存**: 避免重复扫描，提升用户体验  
🔄 **多进程处理**: 支持1-256进程并行生成，充分利用SLURM集群资源  
📊 **任务监控**: 后台任务状态监控，支持日志查看和结果下载  
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
st.subheader("步骤一 · 选择输入数据")
st.markdown("选择或上传一个包含SMILES/SDF的文件，系统会在下方提供快速预览与缓存提示。")

input_method = st.radio(
    "选择输入方式",
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
        help="最大文件大小约4GB。支持百万级分子库的高效处理。",
        key="conformer_upload"
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
                    st.session_state.saved_file_path = file_path
                    st.session_state.saved_work_dir = work_dir_new
                    st.session_state.scan_results_valid = False 
                    st.session_state.last_processed_file_identifier = file_path
                    st.rerun()
            else:
                st.warning("文件已上传但尚未保存到工作目录。点击上方按钮保存文件。")

    elif st.session_state.saved_file_path and os.path.exists(st.session_state.saved_file_path):
        selected_file_path = st.session_state.saved_file_path
        work_dir = st.session_state.saved_work_dir
        current_filename = os.path.basename(selected_file_path)
        st.success(f"使用已保存的文件: {selected_file_path}")
    elif st.session_state.saved_file_path:
        st.warning("之前保存的文件已不存在，请重新上传。")
        st.session_state.saved_file_path = None
        st.session_state.saved_work_dir = None
        st.session_state.scan_results_valid = False

else:  # 使用已保存文件
    st.markdown("**从工作目录中选择文件**")
    
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

input_ready = (uploaded_file is not None) or (selected_file_path is not None and os.path.exists(selected_file_path or ""))

with st.container():
    st.markdown("###### 数据预览与缓存")
    total_mols, scan_successful, is_estimated = render_file_preview_section(
        input_ready,
        input_method,
        uploaded_file,
        selected_file_path,
        current_filename,
        smiles_column
    )

# 构象生成设置
st.subheader("步骤二 · 配置生成参数")
col1, col2, col3 = st.columns(3)
with col1:
    num_conformers = st.number_input("每个分子的构象数:", min_value=1, max_value=100, value=10)
with col2:
    max_attempts = st.number_input("最大嵌入尝试次数:", min_value=1, max_value=500, value=100)
with col3:
    random_seed = st.number_input("随机种子:", min_value=-1, value=42, help="-1表示不指定种子")

# 执行模式设置
st.subheader("步骤三 · 选择执行模式")
st.markdown("根据任务规模选择执行方式。后台模式支持断点恢复并提供脚本工具，前台模式适合小规模快速验证。")

if BACKGROUND_CONFORMER_AVAILABLE:
    execution_mode = st.radio(
        "执行方式",
        ("智能后台执行 (推荐)", "Streamlit内并行执行"),
        index=0,
        help="智能后台执行具有断点恢复功能，不受页面刷新影响；并行执行：在Streamlit内执行，适合小规模测试"
    )
else:
    execution_mode = st.radio(
        "执行方式",
        ("Streamlit内并行执行",),
        index=0,
        help="Streamlit内并行执行：在Streamlit内执行，适合小规模测试"
    )

st.markdown("###### 并行度配置")
if execution_mode == "智能后台执行 (推荐)":
    max_workers_cluster = 256
    default_workers = min(64, multiprocessing.cpu_count() * 2)

    num_workers = st.number_input(
        "工作进程数",
        min_value=1,
        max_value=max_workers_cluster,
        value=default_workers,
        help="用于并行处理分子的进程数（本地推荐≤64，集群模式可扩展至256）"
    )

    if num_workers > 128:
        st.info("💡 **集群建议**: >128 进程通常需要多节点，请在 SLURM 脚本中配置节点信息。")
    elif num_workers > 64:
        st.info("💡 **集群建议**: >64 进程建议在 SLURM 集群上执行以获得最佳性能。")

    st.info("🤖 智能后台模式支持断点恢复，页面刷新不会影响任务进度。")
else:
    num_threads = st.number_input(
        "工作线程数",
        min_value=1,
        max_value=min(128, multiprocessing.cpu_count() * 4),
        value=min(36, multiprocessing.cpu_count() * 2),
        help="用于并行处理分子的线程数（建议不超过128，以免阻塞 UI）"
    )

st.markdown("###### 处理范围")
processing_options = ["处理所有分子", "仅处理前500个分子", "仅处理前10000个分子", "仅处理前100000个分子"]
selected_scope = st.radio("选择处理范围", options=processing_options, index=0, horizontal=True)

limit_map = {
    processing_options[0]: float('inf'),
    processing_options[1]: 500,
    processing_options[2]: 10000,
    processing_options[3]: 100000
}
processing_limit = limit_map[selected_scope]

if execution_mode == "智能后台执行 (推荐)":
    st.markdown("###### 智能后台快速操作")
    if not input_ready or not scan_successful:
        st.info("请在步骤一完成文件选择并成功扫描后，再使用后台快速操作。")
    else:
        col_op1, col_op2, col_op3, col_op4 = st.columns(4)

        with col_op1:
            if st.button("📝 生成构象脚本", type="primary", help="根据当前配置生成独立的Python脚本"):
                if not (selected_file_path or uploaded_file):
                    st.error("❌ 请先选择输入文件")
                else:
                    try:
                        if selected_file_path and os.path.exists(selected_file_path):
                            input_file_path = selected_file_path
                            current_file_name = os.path.basename(selected_file_path)
                            work_directory = os.path.dirname(selected_file_path)
                        elif uploaded_file:
                            if not (st.session_state.saved_file_path and os.path.exists(st.session_state.saved_file_path)):
                                file_path, work_dir_new = save_uploaded_file(uploaded_file)
                                if file_path:
                                    st.session_state.saved_file_path = file_path
                                    st.session_state.saved_work_dir = work_dir_new
                                    input_file_path = file_path
                                    current_file_name = uploaded_file.name
                                    work_directory = work_dir_new
                                else:
                                    st.error("❌ 文件保存失败")
                                    st.stop()
                            else:
                                input_file_path = st.session_state.saved_file_path
                                current_file_name = os.path.basename(st.session_state.saved_file_path)
                                work_directory = st.session_state.saved_work_dir
                        else:
                            st.error("❌ 找不到有效的输入文件")
                            st.stop()

                        file_ext = current_file_name.lower().split('.')[-1]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = f"conformers_{os.path.splitext(current_file_name)[0]}_{timestamp}.sdf"
                        output_path = os.path.join(work_directory, output_filename)

                        script_path, script_name, script_content = generate_conformer_script(
                            input_file_path, output_path, num_conformers, max_attempts,
                            random_seed, num_workers, processing_limit, file_ext, smiles_column
                        )

                        config_filename = f"{script_name}_config.py"
                        config_content = f"""# 构象生成配置文件
# 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

INPUT_FILE = r"{input_file_path}"
OUTPUT_FILE = r"{output_path}"
FILE_TYPE = "{file_ext}"
SMILES_COLUMN = "{smiles_column}"

# 构象参数
NUM_CONFORMERS = {num_conformers}
MAX_ATTEMPTS = {max_attempts}
RANDOM_SEED = {random_seed}
NUM_WORKERS = {num_workers}
PROCESSING_LIMIT = {processing_limit if processing_limit != float('inf') else 'None'}

# 执行模式
EXECUTION_MODE = "{execution_mode}"
PROCESSING_SCOPE = "{selected_scope}"
"""

                        config_path = os.path.join(work_directory, config_filename)
                        with open(config_path, 'w', encoding='utf-8') as f:
                            f.write(config_content)

                        st.success("✅ 脚本生成完成！")
                        st.info(f"📄 主脚本: {script_path}")
                        st.info(f"🏷️ 脚本名称: {script_name}")
                        st.info(f"⚙️ 配置文件: {config_path}")

                        script_size = os.path.getsize(script_path) / 1024
                        config_size = os.path.getsize(config_path) / 1024
                        st.info(f"📊 脚本大小: {script_size:.1f} KB")
                        st.info(f"📊 配置大小: {config_size:.1f} KB")

                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            with open(script_path, 'r', encoding='utf-8') as f:
                                st.download_button(
                                    "📥 下载主脚本",
                                    f.read(),
                                    file_name=os.path.basename(script_path),
                                    mime="text/x-python"
                                )
                        with col_dl2:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                st.download_button(
                                    "📥 下载配置文件",
                                    f.read(),
                                    file_name=os.path.basename(config_path),
                                    mime="text/x-python"
                                )
                    except Exception as e:
                        st.error(f"❌ 脚本生成失败: {e}")
                        import traceback
                        st.code(traceback.format_exc(), language="text")

        with col_op2:
            if st.button("🚀 智能后台运行", type="secondary", help="快速启动智能后台构象生成"):
                if not (selected_file_path or uploaded_file):
                    st.error("❌ 请先选择输入文件")
                else:
                    try:
                        if selected_file_path and os.path.exists(selected_file_path):
                            input_file_path = selected_file_path
                            work_directory = os.path.dirname(selected_file_path)
                            current_file_name = os.path.basename(selected_file_path)
                        elif uploaded_file:
                            if not (st.session_state.saved_file_path and os.path.exists(st.session_state.saved_file_path)):
                                file_path, work_dir_new = save_uploaded_file(uploaded_file)
                                if not file_path:
                                    st.error("❌ 文件保存失败")
                                    st.stop()
                                st.session_state.saved_file_path = file_path
                                st.session_state.saved_work_dir = work_dir_new
                            input_file_path = st.session_state.saved_file_path
                            work_directory = st.session_state.saved_work_dir
                            current_file_name = os.path.basename(st.session_state.saved_file_path)
                        else:
                            st.error("❌ 找不到有效的输入文件")
                            st.stop()

                        file_ext = current_file_name.lower().split('.')[-1]

                        script_file, script_name = run_background_conformer_generation(
                            input_file=input_file_path,
                            output_dir=work_directory,
                            num_conformers=num_conformers,
                            max_attempts=max_attempts,
                            random_seed=random_seed,
                            num_workers=num_workers,
                            processing_limit=processing_limit,
                            file_type=file_ext,
                            smiles_column=smiles_column,
                        )

                        st.success("✅ 智能后台构象生成已启动！")
                        st.info(f"📄 运行脚本: {script_file}")
                        st.info(f"🏷️ 任务名称: {script_name}")

                        log_file = os.path.join(work_directory, f"{script_name}_output.log")
                        progress_file = os.path.join(work_directory, f"{script_name}.progress")
                        st.info(f"📝 日志文件: {log_file}")
                        st.info(f"📈 进度文件: {progress_file}")

                        task_info = {
                            'script_name': script_name,
                            'script_file': script_file,
                            'work_dir': work_directory,
                            'start_time': time.time(),
                            'input_file': input_file_path,
                            'log_file': log_file,
                            'progress_file': progress_file,
                            'config': {
                                'num_workers': num_workers,
                                'num_conformers': num_conformers,
                                'max_attempts': max_attempts,
                                'processing_limit': selected_scope,
                                'file_type': current_file_name.lower().split('.')[-1],
                                'smiles_column': smiles_column
                            }
                        }
                        st.session_state.simple_background_tasks.append(task_info)

                        st.success("🎉 任务已添加到后台任务列表！")
                        st.info("💡 查看下方“后台任务监控中心”了解进度")
                    except Exception as e:
                        st.error(f"❌ 智能后台启动失败: {e}")
                        import traceback
                        st.code(traceback.format_exc(), language="text")

        with col_op3:
            if st.button("🔍 预处理检查", help="检查输入文件并生成预处理报告"):
                if not (selected_file_path or uploaded_file):
                    st.error("❌ 请先选择输入文件")
                else:
                    try:
                        if selected_file_path and os.path.exists(selected_file_path):
                            input_file_path = selected_file_path
                            current_file_name = os.path.basename(selected_file_path)
                            work_directory = os.path.dirname(selected_file_path)
                        elif uploaded_file:
                            if not (st.session_state.saved_file_path and os.path.exists(st.session_state.saved_file_path)):
                                file_path, work_dir_new = save_uploaded_file(uploaded_file)
                                if file_path:
                                    st.session_state.saved_file_path = file_path
                                    st.session_state.saved_work_dir = work_dir_new
                                    input_file_path = file_path
                                    current_file_name = uploaded_file.name
                                    work_directory = work_dir_new
                                else:
                                    st.error("❌ 文件保存失败")
                                    st.stop()
                            else:
                                input_file_path = st.session_state.saved_file_path
                                current_file_name = os.path.basename(st.session_state.saved_file_path)
                                work_directory = st.session_state.saved_work_dir
                        else:
                            st.error("❌ 找不到有效的输入文件")
                            st.stop()

                        file_ext = current_file_name.lower().split('.')[-1]

                        preprocess_script = f'''#!/usr/bin/env python3
"""
分子文件预处理检查脚本
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import os
import sys
import pandas as pd
from rdkit import Chem
from datetime import datetime

def preprocess_check(input_file, file_type, smiles_column="SMILES"):
    """预处理检查函数"""
    print(f"开始预处理检查: {{input_file}}")
    print(f"文件类型: {{file_type}}")
    print(f"检查时间: {{datetime.now()}}")
    print("-" * 50)
    
    # 文件基本信息
    file_size = os.path.getsize(input_file) / (1024 * 1024)
    print(f"文件大小: {{file_size:.2f}} MB")
    
    valid_count = 0
    invalid_count = 0
    total_count = 0
    invalid_examples = []
    
    if file_type == "csv":
        df = pd.read_csv(input_file)
        print(f"CSV行数: {{len(df)}}")
        print(f"CSV列名: {{list(df.columns)}}")
        
        if smiles_column not in df.columns:
            print(f"❌ 错误: 未找到SMILES列 '{{smiles_column}}'")
            return
        
        print(f"开始检查SMILES列: {{smiles_column}}")
        for idx, row in df.iterrows():
            total_count += 1
            smi = str(row[smiles_column])
            mol = Chem.MolFromSmiles(smi)
            if mol:
                valid_count += 1
            else:
                invalid_count += 1
                if len(invalid_examples) < 10:
                    invalid_examples.append(f"行{{idx+1}}: {{smi}}")
            
            if total_count % 10000 == 0:
                print(f"已检查: {{total_count}}, 有效: {{valid_count}}, 无效: {{invalid_count}}")
    
    elif file_type in ["txt", "smi"]:
        print(f"开始检查文本文件...")
        with open(input_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                total_count += 1
                smi = line.strip()
                if smi:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        valid_count += 1
                    else:
                        invalid_count += 1
                        if len(invalid_examples) < 10:
                            invalid_examples.append(f"行{{i+1}}: {{smi}}")
                
                if total_count % 10000 == 0:
                    print(f"已检查: {{total_count}}, 有效: {{valid_count}}, 无效: {{invalid_count}}")
    
    elif file_type == "sdf":
        print(f"开始检查SDF文件...")
        supplier = Chem.ForwardSDMolSupplier(input_file, removeHs=False, sanitize=True)
        for i, mol in enumerate(supplier):
            total_count += 1
            if mol is not None:
                valid_count += 1
            else:
                invalid_count += 1
                if len(invalid_examples) < 10:
                    invalid_examples.append(f"SDF分子{{i+1}}: 解析失败")
            
            if total_count % 1000 == 0:
                print(f"已检查: {{total_count}}, 有效: {{valid_count}}, 无效: {{invalid_count}}")
    
    # 输出结果
    print("=" * 50)
    print(f"预处理检查完成!")
    print(f"总计: {{total_count}}")
    print(f"有效: {{valid_count}} ({{valid_count/total_count*100:.2f}}%)")
    print(f"无效: {{invalid_count}} ({{invalid_count/total_count*100:.2f}}%)")
    
    if invalid_examples:
        print("\\n无效分子示例:")
        for example in invalid_examples:
            print(f"  {{example}}")
        if invalid_count > 10:
            print(f"  ... 还有{{invalid_count-10}}个无效分子")
    
    # 生成建议
    print("\\n处理建议:")
    if invalid_count == 0:
        print("✅ 所有分子都有效，可以直接进行构象生成")
    elif invalid_count / total_count < 0.05:
        print("⚠️ 少量无效分子，建议继续处理并忽略无效分子")
    else:
        print("❌ 无效分子较多，建议先清理数据")
    
    print(f"\\n建议构象数: {{min(50, max(10, valid_count//1000))}}")
    print(f"建议进程数: {{min(34, max(4, valid_count//10000))}}")

if __name__ == "__main__":
    preprocess_check(r"{input_file_path}", "{file_ext}", "{smiles_column}")
'''

                        preprocess_path = os.path.join(work_directory, "preprocess_check.py")
                        with open(preprocess_path, 'w', encoding='utf-8') as f:
                            f.write(preprocess_script)

                        st.success(f"✅ 预处理脚本已生成: {preprocess_path}")

                        st.download_button(
                            "📥 下载预处理脚本",
                            preprocess_script,
                            file_name="preprocess_check.py",
                            mime="text/x-python"
                        )

                        st.info("🔄 正在运行预处理检查...")
                        try:
                            result = subprocess.run(
                                ["python", preprocess_path],
                                capture_output=True,
                                text=True,
                                timeout=60
                            )

                            if result.returncode == 0:
                                st.subheader("📊 预处理检查结果")
                                st.code(result.stdout, language="text")
                            else:
                                st.error("预处理检查出错:")
                                st.code(result.stderr, language="text")

                        except subprocess.TimeoutExpired:
                            st.warning("⏰ 预处理检查超时，请手动运行脚本")
                        except Exception as e:
                            st.error(f"运行预处理检查失败: {e}")
                    except Exception as e:
                        st.error(f"❌ 预处理脚本生成失败: {e}")
                        import traceback
                        st.code(traceback.format_exc(), language="text")

        with col_op4:
            if st.button("🖥️ 生成SLURM脚本", help="生成用于SLURM集群的任务提交脚本"):
                if not (selected_file_path or uploaded_file):
                    st.error("❌ 请先选择输入文件")
                else:
                    try:
                        if selected_file_path and os.path.exists(selected_file_path):
                            input_file_path = selected_file_path
                            current_file_name = os.path.basename(selected_file_path)
                            work_directory = os.path.dirname(selected_file_path)
                        elif uploaded_file:
                            if not (st.session_state.saved_file_path and os.path.exists(st.session_state.saved_file_path)):
                                file_path, work_dir_new = save_uploaded_file(uploaded_file)
                                if file_path:
                                    st.session_state.saved_file_path = file_path
                                    st.session_state.saved_work_dir = work_dir_new
                                    input_file_path = file_path
                                    current_file_name = uploaded_file.name
                                    work_directory = work_dir_new
                                else:
                                    st.error("❌ 文件保存失败")
                                    st.stop()
                            else:
                                input_file_path = st.session_state.saved_file_path
                                current_file_name = os.path.basename(st.session_state.saved_file_path)
                                work_directory = st.session_state.saved_work_dir
                        else:
                            st.error("❌ 找不到有效的输入文件")
                            st.stop()

                        estimated_molecules = total_mols if (total_mols and total_mols > 0) else 100000
                        total_conformers = estimated_molecules * num_conformers
                        estimated_time_hours = max(1, total_conformers / (num_workers * 5000))
                        estimated_memory = max(16, num_workers * 2)

                        slurm_script = f'''#!/bin/bash
#SBATCH --job-name=conformer_{datetime.now().strftime('%m%d')}
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node={num_workers}
#SBATCH --time={int(estimated_time_hours)+1}:00:00
#SBATCH --mem={estimated_memory}G
#SBATCH --output=conformer_%j.out
#SBATCH --error=conformer_%j.err

module purge
# module load your_env_module
# source activate CADD-Toolbox

cd {work_directory}

if [ ! -f "conformer_generation.py" ]; then
    echo "❌ 错误: 找不到 conformer_generation.py"
    exit 1
fi

python conformer_generation.py
'''

                        slurm_path = os.path.join(work_directory, "conformer_slurm.sh")
                        with open(slurm_path, 'w', encoding='utf-8') as f:
                            f.write(slurm_script)

                        st.success(f"✅ SLURM脚本已生成: {slurm_path}")
                        st.download_button(
                            "📥 下载SLURM脚本",
                            slurm_script,
                            file_name="conformer_slurm.sh",
                            mime="text/x-shellscript"
                        )
                    except Exception as e:
                        st.error(f"❌ SLURM脚本生成失败: {e}")
                        import traceback
                        st.code(traceback.format_exc(), language="text")
else:
    st.markdown("###### Streamlit执行说明")
    if not input_ready or not scan_successful:
        st.info("请先完成步骤一的数据扫描，确认文件可用后方可进行前台执行。")
    else:
        st.success("✅ 数据已准备，可在步骤四直接执行并实时查看进度。")

# 文件处理和预览
input_ready = (uploaded_file is not None) or (selected_file_path is not None and os.path.exists(selected_file_path))

# 执行任务
st.subheader("步骤四 · 执行任务")

if not input_ready:
    st.info("请先完成步骤一的文件选择。")
elif not scan_successful:
    st.info("请确认上方数据预览已成功完成。")
elif execution_mode == "智能后台执行 (推荐)":
    st.info("💡 后台执行相关按钮已在步骤三的“智能后台快速操作”中提供。")
else:
    if total_mols in (None, 0):
        st.warning("未检测到可用分子，请检查输入文件。")
    else:
        if total_mols == -1:
            if processing_limit == float('inf'):
                mol_count_label = "CSV中的所有分子"
            else:
                mol_count_label = f"前{int(processing_limit)}个"
        else:
            if processing_limit == float('inf'):
                mol_count_label = f"{total_mols:,}"
            else:
                actual_count = min(total_mols, int(processing_limit))
                mol_count_label = f"{actual_count:,}"

        if processing_limit != float('inf') and total_mols not in (None, -1) and total_mols > processing_limit:
            st.info(f"范围选择: 将处理 {total_mols:,} 个条目中的前{int(processing_limit):,}个")

        if st.session_state.file_size_cache and st.session_state.file_size_cache > LARGE_FILE_THRESHOLD:
            st.warning("⚡ **大文件建议**: 对于百万级分子库，建议选择较小的处理范围以获得更好的性能")

        button_label = f"为 {mol_count_label} 个分子生成 {num_conformers} 个构象 (前台: {num_threads} 线程)"

        if st.button(button_label):
            st.info("🔄 在Streamlit内进行并行构象生成")

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
                                        if not hasattr(st.session_state, 'parse_status_placeholder'):
                                            st.session_state.parse_status_placeholder = st.empty()
                                        st.session_state.parse_status_placeholder.info(
                                            f"🔄 解析进度: {total_processed:,} 行，{len(input_mols):,} 个有效分子")
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
                                st.error(f"SMILES列 '{smiles_column}' 不存在，请重新检查。")
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
                                    if st.session_state.file_size_cache > LARGE_FILE_THRESHOLD and i % 10000 == 0 and i > 0:
                                        if not hasattr(st.session_state, 'parse_status_placeholder'):
                                            st.session_state.parse_status_placeholder = st.empty()
                                        st.session_state.parse_status_placeholder.info(
                                            f"🔄 解析进度: {i:,} 行，{len(input_mols):,} 个有效分子")
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

                    if not input_mols:
                        st.warning("基于当前范围，没有解析到有效分子用于生成。")
                        st.stop()

                except Exception as e:
                    st.error(f"完整解析 '{current_filename}' 时出错: {e}")
                    st.stop()

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
                future_to_idx = {executor.submit(generate_conformers_for_mol, *task): i for i, task in enumerate(tasks)}

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

                    if (current_time - last_update_time > 0.5) or (completed % 100 == 0) or (completed == len(input_mols)):
                        elapsed_time = current_time - start_time
                        progress_bar.progress(progress)

                        completed_metric.metric("已完成", f"{completed:,}/{len(input_mols):,}")
                        success_metric.metric("成功率", f"{success_count/completed*100:.1f}%" if completed > 0 else "0%")

                        if completed > 10:
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
                    st.markdown("\n".join([f"- {msg}" for msg in errors[:100]]))
                    if len(errors) > 100:
                        st.warning(f"还有 {len(errors)-100} 个错误未显示...")

            if processed_mols:
                sdf_output = mols_to_sdf_string(processed_mols)

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

                output_size_mb = len(sdf_output.encode('utf-8')) / (1024 * 1024)
                st.info(f"生成的SDF文件大小: {output_size_mb:.1f} MB")

                st.subheader("生成的SDF预览 (前1000字符)")
                st.code(sdf_output[:1000], language="text")
            else:
                st.warning("没有成功生成构象来创建SDF文件。")

# 简化的后台任务监控区域
st.header("🤖 后台任务监控")

# 添加自动刷新按钮
col_refresh, col_clear = st.columns([3, 1])
with col_refresh:
    if st.button("🔄 刷新状态", help="刷新所有后台任务的状态"):
        st.rerun()
with col_clear:
    if st.button("🗑️ 清理任务", help="清理已完成的任务"):
        # 只保留正在运行的任务
        running_tasks = []
        for task in st.session_state.simple_background_tasks:
            status_info = simple_check_background_status(task['work_dir'], task['script_name'])
            if status_info['status'] == 'running':
                running_tasks.append(task)
        st.session_state.simple_background_tasks = running_tasks
        st.success(f"已清理完成的任务，保留 {len(running_tasks)} 个运行中的任务")
        st.rerun()

if 'simple_background_tasks' in st.session_state and st.session_state.simple_background_tasks:
    # 检查是否有运行中的任务
    has_running_tasks = False
    for task in st.session_state.simple_background_tasks:
        status_info = simple_check_background_status(task['work_dir'], task['script_name'])
        if status_info['status'] == 'running':
            has_running_tasks = True
            break
    
    # 如果有运行中的任务，添加自动刷新提示
    if has_running_tasks:
        st.info("💡 检测到运行中的任务，请定期点击'🔄 刷新状态'按钮查看最新进度")
    
    for idx, task in enumerate(st.session_state.simple_background_tasks):
        with st.expander(f"📋 任务 {idx+1}: {task['script_name']}", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                elapsed = time.time() - task['start_time']
                st.metric("运行时间", f"{elapsed/60:.1f} 分钟")
            
            with col2:
                # 检查任务状态
                status_info = simple_check_background_status(task['work_dir'], task['script_name'])
                status_display = {
                    'running': "🟢 运行中",
                    'completed': "✅ 已完成",
                    'error': "❌ 出错",
                    'unknown': "❓ 未知"
                }
                st.metric("状态", status_display.get(status_info['status'], "❓ 未知"))
            
            with col3:
                config = task['config']
                st.metric("配置", f"{config['num_workers']}进程/{config['num_conformers']}构象")
            
            # 详细信息
            st.code(f"""
任务名称: {task['script_name']}
输入文件: {task['input_file']}
工作目录: {task['work_dir']}
运行脚本: {task['script_file']}
""", language="text")
            
            # 操作按钮
            col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
            
            with col_btn1:
                if st.button(f"📖 查看日志 {idx+1}", key=f"log_{idx}"):
                    log_file = os.path.join(task['work_dir'], f"{task['script_name']}.log")
                    if os.path.exists(log_file):
                        try:
                            with open(log_file, 'r', encoding='utf-8') as f:
                                log_content = f.read()
                            
                            # 只显示最后100行
                            log_lines = log_content.split('\n')
                            recent_logs = '\n'.join(log_lines[-100:])
                            st.code(recent_logs, language="text")
                            
                        except Exception as e:
                            st.error(f"读取日志失败: {e}")
                    else:
                        st.warning("日志文件不存在")
            
            with col_btn2:
                if st.button(f"📊 检查结果 {idx+1}", key=f"result_{idx}"):
                    if status_info['status'] == 'completed':
                        st.success("🎉 任务已完成！")
                        st.code(status_info.get('details', ''), language="text")
                        
                        # 查找输出文件
                        output_files = []
                        for file in os.listdir(task['work_dir']):
                            if (file.startswith('conformers_') or file.startswith('generated_conformers_')) and file.endswith('.sdf'):
                                output_files.append(file)
                        
                        if output_files:
                            for output_file in output_files:
                                full_path = os.path.join(task['work_dir'], output_file)
                                file_size = os.path.getsize(full_path) / (1024*1024)
                                st.info(f"📄 输出文件: {output_file} ({file_size:.1f} MB)")
                                
                                # 提供下载按钮
                                with open(full_path, 'rb') as f:
                                    st.download_button(
                                        f"📥 下载 {output_file}",
                                        f.read(),
                                        file_name=output_file,
                                        mime="chemical/x-mdl-sdfile",
                                        key=f"download_{idx}_{output_file}"
                                    )
                    elif status_info['status'] == 'error':
                        st.error("❌ 任务执行出错")
                        st.code(status_info.get('details', ''), language="text")
                    else:
                        st.info("⏳ 任务仍在运行中...")
            
            with col_btn3:
                if st.button(f"🗑️ 移除任务 {idx+1}", key=f"remove_{idx}"):
                    st.session_state.simple_background_tasks.pop(idx)
                    st.rerun()
            
            with col_btn4:
                if st.button(f"🔍 详细信息 {idx+1}", key=f"detail_{idx}"):
                    st.json(task)
else:
    st.info("🔍 当前没有运行中的后台任务")



# 分隔线
st.divider()

# 使用说明
with st.expander("📖 使用说明", expanded=False):
    st.markdown("""
    ### 功能说明
    
    1. **文件选择**: 上传新文件或选择已保存的SMILES/SDF文件
    2. **构象设置**: 配置每个分子的构象数量和生成参数
    3. **执行模式**: 选择后台多进程执行或前台并行执行
    4. **处理范围**: 选择处理全部或部分分子
    5. **构象生成**: 使用RDKit的ETKDGv3算法生成3D构象
    6. **结果输出**: 生成包含所有构象的SDF文件
    
    ### 执行模式说明
    
    - **智能后台执行 (推荐)**: 先生成 conformer_generation.py 脚本，再通过简单的运行脚本启动后台处理，支持断点恢复，不受页面刷新影响
    - **Streamlit内并行执行**: 在Streamlit内使用线程池执行，适合小规模测试
    
    ### 处理范围说明
    
    - **处理所有分子**: 生产模式，无数量限制，支持百万级化合物库
    - **仅处理前N个分子**: 快速验证功能和参数设置
    
    ### 构象生成参数
    
    - **构象数**: 每个分子生成的3D构象数量（推荐10-50）
    - **最大尝试次数**: RDKit嵌入算法的最大迭代次数
    - **随机种子**: 控制结果可重现性（-1表示随机）
    
    ### 后台任务监控
    
    - **状态监控**: 查看进程状态、运行时间
    - **日志查看**: 实时查看生成日志，了解详细进度
    - **完成检查**: 自动检测任务完成，提供结果下载
    - **任务管理**: 支持清除和管理后台任务
    
    ### 注意事项
    
    - **工作流程**: 先点击"生成构象脚本"，再点击"智能后台运行"启动后台执行
    - **大规模处理**: 百万级化合物库建议使用智能后台执行模式
    - **内存使用**: 大文件可能需要较多内存，监控系统资源
    - **测试建议**: 首次使用建议先用小范围测试验证参数
    - **页面独立**: 智能后台模式支持页面刷新不影响进度
    - **处理时间**: 百万级数据可能需要数小时，可通过后台监控区域查看进度
    """)  
