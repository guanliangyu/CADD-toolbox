# -*- coding: utf-8 -*-
"""
CADD-Toolbox - 多样性筛选页面 (后台执行版)
基于脚本生成和后台执行的多样性筛选功能
"""

import os
import time
import subprocess
from datetime import datetime

import streamlit as st

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

st.set_page_config(page_title="多样性筛选_后台版", layout="wide")
st.title("🚀 多样性筛选 (后台执行版)")

st.markdown(
    """
基于脚本生成和后台执行的多样性筛选功能。

🔧 **功能特点**: 脚本生成、后台执行、进程监控  
📏 **距离度量**: 欧氏距离、曼哈顿距离、余弦距离  
⚡ **执行方式**: 生成独立Python脚本并后台运行  
🎯 **优势**: 不受页面刷新影响，可长时间运行  
"""
)

# 数据目录设置
DATA_DIR = "data"


def list_data_folders():
    """列出data目录下的所有文件夹"""
    if not os.path.exists(DATA_DIR):
        return []
    return [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]


def list_csv_files_in_folder(folder_name):
    """列出指定文件夹中的所有CSV文件"""
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.endswith(".csv")]


def get_file_info(file_path):
    """获取文件基本信息"""
    if not os.path.exists(file_path):
        return None

    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    mod_time = os.path.getmtime(file_path)
    mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")

    return {"size_mb": file_size, "modified": mod_time_str}


def generate_python_script(config):
    """生成多样性筛选的Python脚本"""
    script_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动生成的多样性筛选脚本
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

def log_message(msg):
    """带时间戳的日志输出"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

@torch.no_grad()
def greedy_cosine_gpu_optimized(data_norm, k, seed=42, initial_method='random'):
    """优化的余弦距离贪心算法"""
    torch.manual_seed(seed)
    N = data_norm.size(0)
    device = data_norm.device
    
    selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
    min_dist = torch.full((N,), 2.0, dtype=data_norm.dtype, device=device)
    
    # 选择初始点
    if initial_method == 'random':
        idx = torch.randint(0, N, (1,), device=device).item()
    elif initial_method == 'centroid':
        centroid = torch.mean(data_norm, dim=0, keepdim=True)
        centroid = torch.nn.functional.normalize(centroid, p=2.0, dim=1)
        similarities = torch.matmul(data_norm, centroid.T).squeeze()
        distances = 1 - similarities
        idx = torch.argmax(distances).item()
    else:
        idx = 0
    
    selected_mask[idx] = True
    last_vec = data_norm[idx]
    
    log_message(f"初始分子索引: {idx}")
    
    # 迭代选择
    for i in range(1, k):
        sim = torch.matmul(data_norm, last_vec)
        dist = 1 - sim
        min_dist = torch.minimum(min_dist, dist)
        min_dist[selected_mask] = -1.0
        
        idx = torch.argmax(min_dist).item()
        selected_mask[idx] = True
        last_vec = data_norm[idx]
        
        if (i + 1) % 100 == 0:
            log_message(f"已选择 {i + 1}/{k} 个分子")
    
    # 提取索引
    selected_indices = selected_mask.nonzero(as_tuple=False).squeeze().cpu().tolist()
    if isinstance(selected_indices, int):
        selected_indices = [selected_indices]
    
    return selected_indices

@torch.no_grad()
def greedy_euclidean_gpu_optimized(data_tensor, k, seed=42, initial_method='random'):
    """优化的欧氏距离贪心算法"""
    torch.manual_seed(seed)
    N = data_tensor.size(0)
    device = data_tensor.device
    
    selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
    min_dist = torch.full((N,), float('inf'), dtype=data_tensor.dtype, device=device)
    
    # 选择初始点
    if initial_method == 'random':
        idx = torch.randint(0, N, (1,), device=device).item()
    elif initial_method == 'centroid':
        centroid = torch.mean(data_tensor, dim=0, keepdim=True)
        sq_distances = torch.sum((data_tensor - centroid)**2, dim=1)
        idx = torch.argmax(sq_distances).item()
    else:
        idx = 0
    
    selected_mask[idx] = True
    last_point = data_tensor[idx]
    
    log_message(f"初始分子索引: {idx}")
    
    # 迭代选择
    for i in range(1, k):
        diff = data_tensor - last_point
        sq_dist = torch.sum(diff**2, dim=1)
        min_dist = torch.minimum(min_dist, sq_dist)
        min_dist[selected_mask] = -1.0
        
        idx = torch.argmax(min_dist).item()
        selected_mask[idx] = True
        last_point = data_tensor[idx]
        
        if (i + 1) % 100 == 0:
            log_message(f"已选择 {i + 1}/{k} 个分子")
    
    # 提取索引
    selected_indices = selected_mask.nonzero(as_tuple=False).squeeze().cpu().tolist()
    if isinstance(selected_indices, int):
        selected_indices = [selected_indices]
    
    return selected_indices

def greedy_selection_cpu(data_np, subset_size, distance_method='euclidean', initial_method='random', seed=42):
    """CPU版本贪心算法"""
    np.random.seed(seed)
    N = len(data_np)
    
    # 选择初始点
    if initial_method == 'random':
        initial_idx = np.random.randint(0, N)
    elif initial_method == 'centroid':
        centroid = np.mean(data_np, axis=0)
        distances = np.sum((data_np - centroid)**2, axis=1)
        initial_idx = np.argmax(distances)
    else:
        initial_idx = 0
    
    selected_indices = [initial_idx]
    log_message(f"初始分子索引: {initial_idx}")
    
    # 贪心迭代选择
    for i in range(1, subset_size):
        min_distances = np.full(N, np.inf)
        
        for selected_idx in selected_indices:
            if distance_method == 'euclidean':
                distances = np.sum((data_np - data_np[selected_idx])**2, axis=1)
            elif distance_method == 'manhattan':
                distances = np.sum(np.abs(data_np - data_np[selected_idx]), axis=1)
            elif distance_method == 'cosine':
                data_norm = data_np / (np.linalg.norm(data_np, axis=1, keepdims=True) + 1e-8)
                selected_norm = data_norm[selected_idx]
                similarities = np.dot(data_norm, selected_norm)
                distances = 1 - similarities
            
            min_distances = np.minimum(min_distances, distances)
        
        # 已选点距离设为0
        min_distances[selected_indices] = 0.0
        
        # 选择距离最远的点
        farthest_idx = np.argmax(min_distances)
        selected_indices.append(farthest_idx)
        
        if (i + 1) % 100 == 0:
            log_message(f"已选择 {i + 1}/{subset_size} 个分子")
    
    return selected_indices

def main():
    parser = argparse.ArgumentParser(description='多样性筛选脚本')
    parser.add_argument('--input_file', required=True, help='输入CSV文件路径')
    parser.add_argument('--output_file', required=True, help='输出CSV文件路径')
    parser.add_argument('--subset_size', type=int, required=True, help='筛选后的子集大小')
    parser.add_argument('--distance_method', choices=['euclidean', 'manhattan', 'cosine'], 
                       default='cosine', help='距离计算方法')
    parser.add_argument('--initial_method', choices=['random', 'centroid', 'first'], 
                       default='random', help='初始点选择方法')
    parser.add_argument('--use_gpu', action='store_true', help='使用GPU加速')
    parser.add_argument('--use_half_precision', action='store_true', help='使用半精度浮点数')
    parser.add_argument('--random_seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    log_message("多样性筛选脚本开始执行")
    log_message(f"输入文件: {args.input_file}")
    log_message(f"输出文件: {args.output_file}")
    log_message(f"子集大小: {args.subset_size}")
    log_message(f"距离方法: {args.distance_method}")
    
    start_time = time.time()
    
    try:
        # 读取数据
        log_message("读取输入数据...")
        
        # 快速列类型检测
        sample_df = pd.read_csv(args.input_file, nrows=100)
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = sample_df.select_dtypes(exclude=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            raise ValueError("未找到数值类型的描述符列")
        
        # 读取数值列
        float32_dtypes = {col: 'float32' for col in numeric_cols}
        df_numeric = pd.read_csv(args.input_file, dtype=float32_dtypes, usecols=numeric_cols, 
                               engine='c', low_memory=True)
        
        # 读取非数值列
        if non_numeric_cols:
            df_meta = pd.read_csv(args.input_file, usecols=non_numeric_cols, engine='c')
        else:
            df_meta = pd.DataFrame(index=range(len(df_numeric)))
        
        log_message(f"数据形状: {df_numeric.shape} ({len(numeric_cols)} 个描述符列)")
        
        # 处理缺失值
        nan_count = df_numeric.isna().sum().sum()
        if nan_count > 0:
            log_message(f"发现 {nan_count} 个NaN值，填充为0")
            df_numeric = df_numeric.fillna(0.0)
        
        # 处理无穷大值
        inf_count = np.isinf(df_numeric.values).sum()
        if inf_count > 0:
            log_message(f"发现 {inf_count} 个无穷大值，替换为0")
            df_numeric = df_numeric.replace([np.inf, -np.inf], 0.0)
        
        # 转换为numpy数组
        descriptor_data = df_numeric.values.astype(np.float32)
        del df_numeric
        
        total_molecules = len(descriptor_data)
        final_subset_size = min(args.subset_size, total_molecules)
        
        log_message(f"将从 {total_molecules} 个分子中选择 {final_subset_size} 个代表性分子")
        
        # 执行筛选
        if args.use_gpu and TORCH_AVAILABLE and CUDA_AVAILABLE:
            log_message("使用GPU进行筛选...")
            device = torch.device('cuda')
            
            # 数据传输到GPU
            compute_dtype = torch.float16 if args.use_half_precision else torch.float32
            if args.use_half_precision:
                descriptor_data = descriptor_data.astype(np.float16)
            
            data_tensor = torch.from_numpy(descriptor_data).pin_memory().to(device, non_blocking=True)
            
            # 余弦距离需要预先归一化
            if args.distance_method == 'cosine':
                log_message("预归一化处理 (余弦距离)...")
                norms = data_tensor.pow(2).sum(1, keepdim=True)
                data_tensor = data_tensor * torch.rsqrt(norms + 1e-6)
            
            # 执行筛选算法
            if args.distance_method == 'cosine':
                selected_indices = greedy_cosine_gpu_optimized(
                    data_norm=data_tensor,
                    k=final_subset_size,
                    seed=args.random_seed,
                    initial_method=args.initial_method
                )
            elif args.distance_method == 'euclidean':
                selected_indices = greedy_euclidean_gpu_optimized(
                    data_tensor=data_tensor,
                    k=final_subset_size,
                    seed=args.random_seed,
                    initial_method=args.initial_method
                )
            else:
                # 曼哈顿距离暂时使用CPU
                log_message("曼哈顿距离暂不支持GPU，切换到CPU模式")
                data_np = data_tensor.cpu().numpy()
                selected_indices = greedy_selection_cpu(
                    data_np, final_subset_size, args.distance_method, 
                    args.initial_method, args.random_seed
                )
            
            # 清理GPU内存
            del data_tensor
            torch.cuda.empty_cache()
            
        else:
            log_message("使用CPU进行筛选...")
            # 余弦距离归一化
            if args.distance_method == 'cosine':
                norms = np.linalg.norm(descriptor_data, axis=1, keepdims=True)
                descriptor_data = descriptor_data / (norms + 1e-8)
            
            selected_indices = greedy_selection_cpu(
                descriptor_data, final_subset_size, args.distance_method, 
                args.initial_method, args.random_seed
            )
        
        # 生成结果
        log_message("生成筛选结果...")
        diverse_subset_df = df_meta.iloc[selected_indices].copy()
        
        # 保存结果
        diverse_subset_df.to_csv(args.output_file, index=False)
        
        elapsed_time = time.time() - start_time
        output_size = os.path.getsize(args.output_file) / (1024 * 1024)
        
        log_message(f"筛选完成！用时 {elapsed_time:.1f} 秒")
        log_message(f"结果已保存到: {args.output_file}")
        log_message(f"输出文件大小: {output_size:.2f} MB")
        log_message("脚本执行成功！")
        
    except Exception as e:
        log_message(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    return script_content


def generate_shell_script(python_script_path, config):
    """生成执行Python脚本的Shell脚本"""
    shell_content = f"""#!/bin/bash

# 多样性筛选执行脚本
# 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"

# Python脚本路径
PYTHON_SCRIPT="{python_script_path}"

# 参数设置
INPUT_FILE="{config['input_file']}"
OUTPUT_FILE="{config['output_file']}"
SUBSET_SIZE={config['subset_size']}
DISTANCE_METHOD="{config['distance_method']}"
INITIAL_METHOD="{config['initial_method']}"
RANDOM_SEED={config['random_seed']}

# GPU设置
GPU_FLAG=""
{('GPU_FLAG="--use_gpu"' if config['use_gpu'] else '')}
{('GPU_FLAG="$GPU_FLAG --use_half_precision"' if config.get('use_half_precision', False) else '')}

# 日志文件
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
STDOUT_LOG="$LOG_DIR/{config['job_name']}_stdout.log"
STDERR_LOG="$LOG_DIR/{config['job_name']}_stderr.log"

echo "开始执行多样性筛选 - $(date)" | tee -a "$STDOUT_LOG"
echo "输入文件: $INPUT_FILE" | tee -a "$STDOUT_LOG"
echo "输出文件: $OUTPUT_FILE" | tee -a "$STDOUT_LOG"
echo "子集大小: $SUBSET_SIZE" | tee -a "$STDOUT_LOG"

# 执行Python脚本
python3 "$PYTHON_SCRIPT" \\
    --input_file "$INPUT_FILE" \\
    --output_file "$OUTPUT_FILE" \\
    --subset_size $SUBSET_SIZE \\
    --distance_method "$DISTANCE_METHOD" \\
    --initial_method "$INITIAL_METHOD" \\
    --random_seed $RANDOM_SEED \\
    $GPU_FLAG \\
    2> >(tee -a "$STDERR_LOG" >&2) | tee -a "$STDOUT_LOG"

EXIT_CODE=${{PIPESTATUS[0]}}

if [ $EXIT_CODE -eq 0 ]; then
    echo "筛选成功完成 - $(date)" | tee -a "$STDOUT_LOG"
else
    echo "筛选失败，退出代码: $EXIT_CODE - $(date)" | tee -a "$STDERR_LOG"
fi

exit $EXIT_CODE
"""
    return shell_content


# 文件选择界面
st.subheader("1. 选择输入文件")

folders = list_data_folders()
if not folders:
    st.warning("data目录下没有找到任何文件夹")
    st.stop()

selected_folder = st.selectbox("选择数据文件夹:", folders)

if selected_folder:
    csv_files = list_csv_files_in_folder(selected_folder)

    if not csv_files:
        st.warning(f"文件夹 {selected_folder} 中没有CSV文件")
        st.stop()

    selected_file = st.selectbox("选择描述符CSV文件:", csv_files)

    if selected_file:
        file_path = os.path.join(DATA_DIR, selected_folder, selected_file)

        # 显示文件信息
        file_info = get_file_info(file_path)
        if file_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("文件大小", f"{file_info['size_mb']:.1f} MB")
            with col2:
                st.metric("修改时间", file_info["modified"])
            with col3:
                # 使用session state缓存分子数统计
                cache_key = f"molecule_count_{file_path}_{file_info['modified']}"

                if cache_key not in st.session_state:
                    with st.spinner("统计分子数..."):
                        try:
                            with open(file_path, "r") as f:
                                row_count = sum(1 for line in f) - 1  # 减去表头
                            st.session_state[cache_key] = row_count
                        except Exception:
                            st.session_state[cache_key] = "读取失败"

                row_count = st.session_state[cache_key]
                st.metric("分子数量", row_count)

        # 筛选设置
        st.subheader("2. 筛选设置")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🚀 算法设置**")

            # 距离算法选择
            distance_method = st.selectbox(
                "距离算法:",
                ["euclidean", "manhattan", "cosine"],
                index=2,  # 默认选择余弦距离
                format_func=lambda x: {
                    "euclidean": "欧氏距离 (L2)",
                    "manhattan": "曼哈顿距离 (L1)",
                    "cosine": "余弦距离 (推荐)",
                }[x],
            )

            # 初始点选择方法
            initial_method = st.selectbox(
                "初始点选择:",
                ["random", "centroid", "first"],
                format_func=lambda x: {
                    "random": "随机选择",
                    "centroid": "距质心最远",
                    "first": "第一个分子",
                }[x],
            )

            # GPU设置
            use_gpu = st.checkbox("使用GPU加速", value=True)
            use_half_precision = st.checkbox(
                "启用混合精度 (FP16)", value=True, disabled=not use_gpu
            )

        with col2:
            st.markdown("**筛选参数**")

            # 子集大小设置
            subset_method = st.radio("子集大小设置:", ["按数量", "按比例"])

            if subset_method == "按数量":
                subset_size = st.number_input(
                    "筛选数量:", min_value=1, max_value=100000, value=1000
                )
            else:
                subset_ratio = st.slider("筛选比例 (%):", 0.1, 50.0, 10.0, 0.5)
                # 根据文件估算数量
                if isinstance(row_count, int):
                    subset_size = max(1, int(row_count * subset_ratio / 100))
                    st.info(f"预计筛选数量: {subset_size}")
                else:
                    subset_size = 1000

            # 随机种子
            random_seed = st.number_input(
                "随机种子:", min_value=0, max_value=9999, value=42
            )

            # 输出文件名
            output_filename = st.text_input(
                "输出文件名:",
                value=f"subset_{subset_size}_{distance_method}_{selected_file}",
            )

        # 脚本生成按钮
        st.subheader("3. 脚本生成")

        if st.button("📄 生成执行脚本", type="primary"):
            try:
                # 准备配置
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                job_name = f"subset_{timestamp}"

                config = {
                    "input_file": file_path,
                    "output_file": os.path.join(
                        DATA_DIR, selected_folder, output_filename
                    ),
                    "subset_size": subset_size,
                    "distance_method": distance_method,
                    "initial_method": initial_method,
                    "use_gpu": use_gpu,
                    "use_half_precision": use_half_precision,
                    "random_seed": random_seed,
                    "job_name": job_name,
                }

                # 生成脚本文件路径
                script_dir = os.path.join(DATA_DIR, selected_folder)
                python_script_name = f"subset_selecting_{timestamp}.py"
                shell_script_name = f"subset_selecting_{timestamp}.sh"

                python_script_path = os.path.join(script_dir, python_script_name)
                shell_script_path = os.path.join(script_dir, shell_script_name)

                # 创建logs目录
                logs_dir = os.path.join(script_dir, "logs")
                os.makedirs(logs_dir, exist_ok=True)

                # 生成Python脚本
                python_content = generate_python_script(config)
                with open(python_script_path, "w", encoding="utf-8") as f:
                    f.write(python_content)
                os.chmod(python_script_path, 0o755)

                # 生成Shell脚本
                shell_content = generate_shell_script(python_script_path, config)
                with open(shell_script_path, "w", encoding="utf-8") as f:
                    f.write(shell_content)
                os.chmod(shell_script_path, 0o755)

                st.success("🎉 脚本生成成功！")

                st.code(
                    f"""
生成的文件:
📄 Python脚本: {python_script_name}
🔧 Shell脚本: {shell_script_name}
📁 日志目录: logs/
                """
                )

                # 保存到session state
                st.session_state.current_script = {
                    "python_path": python_script_path,
                    "shell_path": shell_script_path,
                    "config": config,
                    "job_name": job_name,
                    "logs_dir": logs_dir,
                    "stdout_log": os.path.join(logs_dir, f"{job_name}_stdout.log"),
                    "stderr_log": os.path.join(logs_dir, f"{job_name}_stderr.log"),
                }

                # 提供下载
                col1, col2 = st.columns(2)
                with col1:
                    with open(python_script_path, "rb") as f:
                        st.download_button(
                            "📥 下载Python脚本",
                            f.read(),
                            file_name=python_script_name,
                            mime="text/plain",
                        )
                with col2:
                    with open(shell_script_path, "rb") as f:
                        st.download_button(
                            "📥 下载Shell脚本",
                            f.read(),
                            file_name=shell_script_name,
                            mime="text/plain",
                        )

            except Exception as e:
                st.error(f"脚本生成失败: {str(e)}")

        # 执行控制
        st.subheader("4. 执行控制")

        if "current_script" in st.session_state:
            script_info = st.session_state.current_script
            st.info(f"当前脚本: {script_info['job_name']}")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("🚀 执行脚本", type="primary"):
                    try:
                        # 启动后台进程
                        process = subprocess.Popen(
                            ["/bin/bash", script_info["shell_path"]],
                            cwd=os.path.dirname(script_info["shell_path"]),
                            preexec_fn=os.setsid,  # 创建新的进程组
                        )

                        # 保存进程信息
                        if "running_processes" not in st.session_state:
                            st.session_state.running_processes = []

                        process_info = {
                            "pid": process.pid,
                            "job_name": script_info["job_name"],
                            "start_time": time.time(),
                            "stdout_log": script_info["stdout_log"],
                            "stderr_log": script_info["stderr_log"],
                            "output_file": script_info["config"]["output_file"],
                        }

                        st.session_state.running_processes.append(process_info)
                        st.success(f"✅ 后台进程已启动 (PID: {process.pid})")

                    except Exception as e:
                        st.error(f"启动失败: {str(e)}")

            with col2:
                if st.button("📋 查看stdout日志"):
                    if os.path.exists(script_info["stdout_log"]):
                        try:
                            with open(
                                script_info["stdout_log"], "r", encoding="utf-8"
                            ) as f:
                                lines = f.readlines()
                                last_50_lines = (
                                    lines[-50:] if len(lines) > 50 else lines
                                )
                                st.code("".join(last_50_lines))
                        except Exception as e:
                            st.error(f"读取日志失败: {str(e)}")
                    else:
                        st.warning("日志文件不存在")

            with col3:
                if st.button("📋 查看stderr日志"):
                    if os.path.exists(script_info["stderr_log"]):
                        try:
                            with open(
                                script_info["stderr_log"], "r", encoding="utf-8"
                            ) as f:
                                lines = f.readlines()
                                last_50_lines = (
                                    lines[-50:] if len(lines) > 50 else lines
                                )
                                st.code("".join(last_50_lines))
                        except Exception as e:
                            st.error(f"读取日志失败: {str(e)}")
                    else:
                        st.warning("日志文件不存在")

        else:
            st.info("请先生成脚本")

        # 运行状态监控
        st.subheader("5. 运行状态监控")

        if (
            "running_processes" in st.session_state
            and st.session_state.running_processes
        ):
            for i, proc_info in enumerate(st.session_state.running_processes):
                with st.expander(
                    f"任务: {proc_info['job_name']} (PID: {proc_info['pid']})",
                    expanded=True,
                ):

                    # 检查进程状态
                    if PSUTIL_AVAILABLE:
                        try:
                            process = psutil.Process(proc_info["pid"])
                            is_running = process.is_running()
                            status = "🟢 运行中" if is_running else "⚪ 已结束"
                        except psutil.NoSuchProcess:
                            is_running = False
                            status = "⚪ 已结束"
                    else:
                        # 简单检查
                        try:
                            os.kill(proc_info["pid"], 0)
                            is_running = True
                            status = "🟢 运行中"
                        except OSError:
                            is_running = False
                            status = "⚪ 已结束"

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**状态**: {status}")
                    with col2:
                        elapsed = (time.time() - proc_info["start_time"]) / 60
                        st.write(f"**运行时间**: {elapsed:.1f} 分钟")
                    with col3:
                        # 检查输出文件
                        if os.path.exists(proc_info["output_file"]):
                            file_size = os.path.getsize(proc_info["output_file"]) / (
                                1024 * 1024
                            )
                            st.write(f"**输出**: {file_size:.1f} MB")

                            # 提供下载
                            with open(proc_info["output_file"], "rb") as f:
                                st.download_button(
                                    "📥 下载结果",
                                    f.read(),
                                    file_name=os.path.basename(
                                        proc_info["output_file"]
                                    ),
                                    key=f"download_{i}",
                                )
                        else:
                            st.write("**输出**: 未生成")
        else:
            st.info("当前没有运行中的任务")
