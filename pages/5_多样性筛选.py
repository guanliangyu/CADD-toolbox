# -*- coding: utf-8 -*-
"""
CADD-Toolbox - 多样性筛选页面 (脚本生成 + 后台执行版)
-------------------------------------------------
1. 根据用户参数生成 subset_selecting.py & .sh
2. 支持一键后台执行、日志监控、结果下载
"""

import os, sys, json, time, gc, subprocess, multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

# ────────────────────────────────────────────────────────────────────────────────
# 基础设置 & 工具函数
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="多样性筛选 GPU版 (脚本模式)", layout="wide")
st.title("🚀 多样性筛选 GPU高性能版 — 脚本生成 & 后台执行")

DATA_DIR = os.path.abspath("data")
def list_data_folders():
    return [f for f in os.listdir(DATA_DIR)
            if os.path.isdir(os.path.join(DATA_DIR, f))] if os.path.exists(DATA_DIR) else []

def list_csv(folder):
    p = os.path.join(DATA_DIR, folder)
    return [f for f in os.listdir(p) if f.endswith(".csv")] if os.path.exists(p) else []

def file_info(fp):
    s = os.path.getsize(fp)/1024/1024
    m = datetime.fromtimestamp(os.path.getmtime(fp)).strftime("%Y-%m-%d %H:%M:%S")
    return s, m

# ────────────────────────────────────────────────────────────────────────────────
# 1️⃣  脚本模板生成函数
# ────────────────────────────────────────────────────────────────────────────────
PY_TEMPLATE = """#!/usr/bin/env python3

import os, sys, time, json, argparse, gc, io
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn.functional as F
    TORCH_OK = torch.cuda.is_available()
except ImportError:
    TORCH_OK = False

# --------------------------------------------------------------------------- #
# --------------------------- GPU  加速算法实现 ------------------------------ #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def greedy_cosine_gpu(data_norm: torch.Tensor,
                      k: int,
                      seed: int = 42,
                      initial: str = "random") -> List[int]:
    '''FP16 余弦距离贪心,显存常数级 O(N)'''

    torch.manual_seed(seed)
    n = data_norm.shape[0]
    dev = data_norm.device
    mask = torch.zeros(n, dtype=torch.bool, device=dev)
    min_dist = torch.full((n,), 2.0, dtype=data_norm.dtype, device=dev)   # 上界 2

    # 选首点
    if initial == "random":
        idx = torch.randint(0, n, (1,), device=dev).item()
    elif initial == "centroid":
        cent = F.normalize(data_norm.mean(0, keepdim=True), p=2, dim=1)
        d = 1 - (data_norm @ cent.T).squeeze()
        idx = torch.argmax(d).item()
    else:
        idx = 0

    mask[idx] = True
    last = data_norm[idx]

    print(f"[{datetime.now()}]  起始分子: {idx}")

    for i in range(1, k):
        sim = data_norm @ last          # N
        dist = 1 - sim
        min_dist = torch.minimum(min_dist, dist)
        min_dist[mask] = -1.0
        idx = torch.argmax(min_dist).item()
        mask[idx] = True
        last = data_norm[idx]
        if i % 10 == 0 or i == k - 1:
            print(f"[{datetime.now()}]  已选 {i+1}/{k}")

    return mask.nonzero(as_tuple=False).squeeze().cpu().tolist()

@torch.no_grad()
def greedy_euclidean_gpu(data: torch.Tensor,
                         k: int,
                         seed: int = 42,
                         initial: str = "random") -> List[int]:
    '''Squared Euclidean distance version, avoiding sqrt'''

    torch.manual_seed(seed)
    n = data.shape[0]
    dev = data.device
    mask = torch.zeros(n, dtype=torch.bool, device=dev)
    min_dist = torch.full((n,), float("inf"), dtype=data.dtype, device=dev)

    if initial == "random":
        idx = torch.randint(0, n, (1,), device=dev).item()
    elif initial == "centroid":
        cent = data.mean(0, keepdim=True)
        d2 = ((data - cent) ** 2).sum(1)
        idx = torch.argmax(d2).item()
    else:
        idx = 0

    mask[idx] = True
    last = data[idx]
    print(f"[{datetime.now()}]  起始分子: {idx}")

    for i in range(1, k):
        d2 = ((data - last) ** 2).sum(1)
        min_dist = torch.minimum(min_dist, d2)
        min_dist[mask] = -1.0
        idx = torch.argmax(min_dist).item()
        mask[idx] = True
        last = data[idx]
        if i % 10 == 0 or i == k - 1:
            print(f"[{datetime.now()}]  已选 {i+1}/{k}")

    return mask.nonzero(as_tuple=False).squeeze().cpu().tolist()

@torch.no_grad()
def greedy_manhattan_gpu(data: torch.Tensor,
                         k: int,
                         seed: int = 42,
                         initial: str = "random") -> List[int]:
    '''L1 距离逐点更新实现'''

    torch.manual_seed(seed)
    n = data.shape[0]
    dev = data.device
    mask = torch.zeros(n, dtype=torch.bool, device=dev)
    min_dist = torch.full((n,), float("inf"), dtype=data.dtype, device=dev)

    if initial == "random":
        idx = torch.randint(0, n, (1,), device=dev).item()
    elif initial == "centroid":
        cent = data.mean(0, keepdim=True)
        d1 = (data - cent).abs().sum(1)
        idx = torch.argmax(d1).item()
    else:
        idx = 0

    mask[idx] = True
    last = data[idx]
    print(f"[{datetime.now()}]  起始分子: {idx}")

    for i in range(1, k):
        d1 = (data - last).abs().sum(1)
        min_dist = torch.minimum(min_dist, d1)
        min_dist[mask] = -1.0
        idx = torch.argmax(min_dist).item()
        mask[idx] = True
        last = data[idx]
        if i % 10 == 0 or i == k - 1:
            print(f"[{datetime.now()}]  已选 {i+1}/{k}")

    return mask.nonzero(as_tuple=False).squeeze().cpu().tolist()

# ------------------------------ CPU 版本 ------------------------------------ #

def greedy_cpu(mat: np.ndarray,
               k: int,
               metric: str,
               seed: int = 42,
               initial: str = "random") -> List[int]:
    '''简化 CPU 实现,适合小数据或无 GPU 环境'''

    rng = np.random.default_rng(seed)
    n = mat.shape[0]
    if initial == "random":
        idx = rng.integers(n)
    elif initial == "centroid":
        cent = mat.mean(0)
        if metric == "cosine":
            cent /= (np.linalg.norm(cent) + 1e-9)
            d = 1 - mat @ cent
        elif metric == "euclidean":
            d = ((mat - cent) ** 2).sum(1)
        else:
            d = np.abs(mat - cent).sum(1)
        idx = d.argmax()
    else:
        idx = 0

    sel = [idx]
    print(f"[{datetime.now()}]  起始分子: {idx}")

    # 预处理
    if metric == "cosine":
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        dist = 1 - mat_norm @ mat_norm[idx]
    elif metric == "euclidean":
        dist = ((mat - mat[idx]) ** 2).sum(1)
    else:
        dist = np.abs(mat - mat[idx]).sum(1)

    dist[idx] = -1

    for i in range(1, k):
        idx = dist.argmax()
        sel.append(idx)

        if metric == "cosine":
            d_new = 1 - mat_norm @ mat_norm[idx]
        elif metric == "euclidean":
            d_new = ((mat - mat[idx]) ** 2).sum(1)
        else:
            d_new = np.abs(mat - mat[idx]).sum(1)

        dist = np.minimum(dist, d_new)
        dist[sel] = -1
        if i % 10 == 0 or i == k - 1:
            print(f"[{datetime.now()}]  已选 {i+1}/{k}")

    return sel

# --------------------------------------------------------------------------- #
# ------------------------------ 脚本主流程 ---------------------------------- #
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="多样性子集筛选 (GPU‑FP16 超优化版)")
    p.add_argument("-i", "--input",  required=True,  help="输入 CSV（描述符矩阵）")
    p.add_argument("-o", "--output", required=True,  help="输出 CSV（筛选结果）")
    p.add_argument("-k", "--subset_size", type=int, required=True,
                   help="需要选择的分子数量")
    p.add_argument("-m", "--metric", choices=["euclidean", "manhattan", "cosine"],
                   default="cosine", help="距离度量")
    p.add_argument("-a", "--algorithm", choices=["greedy", "sphere"],
                   default="greedy", help="筛选算法")
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    p.add_argument("--fp16", action="store_true", help="GPU 使用半精度")
    p.add_argument("--seed", type=int, default=42, help="随机种子")
    p.add_argument("--initial", choices=["random", "centroid", "first"],
                   default="random", help="首个分子选择策略")
    return p.parse_args()

def read_numeric_matrix(csv_path: str, chunksize: int = 200_000
                        ) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    '''分块读取数值列，峰值常数级内存'''

    sample = pd.read_csv(csv_path, nrows=100)
    num_cols  = sample.select_dtypes(include=[np.number]).columns.tolist()
    meta_cols = sample.select_dtypes(exclude=[np.number]).columns.tolist()
    if not num_cols:
        raise ValueError("CSV 中未检测到数值描述符列！")

    float32_dtypes = {c: "float32" for c in num_cols}
    arr_parts: List[np.ndarray] = []

    print(f"[INFO] 采用 chunksize={chunksize:,} 流式读取数值列 (float32→FP16) …")
    for chunk in pd.read_csv(csv_path,
                             usecols=num_cols,
                             dtype=float32_dtypes,
                             chunksize=chunksize,
                             engine="c",
                             low_memory=True):
        # NaN/inf 处理
        chunk.replace([np.inf, -np.inf], 0, inplace=True)
        chunk.fillna(0, inplace=True)

        # 转换为 FP16 以节省内存（安全转换，避免溢出）
        chunk_array = chunk.to_numpy(dtype=np.float32, copy=False)
        
        # 裁剪到 FP16 安全范围 [-65504, 65504] 避免溢出警告
        np.clip(chunk_array, -65500, 65500, out=chunk_array)
        arr_parts.append(chunk_array.astype(np.float16, copy=False))
        
        del chunk, chunk_array
        gc.collect()

    mat = np.concatenate(arr_parts, axis=0)
    del arr_parts
    gc.collect()

    # ── 元数据单独一次性读（通常远小于数值矩阵，可接受） ──
    meta = (pd.read_csv(csv_path, usecols=meta_cols, engine="c", low_memory=True)
            if meta_cols else pd.DataFrame(index=range(mat.shape[0])))

    print(f"[INFO] 数据流式读取完成 → shape={mat.shape}, dtype={mat.dtype}")
    return mat, num_cols, meta

def main():
    args = parse_args()
    t0 = time.time()
    print("="*60)
    print("多样性子集筛选脚本  (GPU‑FP16 超优化版)")
    print(f"启动时间: {datetime.now()}")
    print("="*60)

    # 1. 读数据 ---------------------------------------------------------------
    print(f"[INFO] 读取 CSV: {args.input}")
    mat, num_cols, meta = read_numeric_matrix(args.input)
    n, d = mat.shape
    print(f"[INFO] 数据形状: {n:,} × {d}  (float16)")

    # 2. 决定设备 -------------------------------------------------------------
    use_gpu = TORCH_OK and (not args.cpu)
    if use_gpu:
        dev = torch.device("cuda")
        dtype = torch.float16 if args.fp16 else torch.float32
        print(f"[INFO] 使用 GPU: {torch.cuda.get_device_name(0)}  dtype={dtype}")
        
        # ⚡ 内存优化：使用 torch.as_tensor 零拷贝 + pin_memory 加速传输
        if not args.fp16:
            mat = mat.astype(np.float32, copy=False)  # 仅在需要时转 FP32
        tensor = torch.as_tensor(mat, device=dev, dtype=dtype)
        del mat  # 立即删除 numpy 数组
        gc.collect()  # 强制垃圾回收
        print(f"[INFO] GPU 数据传输完成，numpy 数组已释放")

        # 余弦需要归一化
        if args.metric == "cosine":
            norms = tensor.pow(2).sum(1, keepdim=True)
            tensor = tensor * torch.rsqrt(norms + 1e-6)

        # 3. 筛选 -------------------------------------------------------------
        if args.algorithm == "greedy":
            if args.metric == "cosine":
                sel = greedy_cosine_gpu(tensor, args.subset_size,
                                        seed=args.seed, initial=args.initial)
            elif args.metric == "euclidean":
                sel = greedy_euclidean_gpu(tensor, args.subset_size,
                                           seed=args.seed, initial=args.initial)
            else:
                sel = greedy_manhattan_gpu(tensor, args.subset_size,
                                           seed=args.seed, initial=args.initial)
        else:
            raise NotImplementedError("sphere_exclusion GPU 版请后续扩展")

        # ⚡ 内存优化：立即清理 GPU 显存
        del tensor
        torch.cuda.empty_cache()
        gc.collect()
        print(f"[INFO] GPU 显存已清理")

    else:   # ---------------------------- CPU 路径 ---------------------------
        print("[INFO] 使用 CPU")
        if args.metric == "cosine":
            # 归一化
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)
        sel = greedy_cpu(mat, args.subset_size, args.metric,
                         seed=args.seed, initial=args.initial)
        
        # ⚡ 内存优化：CPU 计算完成后清理
        del mat
        gc.collect()
        print(f"[INFO] CPU 内存已清理")

    # 4. 写结果 ---------------------------------------------------------------
    print(f"[INFO] 开始写入结果，选中 {len(sel)} 个分子")
    
    selected_count = len(sel)
    
    # 输出1：基础信息CSV（仅元数据）
    meta_df = meta.iloc[sel].copy()
    meta_df.to_csv(args.output, index=False)
    
    # 输出2：完整信息CSV（元数据 + 3D描述符）
    full_output = args.output.replace('.csv', '_with_descriptors.csv')
    print(f"[INFO] 生成完整描述符文件...")
    
    # 重新分块读取选中分子的描述符数据（内存高效）
    print(f"[INFO] 重新读取选中分子的3D描述符数据...")
    
    # 读取选中行的完整数据
    selected_indices = set(sel)
    chunk_size = 50000
    selected_data_parts = []
    
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        chunk_indices = list(range(chunk_start, chunk_end))
        
        # 检查这个chunk中是否有选中的行
        chunk_selected = [i for i in chunk_indices if i in selected_indices]
        if not chunk_selected:
            continue
            
        # 读取这个chunk的数据
        chunk_df = pd.read_csv(args.input, 
                              skiprows=range(1, chunk_start + 1),
                              nrows=chunk_end - chunk_start,
                              low_memory=True)
        
        # 提取选中的行（相对于chunk的索引）
        relative_indices = [i - chunk_start for i in chunk_selected]
        selected_chunk = chunk_df.iloc[relative_indices].copy()
        selected_data_parts.append(selected_chunk)
        
        del chunk_df
        gc.collect()
    
    # 合并所有选中的数据
    if selected_data_parts:
        full_selected_df = pd.concat(selected_data_parts, ignore_index=True)
        del selected_data_parts
        gc.collect()
        
        # 写入完整CSV文件
        full_selected_df.to_csv(full_output, index=False)
        full_size_mb = os.path.getsize(full_output) / (1024*1024)
        
        del full_selected_df
        gc.collect()
        
        print(f"[SUCCESS] 完整描述符文件已生成 → {full_output}")
        print(f"[INFO] 完整文件大小: {full_size_mb:.2f} MB")
    else:
        print(f"[WARNING] 未能生成完整描述符文件")
    
    # 清理内存
    del meta, sel, meta_df
    gc.collect()

    print(f"[SUCCESS] 已选择 {selected_count}/{n} 个分子")
    print(f"[OUTPUT1] 基础信息: {args.output} ({os.path.getsize(args.output)/(1024*1024):.2f} MB)")
    if os.path.exists(full_output):
        print(f"[OUTPUT2] 完整信息: {full_output} ({os.path.getsize(full_output)/(1024*1024):.2f} MB)")
    print(f"[TIME] 总用时: {(time.time()-t0):.1f} s")
    print(f"[INFO] 内存优化版本执行完成")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("发生异常！", e)
        traceback.print_exc()
        sys.exit(1)
"""

SH_TEMPLATE = """#!/bin/bash
# 多样性子集筛选后台执行脚本
# 生成时间: 2025-06-12

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/subset_selecting.py"

# === 根据需要替换下面参数 ===========================================
INPUT_CSV="descriptors.csv"                  # 输入文件
OUTPUT_CSV="subset_1000_cosine.csv"          # 输出文件
K=1000                                       # 子集大小
METRIC="cosine"                              # euclidean | manhattan | cosine
ALGO="greedy"                                # greedy   | sphere
GPU_FLAG="--fp16"                            # 用 GPU+FP16:  --fp16   , 仅 CPU: --cpu
# ====================================================================

STD_LOG="${SCRIPT_DIR}/logs/$(basename $OUTPUT_CSV .csv)_stdout.log"
ERR_LOG="${SCRIPT_DIR}/logs/$(basename $OUTPUT_CSV .csv)_stderr.log"
mkdir -p "$(dirname "$STD_LOG")"

echo ">>>>>>>>  多样性筛选开始  $(date)  <<<<<<<<" | tee -a "$STD_LOG" "$ERR_LOG"
(
  python3 "$PY_SCRIPT" \\
        --input  "$INPUT_CSV" \\
        --output "$OUTPUT_CSV" \\
        --subset_size $K \\
        --metric $METRIC \\
        --algorithm $ALGO \\
        $GPU_FLAG \\
        --seed 42 \\
        --initial random
) 2> >(tee -a "$ERR_LOG" >&2) | tee -a "$STD_LOG"
CODE=${PIPESTATUS[0]}

if [[ $CODE -eq 0 ]]; then
  echo "✅  成功完成  $(date)" | tee -a "$STD_LOG" "$ERR_LOG"
else
  echo "❌  失败退出($CODE)  $(date)" | tee -a "$STD_LOG" "$ERR_LOG"
fi
exit $CODE
"""

def generate_subset_py(cfg:dict, inp:str, outp:str, dst:str):
    """写 subset_selecting.py 到 dst"""
    py = PY_TEMPLATE
    # 模板本身已经是通用 CLI,无需插值,只写文件
    with open(dst, "w", encoding="utf-8") as f:
        f.write(py)
    os.chmod(dst, 0o755)
    return dst

def generate_subset_sh(py_path:str,
                       inp:str, outp:str, k:int,
                       metric:str, algo:str,
                       gpu:bool, fp16:bool,
                       dst:str, log_dir:str):
    os.makedirs(log_dir, exist_ok=True)
    flag = "--fp16" if (gpu and fp16) else ("" if gpu else "--cpu")
    # 使用绝对路径替换模板中的文件名
    sh = SH_TEMPLATE.replace("subset_selecting.py", os.path.basename(py_path)) \
                    .replace("descriptors.csv",           os.path.abspath(inp)) \
                    .replace("subset_1000_cosine.csv",    os.path.abspath(outp)) \
                    .replace("1000",           str(k)) \
                    .replace("cosine",         metric) \
                    .replace("greedy",         algo) \
                    .replace("--fp16", flag if flag else " ")  # 若 --cpu 则留空
    with open(dst, "w", encoding="utf-8") as f:
        f.write(sh)
    os.chmod(dst, 0o755)
    return dst, os.path.join(log_dir,
            f"{os.path.splitext(os.path.basename(outp))[0]}_stdout.log"), \
           os.path.join(log_dir,
            f"{os.path.splitext(os.path.basename(outp))[0]}_stderr.log")

# ────────────────────────────────────────────────────────────────────────────────
# 2️⃣  用户界面 – 参数选择
# ────────────────────────────────────────────────────────────────────────────────
st.subheader("1. 选择输入 CSV 文件")
folders = list_data_folders()
if not folders:
    st.warning("⚠️  data 目录为空")
    st.stop()

folder = st.selectbox("数据文件夹", folders)
csv_files = list_csv(folder)
if not csv_files:
    st.warning("该文件夹下没有 CSV")
    st.stop()

csv_file = st.selectbox("描述符 CSV", csv_files)
csv_path = os.path.abspath(os.path.join(DATA_DIR, folder, csv_file))
size_mb, mtime = file_info(csv_path)
st.info(f"📄 大小: {size_mb:.1f} MB   🕒 修改: {mtime}")

# ──  筛选参数
st.subheader("2. 筛选参数")
col1, col2, col3 = st.columns(3)
with col1:
    subset_size = st.number_input("子集大小", min_value=1, value=1000, step=100)
    metric = st.selectbox("距离度量", ["euclidean", "manhattan", "cosine"])
with col2:
    algo = st.selectbox("筛选算法", ["greedy"])  # sphere_exclusion GPU 版暂不放出
    initial = st.selectbox("首个分子策略", ["random", "centroid", "first"])
with col3:
    use_gpu = st.checkbox("使用 GPU", value=True)
    fp16    = st.checkbox("启用 FP16", value=False, disabled=not use_gpu)

# ──  输出文件名
st.subheader("3. 输出设置")
out_default = f"subset_{subset_size}_{metric}_{csv_file}"
output_csv  = st.text_input("输出文件名", value=out_default)

# ────────────────────────────────────────────────────────────────────────────────
# 3️⃣  生成脚本
# ────────────────────────────────────────────────────────────────────────────────
if st.button("📄 生成脚本", type="primary"):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    script_dir = os.path.abspath(os.path.join(DATA_DIR, folder))
    py_path = os.path.abspath(os.path.join(script_dir, f"subset_select_{stamp}.py"))
    sh_path = os.path.abspath(os.path.join(script_dir, f"subset_select_{stamp}.sh"))
    log_dir = os.path.abspath(os.path.join(script_dir, "logs"))
    out_path = os.path.abspath(os.path.join(script_dir, output_csv))

    cfg = dict()  # 目前 Py 模板无需 cfg
    generate_subset_py(cfg, csv_path, out_path, py_path)
    sh_path, stdout_log, stderr_log = generate_subset_sh(
        py_path, csv_path, out_path,
        subset_size, metric, algo, use_gpu, fp16,
        sh_path, log_dir)

    st.success("脚本生成完毕")
    st.code(f"Python: {py_path}\nShell : {sh_path}")

    # 放入 session_state
    st.session_state.script_info = {
        "python": py_path,
        "shell" : sh_path,
        "dir"   : script_dir,
        "stdout": stdout_log,
        "stderr": stderr_log,
        "output": out_path,
        "name"  : os.path.splitext(os.path.basename(sh_path))[0]
    }

    with open(py_path, "rb") as f:
        st.download_button("下载 Python 脚本", f, file_name=os.path.basename(py_path))
    with open(sh_path, "rb") as f:
        st.download_button("下载 Shell 脚本",  f, file_name=os.path.basename(sh_path))

# ────────────────────────────────────────────────────────────────────────────────
# 4️⃣  执行脚本
# ────────────────────────────────────────────────────────────────────────────────
st.divider()
st.header("🚀 执行 & 监控")

if "script_info" in st.session_state:
    info = st.session_state.script_info
    st.info(f"已加载脚本 **{info['name']}**")

    if st.button("🚀 执行脚本", type="primary"):
        if os.path.exists(info["shell"]):
            # 使用绝对路径执行脚本
            abs_shell_path = os.path.abspath(info["shell"])
            abs_work_dir = os.path.abspath(info["dir"])
            proc = subprocess.Popen(
                ["/bin/bash", abs_shell_path],
                cwd=abs_work_dir, preexec_fn=os.setsid)
            st.success(f"后台进程已启动 (PID {proc.pid})")

            if "running" not in st.session_state:
                st.session_state.running = []
            st.session_state.running.append({
                "pid": proc.pid,
                "name": info["name"],
                "start": time.time(),
                "stdout": info["stdout"],
                "stderr": info["stderr"],
                "output": info["output"]
            })
        else:
            st.error("脚本不存在,请重新生成。")
else:
    st.info("请先生成脚本。")

# ────────────────────────────────────────────────────────────────────────────────
# 5️⃣  日志查看 / 进程监控（与 3D 页面相同逻辑,可复用）
# ────────────────────────────────────────────────────────────────────────────────
if "running" in st.session_state and st.session_state.running:
    st.subheader("🔄 运行中任务")
    import psutil
    for idx, r in enumerate(st.session_state.running):
        with st.expander(f"{r['name']}  (PID {r['pid']})", expanded=False):
            alive = psutil.pid_exists(r['pid']) and psutil.Process(r['pid']).is_running()
            st.markdown(f"**状态**: {'🟢 运行中' if alive else '⚪ 已结束'}")
            st.markdown(f"**运行时间**: {(time.time()-r['start'])/60:.1f} 分钟")
            if os.path.exists(r["output"]):
                size = os.path.getsize(r["output"])/1024/1024
                st.success(f"✅ 已生成输出 ({size:.1f} MB)")
                with open(r["output"], "rb") as f:
                    st.download_button("下载结果", f,
                        file_name=os.path.basename(r["output"]))
            colA, colB = st.columns(2)
            with colA:
                if st.button("查看 stdout 尾部", key=f"out{idx}"):
                    if os.path.exists(r["stdout"]):
                        lines = open(r["stdout"], encoding="utf-8").readlines()[-50:]
                        st.code("".join(lines))
            with colB:
                if st.button("查看 stderr 尾部", key=f"err{idx}"):
                    if os.path.exists(r["stderr"]):
                        lines = open(r["stderr"], encoding="utf-8").readlines()[-50:]
                        st.code("".join(lines))
