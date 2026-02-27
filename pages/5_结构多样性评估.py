"""
CADD-Toolbox - 结构多样性评估页面
基于数值化指纹进行结构多样性分析与比较
"""

import os
import gc
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from utils.tanimoto_parallel import get_max_available_cpus
from utils.structure_diversity_data import (
    DEFAULT_META_SAMPLE_ROWS,
    build_array_signature,
    ensure_faiss_compatible,
    get_file_info,
    list_csv_files_in_folder,
    list_data_folders,
    load_fingerprints_from_csv,
    read_fps_cached,
    subsample_by_ratio,
    subsample_fingerprints,
)
from utils.structure_diversity_similarity import (
    cached_knn_similarity,
    cached_pairwise_similarity,
    calculate_diversity_metrics,
    compute_similarity_matrix_from_fingerprints,
    diversity_stats,
)
from utils.structure_diversity_analysis import (
    embed_umap,
    perform_clustering_analysis,
    perform_dimensionality_reduction,
    perform_optimized_clustering_analysis,
)
from utils.structure_diversity_visualization import (
    monitor_memory_usage,
    plot_clustering_results,
    plot_distribution_comparison,
    plot_nearest_neighbor_distribution,
    render_physchem_distribution_single,
    plot_single_dataset_distribution,
    render_diversity_metrics_list,
    render_physchem_distribution_comparison,
    subset_meta_with_indices,
)

# 尝试导入FAISS
FAISS_IMPORT_ERROR = None
try:
    import faiss
    FAISS_AVAILABLE = True
    # 检查FAISS GPU支持
    try:
        if faiss.get_num_gpus() > 0:
            FAISS_GPU_AVAILABLE = True
        else:
            FAISS_GPU_AVAILABLE = False
    except:
        FAISS_GPU_AVAILABLE = False
except ImportError:
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False
    FAISS_IMPORT_ERROR = "FAISS未安装，将使用sklearn进行相似性计算（较慢）"

# 尝试导入GPU相关库
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import cuml  # noqa: F401
    CUML_AVAILABLE = True
except Exception:
    CUML_AVAILABLE = False

# 抑制警告
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if TORCH_AVAILABLE:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# 设置全局随机种子以确保一致性
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

st.set_page_config(page_title="结构多样性评估", layout="wide")
st.title("📊 结构多样性评估（指纹数据）")

if FAISS_IMPORT_ERROR:
    st.warning(f"⚠️ {FAISS_IMPORT_ERROR}")

st.markdown("""
基于数值化的分子指纹执行多样性分析与可视化。

🔬 **指纹来源**：支持 Morgan/拓扑指纹、E3FP、USRCAT 等数值化特征  
📈 **聚类分析**：MiniBatch K-means、HDBSCAN/DBSCAN 聚类  
📊 **可视化**：IncrementalPCA + UMAP、t-SNE、PCA 等降维  
⚡ **性能优化**：流式加载 + k-NN 采样 + 可选 GPU/FAISS 加速  

> **🚀 核心能力：**
> - **双模式**：优化模式适合中大规模数据（几十万样本），兼容模式用于小规模完整矩阵分析；
> - **资源友好**：float16/分块读取，把 O(N²) 复杂度降为 O(N)；
> - **灵活扩展**：可对任意数值指纹进行比较，无需区分指纹维度；
> - **GPU 支持**：自动检测 FAISS-GPU、CuML、PyTorch 等加速库。

> **💡 使用建议：**
> - 请确保 CSV 中包含数值型指纹列（列名包含 `fingerprint/fp/descriptor/feature/bit/e3fp/rocs/usrcat` 更易识别）；
> - 默认“优化模式”即可处理大部分实际数据，若需要完整相似性矩阵可切换到“兼容模式”。
""")

# 显示当前随机种子
st.info(f"🎲 当前随机种子: {RANDOM_SEED}")

if CUDA_AVAILABLE and TORCH_AVAILABLE and CUML_AVAILABLE:
    st.success("✅ GPU加速可用：CUDA + CuML")
elif CUDA_AVAILABLE and TORCH_AVAILABLE:
    st.warning("⚠️ GPU部分可用：CUDA可用但CuML不可用")
else:
    st.info("ℹ️ 仅CPU模式：GPU不可用")

# 显示FAISS状态
if FAISS_AVAILABLE:
    if FAISS_GPU_AVAILABLE:
        st.success("✅ FAISS-GPU可用：高速相似性搜索")
    else:
        st.info("ℹ️ FAISS-CPU可用：快速相似性搜索")
else:
    st.warning("⚠️ FAISS不可用：将使用sklearn（较慢）")

# 数据目录设置
DATA_DIR = "data"

# ===========================================
# 主程序界面
# ===========================================

# 文件选择界面
st.subheader("1. 选择包含指纹的数据文件")

folders = list_data_folders(DATA_DIR)

if not folders:
    st.warning("data目录下没有找到任何文件夹")
    st.stop()

selected_folder = st.selectbox("选择数据文件夹:", folders)
evaluate_a_only = False
selected_fileA = None
selected_fileB = None
fileA_path = None
fileB_path = None

if selected_folder:
    csv_files = list_csv_files_in_folder(selected_folder, DATA_DIR)
    
    if not csv_files:
        st.warning(f"文件夹 {selected_folder} 中没有CSV文件")
        st.stop()
    
    evaluate_a_only = st.checkbox(
        "仅评估数据集A（不做筛选前后对比）",
        value=False,
        key="evaluate_a_only",
    )

    if evaluate_a_only:
        st.markdown("**数据集A (用于单库评估)**")
        selected_fileA = st.selectbox("选择数据集A的CSV文件:", csv_files, key="fileA")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**数据集A (原始数据集)**")
            selected_fileA = st.selectbox("选择数据集A的CSV文件:", csv_files, key="fileA")

        with col2:
            st.markdown("**数据集B (筛选后数据集)**")
            selected_fileB = st.selectbox("选择数据集B的CSV文件:", csv_files, key="fileB")

    if selected_fileA:
        fileA_path = os.path.join(DATA_DIR, selected_folder, selected_fileA)
        file_infoA = get_file_info(fileA_path)
    else:
        file_infoA = None

    if not evaluate_a_only and selected_fileB:
        fileB_path = os.path.join(DATA_DIR, selected_folder, selected_fileB)
        file_infoB = get_file_info(fileB_path)
    else:
        file_infoB = None

    if evaluate_a_only and file_infoA:
        st.info(f"**文件A信息:**\n- 大小: {file_infoA['size_mb']:.1f} MB\n- 修改时间: {file_infoA['modified']}")
    elif file_infoA and file_infoB:
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"**文件A信息:**\n- 大小: {file_infoA['size_mb']:.1f} MB\n- 修改时间: {file_infoA['modified']}")

        with col2:
            st.info(f"**文件B信息:**\n- 大小: {file_infoB['size_mb']:.1f} MB\n- 修改时间: {file_infoB['modified']}")

# 参数设置
st.subheader("2. 分析参数设置")

# 添加分析模式选择
sample_ratio = 1.0
representative_ratio_A = 1.0
use_representative_sampling_A = False
metadata_mode = "sample"
metadata_sample_rows = DEFAULT_META_SAMPLE_ROWS
stream_chunksize = 200_000
with st.expander("⚡ 分析模式选择", expanded=True):
    analysis_mode = st.selectbox(
        "选择分析模式",
        ["优化模式 (推荐)", "兼容模式"],
        help="优化模式：使用k-NN + 采样 + PCA-UMAP，适合大数据集；兼容模式：完整相似性矩阵，适合小数据集"
    )
    
    if analysis_mode == "优化模式 (推荐)":
        st.success("✅ 使用优化算法：流式读取 + k-NN相似性 + PCA-UMAP降维 + 可选聚类(K-means/DBSCAN)")
        st.caption("提示：优化模式始终采用 IncrementalPCA→UMAP 流程，下面“降维方法”仅在兼容模式下生效。")
        col1, col2, col3 = st.columns(3)
        with col1:
            fp_dtype = st.selectbox("指纹数据类型", ["float16", "float32"], 
                                    help="float16节省一半内存但精度稍低")
        with col2:
            k_neighbors = int(st.number_input("k-NN邻居数", min_value=1, max_value=1000, value=30, step=1,
                                    help="建议范围：20-100；过小会忽略邻域信息，过大增加时间/内存开销。"))
        with col3:
            n_sample_pairs = st.number_input("采样对数", 100_000, 5_000_000, 2_000_000,
                                           help="用于全局统计的随机采样对数")
        sample_ratio = st.slider(
            "采样比例 (优化模式)",
            min_value=1,
            max_value=100,
            value=100,
            step=1,
            help="以百分比抽样两个数据集，防止样本量过大时计算时间过长。"
        ) / 100.0
    else:
        st.warning("⚠️ 兼容模式：计算完整相似性矩阵，内存消耗大，仅适合小数据集")
        max_matrix_samples = st.number_input(
            "最大相似性矩阵样本数",
            min_value=500,
            max_value=20_000,
            value=5_000,
            step=500,
            help="超过该阈值时将随机抽样以避免 O(N²) 矩阵导致内存瓶颈。"
        )

with st.expander("📥 数据读取设置", expanded=False):
    metadata_mode_labels = {
        "none": "不读取元数据（最快）",
        "sample": "采样读取元数据（推荐）",
        "full": "完整读取元数据（最慢）",
    }
    metadata_mode = st.selectbox(
        "元数据加载方式",
        ["none", "sample", "full"],
        format_func=lambda x: metadata_mode_labels.get(x, x),
        index=1,
        help="元数据仅用于辅助展示，不影响相似性与聚类计算。"
    )
    if metadata_mode == "sample":
        metadata_sample_rows = int(st.number_input(
            "元数据采样行数",
            min_value=100,
            max_value=100_000,
            value=DEFAULT_META_SAMPLE_ROWS,
            step=500
        ))
    stream_chunksize = int(st.number_input(
        "CSV分块大小",
        min_value=50_000,
        max_value=1_000_000,
        value=200_000,
        step=50_000,
        help="分块越大速度通常越快，但峰值内存更高。"
    ))

with st.expander("🎯 代表样本设置", expanded=False):
    use_representative_sampling_A = st.checkbox(
        "启用数据集A代表采样",
        help="随机按比例抽取数据集A中的样本，后续分析将仅使用代表样本。"
    )
    if use_representative_sampling_A:
        representative_ratio_A = st.slider(
            "数据集A代表样本比例",
            min_value=1,
            max_value=100,
            value=50,
            step=1,
            help="用于替代完整数据集A的代表样本比例，单位为%。"
        ) / 100.0

# 为两种模式预置参数，避免变量未定义
dim_reduction_method = "UMAP"
perplexity = 30
n_neighbors = 15
min_dist = 0.1
cluster_method = "K-means"
n_clusters = 5
eps = 0.3
min_samples = 5
max_cpu_count = get_max_available_cpus()
tanimoto_n_jobs = min(max_cpu_count, max(1, min(8, max_cpu_count)))
tanimoto_chunk_rows = 128


def _parse_float_from_text(
    raw_value: str,
    *,
    default: float,
    label: str,
    min_value: float | None = None,
) -> float:
    """将文本框输入解析为浮点数，并进行最小值校验。"""
    try:
        parsed = float(str(raw_value).strip())
    except (TypeError, ValueError):
        st.warning(f"{label} 输入无效，已回退为默认值 {default}")
        return float(default)

    if min_value is not None and parsed < min_value:
        st.warning(f"{label} 不能小于 {min_value}，已自动修正为 {min_value}")
        return float(min_value)
    return float(parsed)


def _parse_int_from_text(
    raw_value: str,
    *,
    default: int,
    label: str,
    min_value: int | None = None,
) -> int:
    """将文本框输入解析为整数，并进行最小值校验。"""
    try:
        parsed = int(str(raw_value).strip())
    except (TypeError, ValueError):
        st.warning(f"{label} 输入无效，已回退为默认值 {default}")
        return int(default)

    if min_value is not None and parsed < min_value:
        st.warning(f"{label} 不能小于 {min_value}，已自动修正为 {min_value}")
        return int(min_value)
    return int(parsed)


with st.expander("📊 相似性和聚类参数", expanded=True):
    if analysis_mode == "优化模式 (推荐)":
        col1, col2 = st.columns(2)
    else:
        col1, col2, col3 = st.columns(3)
    
    with col1:
        if analysis_mode == "优化模式 (推荐)":
            similarity_options = ["cosine", "euclidean"]
            similarity_help = (
                "优化模式使用 FAISS k-NN 与采样，当前支持 cosine/euclidean。"
                " 如需 tanimoto，请切换到兼容模式。"
            )
        else:
            similarity_options = ["cosine", "euclidean", "tanimoto"]
            similarity_help = "cosine: 余弦相似性; euclidean: 欧几里得距离; tanimoto: Tanimoto系数"

        similarity_metric = st.selectbox(
            "相似性度量方法",
            similarity_options,
            help=similarity_help
        )

        cluster_method = st.selectbox(
            "聚类方法",
            ["K-means", "DBSCAN"],
            help="选择单一聚类算法进行分析与可视化。"
        )
    
    with col2:
        if cluster_method == "K-means":
            n_clusters = st.slider(
                "K-means聚类数",
                min_value=2,
                max_value=50,
                value=5,
                help="K-means聚类的簇数"
            )
            st.caption("当前未使用 DBSCAN 参数。")
        else:
            eps_text = st.text_input(
                "DBSCAN eps参数",
                value="0.3",
                key="dbscan_eps_input",
                help="DBSCAN聚类的邻域半径（建议 > 0，常用 0.1~1.0）"
            )
            eps = _parse_float_from_text(
                eps_text,
                default=0.3,
                label="DBSCAN eps参数",
                min_value=0.000001,
            )

            min_samples_text = st.text_input(
                "DBSCAN最小样本数",
                value="5",
                key="dbscan_min_samples_input",
                help="DBSCAN聚类的最小样本数（整数，建议 >= 2）"
            )
            min_samples = _parse_int_from_text(
                min_samples_text,
                default=5,
                label="DBSCAN最小样本数",
                min_value=2,
            )
            st.caption("当前未使用 K-means 参数。")
    
    if analysis_mode == "优化模式 (推荐)":
        st.caption("优化模式中降维流程固定为 IncrementalPCA→UMAP，t-SNE/UMAP/PCA 选择仅兼容模式可用。")
    else:
        with col3:
            dim_reduction_method = st.selectbox(
                "降维方法",
                ["t-SNE", "UMAP", "PCA"],
                help="兼容模式下用于绘制聚类/分布图"
            )
            
            if dim_reduction_method == "t-SNE":
                perplexity_text = st.text_input(
                    "t-SNE困惑度",
                    value="30",
                    key="tsne_perplexity_input",
                    help="推荐值：5-50，影响局部vs全局结构"
                )
                perplexity = _parse_float_from_text(
                    perplexity_text,
                    default=30.0,
                    label="t-SNE困惑度",
                    min_value=1.0,
                )
            elif dim_reduction_method == "UMAP":
                n_neighbors = st.slider(
                    "UMAP邻居数",
                    min_value=2,
                    max_value=100,
                    value=15,
                    help="推荐值：5-50，控制局部vs全局结构平衡"
                )
                min_dist = st.slider(
                    "UMAP最小距离",
                    min_value=0.01,
                    max_value=0.99,
                    value=0.1,
                    step=0.05,
                    format="%.2f",
                    help="推荐值：0.1-0.2，控制嵌入点间最小距离"
                )

        if similarity_metric == "tanimoto":
            st.markdown("**Tanimoto 多CPU加速设置**")
            tani_col1, tani_col2 = st.columns(2)
            with tani_col1:
                tanimoto_n_jobs = int(st.number_input(
                    "Tanimoto并行CPU数",
                    min_value=1,
                    max_value=max_cpu_count,
                    value=tanimoto_n_jobs,
                    step=1,
                    help=f"自动识别到系统可用CPU上限: {max_cpu_count}"
                ))
            with tani_col2:
                tanimoto_chunk_rows = int(st.number_input(
                    "Tanimoto行块大小",
                    min_value=16,
                    max_value=1024,
                    value=128,
                    step=16,
                    help="每个任务处理的行数。越大开销越小，但内存占用越高。"
                ))
            st.caption(f"当前进程可用CPU上限: {max_cpu_count}")

# Debug模式设置
debug_mode = False
force_device = None
with st.expander("🔧 Debug模式", expanded=False):
    debug_mode = st.checkbox("启用Debug模式", help="允许手动选择CPU/GPU并对比计算结果")
    
    if debug_mode:
        custom_seed = st.number_input(
            "自定义随机种子", 
            min_value=1, 
            max_value=99999, 
            value=RANDOM_SEED, 
            help="设置随机种子以确保结果可重现性"
        )
        if custom_seed != RANDOM_SEED:
            globals()['RANDOM_SEED'] = custom_seed
            np.random.seed(custom_seed)
            st.info(f"🔧 已更新随机种子为: {custom_seed}")
        
        if analysis_mode == "兼容模式":
            force_device = st.selectbox(
                "强制使用设备",
                ["auto", "cpu", "gpu"],
                help="auto: 自动选择最优设备; cpu: 强制使用CPU; gpu: 强制使用GPU(如果可用)"
            )
        else:
            st.caption("优化模式下设备选择由算法自动决定，`强制使用设备`仅在兼容模式生效。")

# 开始分析
st.subheader("3. 分析结果")

results_container = st.container()
can_start_analysis = bool(fileA_path) and (evaluate_a_only or bool(fileB_path))
with results_container:
    if not can_start_analysis:
        st.info("请先选择要分析的CSV文件，然后点击开始评估按钮")
    else:
        st.markdown("**准备分析的文件:**")
        if evaluate_a_only:
            st.text(f"数据集A: {selected_fileA}")
            st.caption("当前模式：仅评估数据集A，不进行筛选前后对比。")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.text(f"数据集A: {selected_fileA}")
            with col2:
                st.text(f"数据集B: {selected_fileB}")

if st.button("🚀 开始指纹多样性分析", type="primary", disabled=not can_start_analysis) and can_start_analysis:
    with st.spinner("正在进行指纹多样性分析..."):
        # 显示内存使用情况
        mem_usage = monitor_memory_usage()
        st.sidebar.info(
            f"内存使用情况:\n"
            f"- RSS: {mem_usage['rss']:.1f} MB\n"
            f"- 内存占用: {mem_usage['percent']:.1f}%"
        )
        
        fingerprints_A = fingerprints_B = None
        fp_cols_A = fp_cols_B = None
        meta_A = meta_B = None
        ratio_idx_A = ratio_idx_B = None
        representative_idx_A = None

        if analysis_mode == "优化模式 (推荐)":
            if similarity_metric == "tanimoto":
                st.error("优化模式暂不支持 tanimoto，请切换到兼容模式或使用 cosine/euclidean。")
                st.stop()
            # 使用优化的流式加载
            st.info("🚀 使用优化模式进行分析...")
            fingerprints_A, fp_cols_A, meta_A = read_fps_cached(
                fileA_path,
                chunksize=stream_chunksize,
                fp_dtype=fp_dtype,
                meta_mode=metadata_mode,
                meta_sample_rows=metadata_sample_rows
            )
            if not evaluate_a_only:
                fingerprints_B, fp_cols_B, meta_B = read_fps_cached(
                    fileB_path,
                    chunksize=stream_chunksize,
                    fp_dtype=fp_dtype,
                    meta_mode=metadata_mode,
                    meta_sample_rows=metadata_sample_rows
                )
            fingerprints_A, meta_A, ratio_idx_A = subsample_by_ratio(
                fingerprints_A,
                sample_ratio,
                "数据集A",
                meta_A,
                random_seed=RANDOM_SEED
            )
            if not evaluate_a_only:
                fingerprints_B, meta_B, ratio_idx_B = subsample_by_ratio(
                    fingerprints_B,
                    sample_ratio,
                    "数据集B",
                    meta_B,
                    random_seed=RANDOM_SEED
                )
            if use_representative_sampling_A:
                fingerprints_A, meta_A, representative_idx_A = subsample_by_ratio(
                    fingerprints_A,
                    representative_ratio_A,
                    "数据集A代表样本",
                    meta_A,
                    random_seed=RANDOM_SEED
                )
        else:
            # 使用兼容模式
            st.info("⚡ 使用兼容模式进行分析...")
            fingerprints_A, meta_A, fp_cols_A = load_fingerprints_from_csv(
                fileA_path,
                chunksize=stream_chunksize,
                meta_mode=metadata_mode,
                meta_sample_rows=metadata_sample_rows
            )
            if not evaluate_a_only:
                fingerprints_B, meta_B, fp_cols_B = load_fingerprints_from_csv(
                    fileB_path,
                    chunksize=stream_chunksize,
                    meta_mode=metadata_mode,
                    meta_sample_rows=metadata_sample_rows
                )
            if use_representative_sampling_A:
                fingerprints_A, meta_A, representative_idx_A = subsample_by_ratio(
                    fingerprints_A,
                    representative_ratio_A,
                    "数据集A代表样本",
                    meta_A,
                    random_seed=RANDOM_SEED
                )

        if fingerprints_A is not None and (evaluate_a_only or fingerprints_B is not None):
            results_container.empty()
            
            with results_container:
                if evaluate_a_only:
                    st.success(f"✅ 成功加载: 数据集A {len(fingerprints_A):,}个样本")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("数据集A样本数", f"{len(fingerprints_A):,}")
                    with col2:
                        st.metric("指纹维度", len(fp_cols_A))
                    with col3:
                        st.metric("评估模式", "仅数据集A")
                else:
                    st.success(f"✅ 成功加载: 数据集A {len(fingerprints_A):,}个样本，数据集B {len(fingerprints_B):,}个样本")

                    # 显示基本统计信息
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("数据集A样本数", f"{len(fingerprints_A):,}")
                    with col2:
                        st.metric("数据集B样本数", f"{len(fingerprints_B):,}")
                    with col3:
                        st.metric("指纹维度", len(fp_cols_A))
                    with col4:
                        st.metric("选择比例", f"{len(fingerprints_B)/len(fingerprints_A)*100:.1f}%")
                
                # 根据分析模式选择不同的计算方法
                if analysis_mode == "优化模式 (推荐)":
                    # ===========================================
                    # 优化模式：使用k-NN + 采样方法
                    # ===========================================
                    st.markdown("### ⚡ 优化模式指纹分析")
                    
                    # L2归一化（余弦相似性需要）
                    if similarity_metric == "cosine":
                        if FAISS_AVAILABLE:
                            import faiss
                            fingerprints_A = ensure_faiss_compatible(fingerprints_A)
                            faiss.normalize_L2(fingerprints_A)
                            if not evaluate_a_only:
                                fingerprints_B = ensure_faiss_compatible(fingerprints_B)
                                faiss.normalize_L2(fingerprints_B)
                        else:
                            fingerprints_A = fingerprints_A / (np.linalg.norm(fingerprints_A, axis=1, keepdims=True) + 1e-10)
                            if not evaluate_a_only:
                                fingerprints_B = fingerprints_B / (np.linalg.norm(fingerprints_B, axis=1, keepdims=True) + 1e-10)

                    # 计算k-NN相似性
                    st.info(f"计算k-NN相似性 (k={k_neighbors})...")
                    dataset_sig_A = build_array_signature(fingerprints_A)
                    dataset_sig_B = build_array_signature(fingerprints_B) if not evaluate_a_only else None
                    try:
                        knn_A = cached_knn_similarity(
                            fingerprints_A,
                            metric=similarity_metric,
                            k=k_neighbors,
                            use_gpu=True,
                            dataset_sig=dataset_sig_A
                        )
                        knn_B = None
                        if not evaluate_a_only:
                            knn_B = cached_knn_similarity(
                                fingerprints_B,
                                metric=similarity_metric,
                                k=k_neighbors,
                                use_gpu=True,
                                dataset_sig=dataset_sig_B
                            )
                    except Exception as e:
                        st.error(f"k-NN计算出错: {str(e)}")
                        st.info("回退到sklearn计算...")
                        # 回退到sklearn方法
                        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
                        if similarity_metric == "cosine":
                            sim_A = cosine_similarity(fingerprints_A)
                            sim_B = cosine_similarity(fingerprints_B) if not evaluate_a_only else None
                        else:
                            dist_A = euclidean_distances(fingerprints_A)
                            max_dist_A = dist_A.max()
                            sim_A = np.ones_like(dist_A, dtype=np.float32) if max_dist_A <= 1e-12 else 1 - (dist_A / max_dist_A)
                            sim_B = None
                            if not evaluate_a_only:
                                dist_B = euclidean_distances(fingerprints_B)
                                max_dist_B = dist_B.max()
                                sim_B = np.ones_like(dist_B, dtype=np.float32) if max_dist_B <= 1e-12 else 1 - (dist_B / max_dist_B)

                        # 提取k-NN
                        np.fill_diagonal(sim_A, -1)
                        knn_A = np.sort(sim_A, axis=1)[:, -k_neighbors:]
                        knn_B = None
                        if not evaluate_a_only and sim_B is not None:
                            np.fill_diagonal(sim_B, -1)
                            knn_B = np.sort(sim_B, axis=1)[:, -k_neighbors:]
                            del sim_A, sim_B  # 清理内存
                        else:
                            del sim_A  # 清理内存
                        gc.collect()

                    if knn_A is not None and (evaluate_a_only or knn_B is not None):
                        # 采样成对相似性
                        st.info(f"随机采样成对相似性 ({n_sample_pairs:,} 对)...")
                        pair_A = cached_pairwise_similarity(
                            fingerprints_A,
                            n_pairs=n_sample_pairs,
                            metric=similarity_metric,
                            seed=RANDOM_SEED + 101,
                            dataset_sig=dataset_sig_A
                        )
                        pair_B = None
                        if not evaluate_a_only:
                            pair_B = cached_pairwise_similarity(
                                fingerprints_B,
                                n_pairs=n_sample_pairs,
                                metric=similarity_metric,
                                seed=RANDOM_SEED + 202,
                                dataset_sig=dataset_sig_B
                            )

                        # 计算多样性指标
                        metrics_A = diversity_stats(knn_A, pair_A)
                        dataset_metrics = [("数据集A", metrics_A)]
                        if not evaluate_a_only and pair_B is not None:
                            metrics_B = diversity_stats(knn_B, pair_B)
                            dataset_metrics.append(("数据集B", metrics_B))

                        # 列表形式展示（每行一个数据集）
                        render_diversity_metrics_list(dataset_metrics)

                        # 最近邻分布分析
                        st.markdown("### 🎯 最近邻分布")
                        if evaluate_a_only:
                            fig = plot_nearest_neighbor_distribution(knn_sim=knn_A, title="Dataset A Nearest Neighbor Distribution")
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
                        else:
                            col1, col2 = st.columns(2)

                            with col1:
                                fig = plot_nearest_neighbor_distribution(knn_sim=knn_A, title="Dataset A Nearest Neighbor Distribution")
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)

                            with col2:
                                fig = plot_nearest_neighbor_distribution(knn_sim=knn_B, title="Dataset B Nearest Neighbor Distribution")
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)

                        if evaluate_a_only:
                            render_physchem_distribution_single(meta_A, random_seed=RANDOM_SEED)
                        else:
                            render_physchem_distribution_comparison(meta_A, meta_B, random_seed=RANDOM_SEED)
                        
                        # 优化聚类分析
                        st.markdown("### 🔍 优化聚类分析")
                        with st.spinner("执行优化聚类分析..."):
                            clustering_resultsA = perform_optimized_clustering_analysis(
                                fingerprints_A, 
                                cluster_method=cluster_method,
                                n_clusters=n_clusters,
                                eps=eps,
                                min_samples=min_samples,
                                use_minibatch=True,
                                random_seed=RANDOM_SEED
                            )
                            clustering_resultsB = None
                            if not evaluate_a_only:
                                clustering_resultsB = perform_optimized_clustering_analysis(
                                    fingerprints_B,
                                    cluster_method=cluster_method,
                                    n_clusters=n_clusters,
                                    eps=eps,
                                    min_samples=min_samples,
                                    use_minibatch=True,
                                    random_seed=RANDOM_SEED
                                )

                            if evaluate_a_only:
                                if clustering_resultsA:
                                    fig = plot_clustering_results(clustering_resultsA, "Dataset A Optimized Clustering", "PCA-UMAP")
                                    st.pyplot(fig)
                                    plt.close(fig)
                            elif clustering_resultsA and clustering_resultsB:
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig = plot_clustering_results(clustering_resultsA, "Dataset A Optimized Clustering", "PCA-UMAP")
                                    st.pyplot(fig)
                                    plt.close(fig)

                                with col2:
                                    fig = plot_clustering_results(clustering_resultsB, "Dataset B Optimized Clustering", "PCA-UMAP")
                                    st.pyplot(fig)
                                    plt.close(fig)

                        # 结构分布分析
                        st.markdown("### 📊 指纹空间分布分析")

                        if evaluate_a_only:
                            st.info(f"使用优化降维分析数据集A ({len(fingerprints_A):,} 个样本)...")
                            coords_A = embed_umap(
                                fingerprints_A,
                                n_pca=128,
                                n_components=2,
                                random_seed=RANDOM_SEED
                            )

                            if coords_A is not None:
                                st.markdown("**数据集A结构分布** (PCA-UMAP)")
                                center_A = np.mean(coords_A, axis=0)
                                dispersion_A = np.mean(np.linalg.norm(coords_A - center_A, axis=1))
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("样本数", f"{len(coords_A):,}")
                                with col2:
                                    st.metric("数据集A离散度", f"{dispersion_A:.3f}")

                                fig = plot_single_dataset_distribution(coords_A, dataset_name="Dataset A")
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                        else:
                            combined_fingerprints = np.vstack([fingerprints_A, fingerprints_B])
                            st.info(f"使用优化降维分析合并数据集 ({len(combined_fingerprints):,} 个样本)...")

                            coords = embed_umap(
                                combined_fingerprints,
                                n_pca=128,
                                n_components=2,
                                random_seed=RANDOM_SEED
                            )

                            if coords is not None:
                                # 分离坐标
                                coords_A = coords[:len(fingerprints_A)]
                                coords_B = coords[len(fingerprints_A):]

                                st.markdown("**结构分布对比** (PCA-UMAP)")
                                st.markdown("""
                                - 蓝色点：数据集A
                                - 橙色点：数据集B
                                """)

                                center_A = np.mean(coords_A, axis=0)
                                center_B = np.mean(coords_B, axis=0)
                                center_distance = np.linalg.norm(center_A - center_B)

                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("中心点距离", f"{center_distance:.3f}")
                                with col2:
                                    dispersion_A = np.mean(np.linalg.norm(coords_A - center_A, axis=1))
                                    st.metric("数据集A离散度", f"{dispersion_A:.3f}")
                                with col3:
                                    dispersion_B = np.mean(np.linalg.norm(coords_B - center_B, axis=1))
                                    st.metric("数据集B离散度", f"{dispersion_B:.3f}")

                                fig = plot_distribution_comparison(coords_A, coords_B)
                                st.pyplot(fig)
                                plt.close(fig)

                        st.success("🎉 优化模式指纹多样性评估完成")
                    else:
                        st.error("❌ 未能计算k-NN相似性，请检查指纹列或FAISS配置")
                        st.stop()
                
                else:
                    # ===========================================
                    # 兼容模式：完整相似性矩阵方法
                    # ===========================================
                    st.markdown("### 📊 兼容模式指纹相似性分析")
                    
                    st.info(f"使用 {similarity_metric} 相似性度量计算完整相似性矩阵...")
                    fingerprints_A, sample_idx_A = subsample_fingerprints(
                        fingerprints_A,
                        max_matrix_samples,
                        "数据集A",
                        random_seed=RANDOM_SEED
                    )
                    sample_idx_B = None
                    if not evaluate_a_only:
                        fingerprints_B, sample_idx_B = subsample_fingerprints(
                            fingerprints_B,
                            max_matrix_samples,
                            "数据集B",
                            random_seed=RANDOM_SEED
                        )
                    sim_matrixA = compute_similarity_matrix_from_fingerprints(
                        fingerprints_A,
                        similarity_metric,
                        confirm_key_suffix="dataset_a",
                        force_device=force_device if debug_mode else None,
                        tanimoto_n_jobs=tanimoto_n_jobs,
                        tanimoto_chunk_rows=tanimoto_chunk_rows
                    )
                    sim_matrixB = None
                    if not evaluate_a_only:
                        sim_matrixB = compute_similarity_matrix_from_fingerprints(
                            fingerprints_B,
                            similarity_metric,
                            confirm_key_suffix="dataset_b",
                            force_device=force_device if debug_mode else None,
                            tanimoto_n_jobs=tanimoto_n_jobs,
                            tanimoto_chunk_rows=tanimoto_chunk_rows
                        )
                    
                    if sim_matrixA is not None and (evaluate_a_only or sim_matrixB is not None):
                        if sample_idx_A is not None or (not evaluate_a_only and sample_idx_B is not None):
                            st.info("ℹ️ 已基于抽样子集计算兼容模式结果，可通过增大“最大相似性矩阵样本数”获取更多样本。")

                        metrics_A = calculate_diversity_metrics(sim_matrixA, random_seed=RANDOM_SEED)
                        dataset_metrics = [("数据集A", metrics_A)]
                        if not evaluate_a_only and sim_matrixB is not None:
                            metrics_B = calculate_diversity_metrics(sim_matrixB, random_seed=RANDOM_SEED)
                            dataset_metrics.append(("数据集B", metrics_B))

                        # 列表形式展示（每行一个数据集）
                        render_diversity_metrics_list(dataset_metrics)
                        
                        # 最近邻分布分析
                        st.markdown("### 🎯 最近邻分布")
                        if evaluate_a_only:
                            fig = plot_nearest_neighbor_distribution(sim_matrix=sim_matrixA, title="Dataset A Nearest Neighbor Distribution")
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
                        else:
                            col1, col2 = st.columns(2)

                            with col1:
                                fig = plot_nearest_neighbor_distribution(sim_matrix=sim_matrixA, title="Dataset A Nearest Neighbor Distribution")
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)

                            with col2:
                                fig = plot_nearest_neighbor_distribution(sim_matrix=sim_matrixB, title="Dataset B Nearest Neighbor Distribution")
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)

                        meta_A_for_distribution = subset_meta_with_indices(meta_A, sample_idx_A)
                        if evaluate_a_only:
                            render_physchem_distribution_single(
                                meta_A_for_distribution,
                                random_seed=RANDOM_SEED
                            )
                        else:
                            meta_B_for_distribution = subset_meta_with_indices(meta_B, sample_idx_B)
                            render_physchem_distribution_comparison(
                                meta_A_for_distribution,
                                meta_B_for_distribution,
                                random_seed=RANDOM_SEED
                            )
                        
                        # 聚类分析
                        st.markdown("### 🔍 聚类分析")
                        with st.spinner("执行聚类分析..."):
                            clustering_resultsA = perform_clustering_analysis(
                                sim_matrixA, 
                                cluster_method=cluster_method,
                                n_clusters=n_clusters,
                                eps=eps,
                                min_samples=min_samples,
                                method=dim_reduction_method,
                                perplexity=perplexity if dim_reduction_method == "t-SNE" else 30.0,
                                n_neighbors=n_neighbors if dim_reduction_method == "UMAP" else 15,
                                min_dist=min_dist if dim_reduction_method == "UMAP" else 0.1,
                                force_device=force_device if debug_mode else None,
                                random_seed=RANDOM_SEED
                            )
                            clustering_resultsB = None
                            if not evaluate_a_only:
                                clustering_resultsB = perform_clustering_analysis(
                                    sim_matrixB,
                                    cluster_method=cluster_method,
                                    n_clusters=n_clusters,
                                    eps=eps,
                                    min_samples=min_samples,
                                    method=dim_reduction_method,
                                    perplexity=perplexity if dim_reduction_method == "t-SNE" else 30.0,
                                    n_neighbors=n_neighbors if dim_reduction_method == "UMAP" else 15,
                                    min_dist=min_dist if dim_reduction_method == "UMAP" else 0.1,
                                    force_device=force_device if debug_mode else None,
                                    random_seed=RANDOM_SEED
                                )

                            if evaluate_a_only:
                                fig = plot_clustering_results(clustering_resultsA, "Dataset A Clustering Results", dim_reduction_method)
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig = plot_clustering_results(clustering_resultsA, "Dataset A Clustering Results", dim_reduction_method)
                                    st.pyplot(fig)
                                    plt.close(fig)

                                with col2:
                                    fig = plot_clustering_results(clustering_resultsB, "Dataset B Clustering Results", dim_reduction_method)
                                    st.pyplot(fig)
                                    plt.close(fig)
                        
                        # 结构分布分析
                        st.markdown("### 📊 指纹空间分布分析")
                        if evaluate_a_only:
                            st.info(f"使用{dim_reduction_method}对数据集A进行降维...")
                            coords_A = perform_dimensionality_reduction(
                                sim_matrixA,
                                method=dim_reduction_method,
                                perplexity=perplexity if dim_reduction_method == "t-SNE" else None,
                                n_neighbors=n_neighbors if dim_reduction_method == "UMAP" else None,
                                min_dist=min_dist if dim_reduction_method == "UMAP" else None,
                                force_device=force_device if debug_mode else None,
                                random_seed=RANDOM_SEED
                            )

                            if coords_A is not None:
                                st.markdown(f"**数据集A结构分布** ({dim_reduction_method})")
                                center_A = np.mean(coords_A, axis=0)
                                dispersion_A = np.mean(np.linalg.norm(coords_A - center_A, axis=1))
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("样本数", f"{len(coords_A):,}")
                                with col2:
                                    st.metric("数据集A离散度", f"{dispersion_A:.3f}")

                                fig = plot_single_dataset_distribution(coords_A, dataset_name="Dataset A")
                                if fig:
                                    st.pyplot(fig)
                                    plt.close(fig)
                        else:
                            # 合并数据进行分布比较
                            combined_fingerprints = np.vstack([fingerprints_A, fingerprints_B])
                            st.info(f"计算合并数据集的相似性矩阵 ({len(combined_fingerprints)} 个样本)...")
                            sim_matrix_combined = compute_similarity_matrix_from_fingerprints(
                                combined_fingerprints,
                                similarity_metric,
                                confirm_key_suffix="combined",
                                force_device=force_device if debug_mode else None,
                                tanimoto_n_jobs=tanimoto_n_jobs,
                                tanimoto_chunk_rows=tanimoto_chunk_rows
                            )

                            if sim_matrix_combined is not None:
                                st.info(f"使用{dim_reduction_method}进行降维...")
                                coords = perform_dimensionality_reduction(
                                    sim_matrix_combined,
                                    method=dim_reduction_method,
                                    perplexity=perplexity if dim_reduction_method == "t-SNE" else None,
                                    n_neighbors=n_neighbors if dim_reduction_method == "UMAP" else None,
                                    min_dist=min_dist if dim_reduction_method == "UMAP" else None,
                                    force_device=force_device if debug_mode else None,
                                    random_seed=RANDOM_SEED
                                )

                                if coords is not None:
                                    # 分离坐标
                                    coords_A = coords[:len(fingerprints_A)]
                                    coords_B = coords[len(fingerprints_A):]

                                    st.markdown(f"**结构分布对比** ({dim_reduction_method})")
                                    st.markdown("""
                                    - 蓝色点：原始完整数据集（数据集A）
                                    - 橙色点：被选中的子集（数据集B）
                                    """)

                                    # 简化的分布指标
                                    center_A = np.mean(coords_A, axis=0)
                                    center_B = np.mean(coords_B, axis=0)
                                    center_distance = np.linalg.norm(center_A - center_B)

                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("中心点距离", f"{center_distance:.3f}")
                                    with col2:
                                        dispersion_A = np.mean(np.linalg.norm(coords_A - center_A, axis=1))
                                        st.metric("数据集A离散度", f"{dispersion_A:.3f}")
                                    with col3:
                                        dispersion_B = np.mean(np.linalg.norm(coords_B - center_B, axis=1))
                                        st.metric("数据集B离散度", f"{dispersion_B:.3f}")

                                    fig = plot_distribution_comparison(coords_A, coords_B)
                                    st.pyplot(fig)
                                    plt.close(fig)
                
                # 显示最终结果和内存使用情况
                mem_usage_final = monitor_memory_usage()
                
                if analysis_mode == "优化模式 (推荐)":
                    st.success("✅ 优化模式指纹多样性分析完成！")
                    st.info("🎯 工作流：流式读取 → k-NN 采样 → PCA/UMAP 降维 → MiniBatch 聚类")
                else:
                    st.success("✅ 兼容模式指纹多样性分析完成！")
                
                # 显示GPU内存使用情况（如果可用）
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    gpu_mem_alloc = torch.cuda.memory_allocated() / 1024**2
                    gpu_mem_cached = torch.cuda.memory_reserved() / 1024**2
                    st.sidebar.success(
                        f"GPU内存状态:\n"
                        f"- 已分配: {gpu_mem_alloc:.1f}MB\n"
                        f"- 已缓存: {gpu_mem_cached:.1f}MB"
                    )
                
                # 显示最终内存使用
                st.sidebar.info(
                    f"最终内存使用:\n"
                    f"- RSS: {mem_usage_final['rss']:.1f} MB\n"
                    f"- 内存占用: {mem_usage_final['percent']:.1f}%"
                )
                
                # 清理内存
                if analysis_mode == "优化模式 (推荐)":
                    del fingerprints_A, fingerprints_B
                    if 'combined_fingerprints' in locals():
                        del combined_fingerprints
                    gc.collect()
                    if TORCH_AVAILABLE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
        else:
            st.error("无法加载指纹数据，请检查文件格式或指纹列是否为数值型")
            st.stop()
