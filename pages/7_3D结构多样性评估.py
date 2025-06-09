"""
CADD-Toolbox - 3D结构多样性评估页面
基于预计算的3D分子指纹进行结构多样性分析和比较
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import psutil
from datetime import datetime
import traceback
from scipy import stats
from scipy.stats import gaussian_kde

# 尝试导入GPU相关库
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    import cupy as cp
    from cuml.manifold import TSNE as cuTSNE
    from cuml.cluster import KMeans as cuKMeans, DBSCAN as cuDBSCAN
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from sklearn.decomposition import PCA

# 抑制警告
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
if TORCH_AVAILABLE:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# 设置全局随机种子以确保一致性
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

st.set_page_config(page_title="3D结构多样性评估", layout="wide")
st.title("🎯 3D结构多样性评估")

st.markdown("""
基于预计算的3D分子指纹进行结构多样性分析和比较。

🔬 **3D指纹**: E3FP、ROCS、Pharmacophore、USRCAT等  
📈 **聚类分析**: K-means、DBSCAN聚类  
📊 **可视化**: t-SNE、UMAP、PCA降维  
⚡ **加速**: GPU加速计算（可选）  

> **💡 使用说明：**
> - 请确保CSV文件包含预计算的3D指纹数据
> - 支持多种3D指纹格式（向量、位向量等）
> - 自动检测指纹列并计算相似性矩阵
""")

# 显示当前随机种子
st.info(f"🎲 当前随机种子: {RANDOM_SEED}")

if CUDA_AVAILABLE and TORCH_AVAILABLE and CUML_AVAILABLE:
    st.success("✅ GPU加速可用：CUDA + CuML")
elif CUDA_AVAILABLE and TORCH_AVAILABLE:
    st.warning("⚠️ GPU部分可用：CUDA可用但CuML不可用")
else:
    st.info("ℹ️ 仅CPU模式：GPU不可用")

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
    return [f for f in os.listdir(folder_path) if f.endswith('.csv')]

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

def initialize_cuda():
    """初始化CUDA设备并返回设备信息"""
    try:
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda" if cuda_available else "cpu")
        
        if cuda_available:
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1024**2
            gpu_mem_cached = torch.cuda.memory_reserved(0) / 1024**2
            
            st.sidebar.success("✅ CUDA可用，将使用GPU加速")
            st.sidebar.info(
                f"GPU信息:\n"
                f"- 设备: {gpu_name}\n"
                f"- 总显存: {gpu_mem_total:.1f}MB\n"
                f"- 已分配: {gpu_mem_alloc:.1f}MB\n"
                f"- 已缓存: {gpu_mem_cached:.1f}MB"
            )
        else:
            st.sidebar.info("ℹ️ CUDA不可用，将使用CPU计算")
        
        return cuda_available, device
    except Exception as e:
        st.sidebar.error(f"GPU初始化错误: {str(e)}")
        return False, torch.device("cpu")

def load_fingerprints_from_csv(file_path, fingerprint_cols=None):
    """从CSV文件加载预计算的3D指纹
    
    Args:
        file_path: CSV文件路径
        fingerprint_cols: 指纹列名列表，如果为None则自动检测
    
    Returns:
        fingerprints: 指纹数组
        df: 原始DataFrame
        fp_columns: 指纹列名列表
    """
    try:
        df = pd.read_csv(file_path)
        
        if fingerprint_cols is None:
            # 自动检测指纹列
            # 通常指纹列包含数值数据且列名可能包含特定关键词
            potential_fp_cols = []
            keywords = ['fingerprint', 'fp', 'descriptor', 'feature', 'bit', 'e3fp', 'rocs', 'usrcat']
            
            for col in df.columns:
                # 检查是否为数值列
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    potential_fp_cols.append(col)
                # 或者检查列名是否包含指纹相关关键词
                elif any(keyword.lower() in col.lower() for keyword in keywords):
                    try:
                        # 尝试转换为数值
                        pd.to_numeric(df[col], errors='raise')
                        potential_fp_cols.append(col)
                    except:
                        continue
            
            fingerprint_cols = potential_fp_cols
        
        if not fingerprint_cols:
            st.error("未找到指纹列，请手动指定指纹列")
            return None, None, None
        
        # 提取指纹数据
        fingerprints = df[fingerprint_cols].values
        
        # 检查和处理缺失值
        if np.isnan(fingerprints).any():
            st.warning("指纹数据中存在缺失值，将用0填充")
            fingerprints = np.nan_to_num(fingerprints)
        
        st.success(f"成功加载 {len(fingerprints)} 个样本，{len(fingerprint_cols)} 维指纹")
        
        return fingerprints, df, fingerprint_cols
        
    except Exception as e:
        st.error(f"读取指纹数据时出错: {str(e)}")
        return None, None, None

def compute_similarity_matrix_from_fingerprints(fingerprints, metric='cosine'):
    """从指纹数组计算相似性矩阵
    
    Args:
        fingerprints: 指纹数组 (n_samples, n_features)
        metric: 相似性度量方法
    
    Returns:
        similarity_matrix: 相似性矩阵
    """
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        n_samples = len(fingerprints)
        status_text.text(f"计算 {n_samples}x{n_samples} 相似性矩阵...")
        
        if metric == 'cosine':
            # 余弦相似性
            similarity_matrix = cosine_similarity(fingerprints)
        elif metric == 'euclidean':
            # 欧几里得距离转换为相似性
            distances = euclidean_distances(fingerprints)
            max_dist = np.max(distances)
            similarity_matrix = 1 - (distances / max_dist)
        elif metric == 'tanimoto':
            # Tanimoto相似性（适用于二进制指纹）
            similarity_matrix = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i, n_samples):
                    fp1, fp2 = fingerprints[i], fingerprints[j]
                    intersection = np.sum(np.minimum(fp1, fp2))
                    union = np.sum(np.maximum(fp1, fp2))
                    sim = intersection / union if union > 0 else 0
                    similarity_matrix[i, j] = similarity_matrix[j, i] = sim
                progress_bar.progress((i + 1) / n_samples)
        else:
            raise ValueError(f"不支持的相似性度量: {metric}")
        
        progress_bar.empty()
        status_text.empty()
        
        return similarity_matrix
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"计算相似性矩阵时出错: {str(e)}")
        return None

def calculate_diversity_metrics(sim_matrix):
    """计算多样性指标"""
    if sim_matrix is None:
        return {}
    
    # 排除对角线元素（自身相似性）
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    off_diagonal = sim_matrix[mask]
    
    return {
        'Mean Similarity': np.mean(off_diagonal),
        'Median Similarity': np.median(off_diagonal),
        'Similarity Std': np.std(off_diagonal),
        'Min Similarity': np.min(off_diagonal),
        'Max Similarity': np.max(off_diagonal),
        'Shannon Entropy': -np.sum(off_diagonal * np.log2(off_diagonal + 1e-10)) / len(off_diagonal)
    }

def plot_nearest_neighbor_distribution(sim_matrix, title="Nearest Neighbor Distribution"):
    """绘制最近邻分布"""
    if sim_matrix is None:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 计算每个分子的最近邻相似性（排除自身）
    np.fill_diagonal(sim_matrix, 0)  # 排除自身相似性
    nearest_neighbors = np.max(sim_matrix, axis=1)
    
    # 最近邻相似性分布
    ax1.hist(nearest_neighbors, bins=30, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Nearest Neighbor Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Nearest Neighbor Similarity Distribution')
    ax1.grid(True, alpha=0.3)
    
    # 累积分布
    sorted_nn = np.sort(nearest_neighbors)
    y = np.arange(1, len(sorted_nn) + 1) / len(sorted_nn)
    ax2.plot(sorted_nn, y, linewidth=2)
    ax2.set_xlabel('Nearest Neighbor Similarity')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def perform_clustering_analysis(sim_matrix, n_clusters=5, eps=0.3, min_samples=5, perplexity=30.0, force_device=None):
    """进行聚类分析（3D版本）"""
    np.random.seed(RANDOM_SEED)
    
    cuda_available, device = initialize_cuda()
    
    if force_device == "cpu":
        st.info("🔧 Debug模式：强制使用CPU计算")
        cuda_available = False
    elif force_device == "gpu":
        if not cuda_available:
            st.warning("🔧 Debug模式：强制GPU失败，回退到CPU")
        else:
            st.info("🔧 Debug模式：强制使用GPU计算")

    # 统一的距离矩阵计算
    dist_matrix = 1 - sim_matrix
    dist_matrix = np.clip(dist_matrix, 0, 2)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(dist_matrix, 0)

    if not cuda_available or force_device == "cpu":
        st.warning("⚠️ 使用CPU进行计算")
        
        # CPU版本的t-SNE
        tsne_cpu = TSNE(
            n_components=2, 
            metric='precomputed', 
            random_state=RANDOM_SEED,
            init='random', 
            learning_rate='auto',
            perplexity=min(perplexity, (len(dist_matrix) - 1) // 3),
            n_iter=1000,
            verbose=0
        )
        coords = tsne_cpu.fit_transform(dist_matrix)

        # CPU版本的K-means
        kmeans_cpu = KMeans(
            n_clusters=n_clusters, 
            random_state=RANDOM_SEED, 
            n_init=10,
            algorithm='lloyd',
            max_iter=300,
            tol=1e-4
        )
        clusters = kmeans_cpu.fit_predict(coords)

        # CPU版本的DBSCAN
        dbscan_cpu = DBSCAN(
            metric='precomputed', 
            eps=eps, 
            min_samples=min_samples,
            algorithm='auto',
            leaf_size=30
        )
        dbscan_clusters = dbscan_cpu.fit_predict(dist_matrix)

        return {
            'coords': coords,
            'kmeans_clusters': clusters,
            'dbscan_clusters': dbscan_clusters
        }
    else:
        import torch 
        import cupy as cp

        st.info("🚀 GPU计算路径")
        
        dist_matrix_gpu = cp.asarray(dist_matrix, dtype=cp.float32)

        # t-SNE使用CPU版本确保一致性
        st.info("🔧 t-SNE使用CPU版本确保一致性...")
        tsne_cpu = TSNE(
            n_components=2,
            perplexity=min(perplexity, (len(dist_matrix) - 1) // 3),
            random_state=RANDOM_SEED,
            metric='precomputed',
            init='random',
            learning_rate='auto',
            n_iter=1000,
            verbose=0
        )
        coords = tsne_cpu.fit_transform(dist_matrix)

        # GPU版本的K-means
        st.info("🚀 使用GPU K-means...")
        try:
            cp.random.seed(RANDOM_SEED)
            
            kmeans_gpu = cuKMeans(
                n_clusters=n_clusters, 
                random_state=RANDOM_SEED, 
                n_init=10,
                max_iter=300,
                tol=1e-4
            )
            coords_gpu = cp.asarray(coords, dtype=cp.float32)
            clusters_gpu = kmeans_gpu.fit_predict(coords_gpu)
            clusters = cp.asnumpy(clusters_gpu)
        except Exception as e:
            st.warning(f"GPU K-means失败，回退到CPU: {str(e)}")
            kmeans_cpu = KMeans(
                n_clusters=n_clusters, 
                random_state=RANDOM_SEED, 
                n_init=10,
                algorithm='lloyd',
                max_iter=300,
                tol=1e-4
            )
            clusters = kmeans_cpu.fit_predict(coords)

        # GPU版本的DBSCAN
        st.info("🚀 使用GPU DBSCAN...")
        try:
            dbscan_gpu = cuDBSCAN(
                metric='precomputed', 
                eps=eps, 
                min_samples=min_samples
            )
            dbscan_clusters_gpu = dbscan_gpu.fit_predict(dist_matrix_gpu)
            dbscan_clusters = cp.asnumpy(dbscan_clusters_gpu)
        except Exception as e:
            st.warning(f"GPU DBSCAN失败，回退到CPU: {str(e)}")
            dbscan_cpu = DBSCAN(
                metric='precomputed', 
                eps=eps, 
                min_samples=min_samples,
                algorithm='auto',
                leaf_size=30
            )
            dbscan_clusters = dbscan_cpu.fit_predict(dist_matrix)

        if torch.cuda.is_available():
            gpu_mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
            gpu_mem_cached = torch.cuda.memory_reserved(device) / 1024**2
            st.success(
                f"✅ GPU加速聚类分析完成:\n"
                f"- GPU内存使用: {gpu_mem_alloc:.1f}MB\n"
                f"- GPU缓存: {gpu_mem_cached:.1f}MB"
            )

        return {
            'coords': coords,
            'kmeans_clusters': clusters,
            'dbscan_clusters': dbscan_clusters
        }

def plot_clustering_results(clustering_results, title="3D Clustering Results"):
    """绘制聚类结果"""
    coords = clustering_results['coords']
    kmeans_clusters = clustering_results['kmeans_clusters']
    dbscan_clusters = clustering_results['dbscan_clusters']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # K-means结果
    scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], c=kmeans_clusters, cmap='tab10', alpha=0.7)
    ax1.set_title("K-means Clustering")
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter1, ax=ax1, label='Cluster ID')
    
    # DBSCAN结果
    scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], c=dbscan_clusters, cmap='tab10', alpha=0.7)
    ax2.set_title("DBSCAN Clustering")
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter2, ax=ax2, label='Cluster ID')
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig

def perform_dimensionality_reduction(similarity_matrix, method="t-SNE", perplexity=30, n_neighbors=15, min_dist=0.1, force_device=None):
    """执行降维操作（3D版本）"""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    np.random.seed(RANDOM_SEED)
    
    debug_info = st.empty()
    start_time = time.time()
    
    cuda_available, device = initialize_cuda()
    
    if force_device == "cpu":
        st.info("🔧 Debug模式：降维强制使用CPU")
        cuda_available = False
    elif force_device == "gpu":
        if not cuda_available:
            st.warning("🔧 Debug模式：降维强制GPU失败，回退到CPU")
        else:
            st.info("🔧 Debug模式：降维强制使用GPU")
    
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, 2)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)
    
    try:
        if method == "t-SNE":
            debug_info.info("🔧 使用CPU版本t-SNE确保结果一致性...")
            
            effective_perplexity = min(perplexity, (len(distance_matrix) - 1) // 3)
            
            tsne = TSNE(
                n_components=2,
                perplexity=effective_perplexity,
                random_state=RANDOM_SEED,
                metric='precomputed',
                init='random',
                learning_rate='auto',
                n_iter=1000,
                verbose=0
            )
            coords = tsne.fit_transform(distance_matrix)
        
        elif method == "UMAP":
            debug_info.info("🔧 使用UMAP进行降维...")
            if UMAP_AVAILABLE:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric='precomputed',
                    random_state=RANDOM_SEED
                )
                coords = reducer.fit_transform(distance_matrix)
            else:
                st.error("UMAP库未安装，请安装umap-learn")
                return None
        
        elif method == "PCA":
            debug_info.info("🔧 使用PCA进行降维...")
            pca = PCA(n_components=2, random_state=RANDOM_SEED)
            coords = pca.fit_transform(similarity_matrix)
        
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        if coords.shape[1] != 2:
            raise ValueError(f"降维结果维度不正确: {coords.shape}")
        
        debug_info.success(f"✅ 降维完成 ({time.time() - start_time:.2f}秒)")
        return coords
    
    except Exception as e:
        debug_info.error(f"降维过程出错: {str(e)}")
        st.error(f"降维失败: {str(e)}")
        return None
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def plot_distribution_comparison(coords_A, coords_B, metrics):
    """绘制分布对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 散点图
    scatter1 = ax1.scatter(coords_A[:, 0], coords_A[:, 1], c='blue', alpha=0.6, s=30, label='Dataset A')
    scatter2 = ax1.scatter(coords_B[:, 0], coords_B[:, 1], c='orange', alpha=0.6, s=30, label='Dataset B')
    ax1.set_title('3D Structure Distribution Scatter Plot')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.legend()
    
    # 密度等高线图
    x = np.concatenate([coords_A[:, 0], coords_B[:, 0]])
    y = np.concatenate([coords_A[:, 1], coords_B[:, 1]])
    
    if len(coords_A) > 1 and len(coords_B) > 1:
        xmin, xmax = x.min() - 1, x.max() + 1
        ymin, ymax = y.min() - 1, y.max() + 1
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        
        values_A = np.vstack([coords_A[:, 0], coords_A[:, 1]])
        values_B = np.vstack([coords_B[:, 0], coords_B[:, 1]])
        
        try:
            kernel_A = gaussian_kde(values_A)
            kernel_B = gaussian_kde(values_B)
            
            z_A = np.reshape(kernel_A(positions), xx.shape)
            z_B = np.reshape(kernel_B(positions), xx.shape)
            
            ax2.contour(xx, yy, z_A, levels=5, colors='blue', alpha=0.5, label='Dataset A')
            ax2.contour(xx, yy, z_B, levels=5, colors='orange', alpha=0.5, label='Dataset B')
            ax2.legend()
        except:
            ax2.text(0.5, 0.5, 'Density plot not available\n(insufficient data)', 
                    ha='center', va='center', transform=ax2.transAxes)
    
    ax2.set_title('3D Structure Density Contour Plot')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    
    plt.tight_layout()
    return fig

def monitor_memory_usage():
    """监控内存使用"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024**2,
        'vms': memory_info.vms / 1024**2,
        'percent': process.memory_percent()
    }

# ===========================================
# 主程序界面
# ===========================================

# 文件选择界面
st.subheader("1. 选择包含3D指纹的数据文件")

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
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**数据集A (原始数据集)**")
        selected_fileA = st.selectbox("选择数据集A的CSV文件:", csv_files, key="fileA")
        
    with col2:
        st.markdown("**数据集B (筛选后数据集)**")
        selected_fileB = st.selectbox("选择数据集B的CSV文件:", csv_files, key="fileB")
    
    if selected_fileA and selected_fileB:
        fileA_path = os.path.join(DATA_DIR, selected_folder, selected_fileA)
        fileB_path = os.path.join(DATA_DIR, selected_folder, selected_fileB)
        
        # 显示文件信息
        col1, col2 = st.columns(2)
        
        with col1:
            file_infoA = get_file_info(fileA_path)
            if file_infoA:
                st.info(f"**文件A信息:**\n- 大小: {file_infoA['size_mb']:.1f} MB\n- 修改时间: {file_infoA['modified']}")
        
        with col2:
            file_infoB = get_file_info(fileB_path)
            if file_infoB:
                st.info(f"**文件B信息:**\n- 大小: {file_infoB['size_mb']:.1f} MB\n- 修改时间: {file_infoB['modified']}")

# 参数设置
st.subheader("2. 分析参数设置")

with st.expander("📊 相似性和聚类参数", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        similarity_metric = st.selectbox(
            "相似性度量方法",
            ["cosine", "euclidean", "tanimoto"],
            help="cosine: 余弦相似性; euclidean: 欧几里得距离; tanimoto: Tanimoto系数"
        )
        
        n_clusters = st.slider(
            "K-means聚类数",
            min_value=2,
            max_value=20,
            value=5,
            help="K-means聚类的簇数"
        )
    
    with col2:
        eps = st.slider(
            "DBSCAN eps参数",
            min_value=0.1,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="DBSCAN聚类的邻域半径"
        )
        
        min_samples = st.slider(
            "DBSCAN最小样本数",
            min_value=2,
            max_value=20,
            value=5,
            help="DBSCAN聚类的最小样本数"
        )
    
    with col3:
        dim_reduction_method = st.selectbox(
            "降维方法",
            ["t-SNE", "UMAP", "PCA"],
            help="选择降维可视化方法"
        )
        
        if dim_reduction_method == "t-SNE":
            perplexity = st.slider(
                "t-SNE困惑度",
                min_value=5,
                max_value=50,
                value=30,
                help="推荐值：5-50，影响局部vs全局结构"
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

# Debug模式设置
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

force_device = None
if debug_mode:
    force_device = st.selectbox(
        "强制使用设备",
        ["auto", "cpu", "gpu"],
        help="auto: 自动选择最优设备; cpu: 强制使用CPU; gpu: 强制使用GPU(如果可用)"
    )

# 开始分析
st.subheader("3. 分析结果")

results_container = st.container()
with results_container:
    if 'selected_fileA' not in locals() or 'selected_fileB' not in locals():
        st.info("请先选择要分析的CSV文件，然后点击开始评估按钮")
    else:
        st.markdown("**准备分析的文件:**")
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"数据集A: {selected_fileA}")
        with col2:
            st.text(f"数据集B: {selected_fileB}")

if st.button("🚀 开始3D结构多样性分析", type="primary") and 'selected_fileA' in locals() and 'selected_fileB' in locals():
    with st.spinner("正在进行3D结构多样性分析..."):
        # 显示内存使用情况
        mem_usage = monitor_memory_usage()
        st.sidebar.info(
            f"内存使用情况:\n"
            f"- RSS: {mem_usage['rss']:.1f} MB\n"
            f"- 内存占用: {mem_usage['percent']:.1f}%"
        )
        
        # 加载3D指纹数据
        fingerprints_A, df_A, fp_cols_A = load_fingerprints_from_csv(fileA_path)
        fingerprints_B, df_B, fp_cols_B = load_fingerprints_from_csv(fileB_path)
        
        if fingerprints_A is not None and fingerprints_B is not None:
            results_container.empty()
            
            with results_container:
                st.success(f"✅ 成功加载: 数据集A {len(fingerprints_A)}个样本，数据集B {len(fingerprints_B)}个样本")
                
                # 显示基本统计信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("数据集A样本数", len(fingerprints_A))
                with col2:
                    st.metric("数据集B样本数", len(fingerprints_B))
                with col3:
                    st.metric("指纹维度", len(fp_cols_A))
                with col4:
                    st.metric("选择比例", f"{len(fingerprints_B)/len(fingerprints_A)*100:.1f}%")
                
                # 计算相似性矩阵
                st.markdown("### 📊 3D指纹相似性分析")
                
                st.info(f"使用 {similarity_metric} 相似性度量计算相似性矩阵...")
                sim_matrixA = compute_similarity_matrix_from_fingerprints(fingerprints_A, similarity_metric)
                sim_matrixB = compute_similarity_matrix_from_fingerprints(fingerprints_B, similarity_metric)
                
                if sim_matrixA is not None and sim_matrixB is not None:
                    # 显示多样性指标
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**数据集A多样性指标**")
                        metrics_A = calculate_diversity_metrics(sim_matrixA)
                        for key, value in metrics_A.items():
                            st.metric(key, f"{value:.4f}")
                    
                    with col2:
                        st.markdown("**数据集B多样性指标**")
                        metrics_B = calculate_diversity_metrics(sim_matrixB)
                        for key, value in metrics_B.items():
                            st.metric(key, f"{value:.4f}")
                    
                    # 最近邻分布分析
                    st.markdown("### 🎯 最近邻分布")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = plot_nearest_neighbor_distribution(sim_matrixA, "Dataset A Nearest Neighbor Distribution")
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    with col2:
                        fig = plot_nearest_neighbor_distribution(sim_matrixB, "Dataset B Nearest Neighbor Distribution")
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # 聚类分析
                    st.markdown("### 🔍 聚类分析")
                    with st.spinner("执行聚类分析..."):
                        clustering_resultsA = perform_clustering_analysis(
                            sim_matrixA, 
                            n_clusters=n_clusters,
                            eps=eps,
                            min_samples=min_samples,
                            perplexity=perplexity if dim_reduction_method == "t-SNE" else 30.0,
                            force_device=force_device if debug_mode else None
                        )
                        clustering_resultsB = perform_clustering_analysis(
                            sim_matrixB, 
                            n_clusters=n_clusters,
                            eps=eps,
                            min_samples=min_samples,
                            perplexity=perplexity if dim_reduction_method == "t-SNE" else 30.0,
                            force_device=force_device if debug_mode else None
                        )
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            fig = plot_clustering_results(clustering_resultsA, "Dataset A 3D Clustering Results")
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        with col2:
                            fig = plot_clustering_results(clustering_resultsB, "Dataset B 3D Clustering Results")
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    # 结构分布分析
                    st.markdown("### 📊 3D结构分布分析")
                    
                    # 合并数据进行分布比较
                    combined_fingerprints = np.vstack([fingerprints_A, fingerprints_B])
                    st.info(f"计算合并数据集的相似性矩阵 ({len(combined_fingerprints)} 个样本)...")
                    sim_matrix_combined = compute_similarity_matrix_from_fingerprints(combined_fingerprints, similarity_metric)
                    
                    if sim_matrix_combined is not None:
                        st.info(f"使用{dim_reduction_method}进行降维...")
                        coords = perform_dimensionality_reduction(
                            sim_matrix_combined,
                            method=dim_reduction_method,
                            perplexity=perplexity if dim_reduction_method == "t-SNE" else None,
                            n_neighbors=n_neighbors if dim_reduction_method == "UMAP" else None,
                            min_dist=min_dist if dim_reduction_method == "UMAP" else None,
                            force_device=force_device if debug_mode else None
                        )
                        
                        if coords is not None:
                            # 分离坐标
                            coords_A = coords[:len(fingerprints_A)]
                            coords_B = coords[len(fingerprints_A):]
                            
                            st.markdown(f"**3D结构分布对比** ({dim_reduction_method})")
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
                            
                            fig = plot_distribution_comparison(coords_A, coords_B, {})
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # 显示GPU内存使用情况（如果可用）
                            if torch.cuda.is_available():
                                gpu_mem_alloc = torch.cuda.memory_allocated() / 1024**2
                                gpu_mem_cached = torch.cuda.memory_reserved() / 1024**2
                                st.sidebar.success(
                                    f"GPU内存状态:\n"
                                    f"- 已分配: {gpu_mem_alloc:.1f}MB\n"
                                    f"- 已缓存: {gpu_mem_cached:.1f}MB"
                                )
                            
                            st.success("✅ 3D结构多样性分析完成！")
                
        else:
            st.error("无法加载指纹数据，请检查文件格式") 