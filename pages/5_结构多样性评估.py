"""
CADD-Toolbox - 结构多样性评估页面
基于数值化指纹进行结构多样性分析与比较
"""

import os
import sys
import math
import gc
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, IncrementalPCA
import psutil
from datetime import datetime
import traceback
from scipy import stats
from scipy.stats import gaussian_kde

# 尝试导入FAISS
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
    st.warning("⚠️ FAISS未安装，将使用sklearn进行相似性计算（较慢）")

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

st.set_page_config(page_title="结构多样性评估", layout="wide")
st.title("📊 结构多样性评估（指纹数据）")

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

def read_fps(csv_path: str, chunksize: int = 200_000, fp_dtype="float16", 
             fingerprint_cols=None) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """流式加载指纹列，峰值内存≈常数级
    
    Args:
        csv_path: CSV文件路径
        chunksize: 分块大小，控制峰值内存
        fp_dtype: 指纹数据类型，float16可节省内存
        fingerprint_cols: 指纹列名列表，如果为None则自动检测
    
    Returns:
        fps: 指纹数组 (N, D)
        fp_columns: 指纹列名列表  
        meta: 元数据DataFrame
    """
    try:
        # 先读取小样本检测列类型
        sample = pd.read_csv(csv_path, nrows=100, low_memory=True)
        
        if fingerprint_cols is None:
            # 自动检测数值型指纹列
            keywords = ['fingerprint', 'fp', 'descriptor', 'feature', 'bit', 'e3fp', 'rocs', 'usrcat']
            num_cols = []
            
            for col in sample.columns:
                # 检查是否为数值列
                if sample[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    num_cols.append(col)
                # 或者检查列名是否包含指纹相关关键词
                elif any(keyword.lower() in col.lower() for keyword in keywords):
                    try:
                        pd.to_numeric(sample[col], errors='raise')
                        num_cols.append(col)
                    except:
                        continue
            fingerprint_cols = num_cols
        
        if not fingerprint_cols:
            raise ValueError("未检测到数值型指纹列")
        
        # 非数值列作为元数据
        meta_cols = [col for col in sample.columns if col not in fingerprint_cols]
        
        st.info(f"检测到 {len(fingerprint_cols)} 个指纹列，{len(meta_cols)} 个元数据列")
        
        # 流式读取指纹数据
        fp_parts = []
        total_rows = 0
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 先尝试读取第一个chunk来验证数据类型兼容性
        first_chunk_iter = pd.read_csv(
            csv_path,
            usecols=fingerprint_cols,
            chunksize=chunksize,
            low_memory=True,
            engine="c"
        )
        
        # 读取第一个chunk进行数据类型验证
        first_chunk = next(first_chunk_iter)
        
        # 测试数据类型转换
        test_chunk = first_chunk.copy()
        test_chunk.replace([np.inf, -np.inf], 0, inplace=True)
        test_chunk.fillna(0, inplace=True)
        
        # 检查哪些列可以安全转换为目标类型
        safe_cols = []
        problematic_cols = []
        
        for col in fingerprint_cols:
            try:
                # 测试转换
                test_values = pd.to_numeric(test_chunk[col], errors='coerce').astype(fp_dtype)
                safe_cols.append(col)
            except (ValueError, OverflowError) as e:
                problematic_cols.append(col)
                st.warning(f"列 '{col}' 无法转换为 {fp_dtype}，将使用 float32")
        
        if problematic_cols:
            st.info(f"检测到 {len(problematic_cols)} 个列需要使用 float32 而非 {fp_dtype}")
            # 为有问题的列使用float32
            actual_dtype = 'float32'
        else:
            actual_dtype = fp_dtype
        
        # 首先处理已经读取的第一个chunk
        try:
            # 处理第一个chunk
            chunk = first_chunk.copy()
            status_text.text(f"处理第 1 个数据块，当前行数: {total_rows}")
            
            # 处理异常值和缺失值
            chunk.replace([np.inf, -np.inf], 0, inplace=True)
            chunk.fillna(0, inplace=True)
            
            # 安全的数据类型转换
            for col in chunk.columns:
                try:
                    chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
                except:
                    pass
            
            # 转换为numpy
            chunk_array = chunk.to_numpy(dtype=actual_dtype, copy=False)
            
            # 检查数组有效性
            if np.isnan(chunk_array).any() or np.isinf(chunk_array).any():
                chunk_array = np.nan_to_num(chunk_array, nan=0.0, posinf=0.0, neginf=0.0)
            
            fp_parts.append(chunk_array)
            total_rows += len(chunk_array)
            del chunk
            gc.collect()
            
        except Exception as e:
            st.warning(f"处理第 1 个数据块时出错: {str(e)}，跳过此块")
        
        # 继续读取剩余的chunks
        chunk_iter = pd.read_csv(
            csv_path,
            usecols=fingerprint_cols,
            chunksize=chunksize,
            low_memory=True,
            engine="c"
        )
        
        # 跳过第一个chunk（已经处理过了）
        try:
            next(chunk_iter)
        except StopIteration:
            # 文件只有一个chunk，已经处理完毕
            pass
        
        for i, chunk in enumerate(chunk_iter, start=1):
            status_text.text(f"读取第 {i+1} 个数据块，当前行数: {total_rows}")
            
            try:
                # 处理异常值和缺失值
                chunk.replace([np.inf, -np.inf], 0, inplace=True)
                chunk.fillna(0, inplace=True)
                
                # 安全的数据类型转换
                for col in chunk.columns:
                    try:
                        chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
                    except:
                        pass
                
                # 转换为numpy
                chunk_array = chunk.to_numpy(dtype=actual_dtype, copy=False)
                
                # 检查数组有效性
                if np.isnan(chunk_array).any() or np.isinf(chunk_array).any():
                    chunk_array = np.nan_to_num(chunk_array, nan=0.0, posinf=0.0, neginf=0.0)
                
                fp_parts.append(chunk_array)
                total_rows += len(chunk_array)
                
            except Exception as e:
                st.warning(f"处理第 {i+1} 个数据块时出错: {str(e)}，跳过此块")
                continue
            
            del chunk
            gc.collect()
            
            # 更新进度（估算）
            if i < 5:  # 前几个块用于估算
                progress_bar.progress(min(0.9, (i + 1) * 0.15))
        
        progress_bar.progress(0.95)
        status_text.text("合并数据块...")
        
        # 合并所有分块
        fps = np.concatenate(fp_parts, axis=0)
        del fp_parts
        gc.collect()
        
        progress_bar.progress(1.0)
        status_text.text("读取元数据...")
        
        # 读取元数据（如果有）
        if meta_cols:
            meta = pd.read_csv(csv_path, usecols=meta_cols, low_memory=True)
        else:
            meta = pd.DataFrame(index=np.arange(fps.shape[0]))
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ 流式加载完成: {len(fps):,} 个样本，{len(fingerprint_cols)} 维指纹 ({actual_dtype})")
        st.info(f"📊 内存使用: {fps.nbytes / 1024**2:.1f} MB")
        
        if actual_dtype != fp_dtype:
            st.info(f"💡 数据类型已从 {fp_dtype} 调整为 {actual_dtype} 以确保兼容性")
        
        return fps, fingerprint_cols, meta
        
    except Exception as e:
        st.error(f"流式读取指纹数据时出错: {str(e)}")
        return None, None, None

def subsample_fingerprints(fps: np.ndarray, max_samples: int, label: str) -> tuple[np.ndarray, np.ndarray | None]:
    """若样本数超过阈值，则随机下采样以避免完整相似性矩阵造成内存瓶颈"""
    if max_samples <= 0 or len(fps) <= max_samples:
        return fps, None

    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.sort(rng.choice(len(fps), max_samples, replace=False))
    st.warning(
        f"{label} 含 {len(fps):,} 个样本，已随机抽样 {max_samples:,} 个用于兼容模式计算。"
        " 如需完整矩阵，请降低数据集规模或切换优化模式。"
    )
    return fps[indices], indices


def subsample_by_ratio(fps: np.ndarray, ratio: float, label: str, meta: pd.DataFrame | None = None) -> tuple[np.ndarray, pd.DataFrame | None, np.ndarray | None]:
    """按照指定比例随机抽样指纹数据。ratio=1.0 表示不采样。"""
    if fps is None or len(fps) == 0 or ratio >= 0.999:
        return fps, meta, None

    rng = np.random.default_rng(RANDOM_SEED)
    sample_size = max(1, int(len(fps) * ratio))
    sample_size = min(sample_size, len(fps))
    indices = np.sort(rng.choice(len(fps), sample_size, replace=False))
    st.info(f"{label} 已按 {ratio*100:.1f}% 比例抽样 {sample_size:,}/{len(fps):,} 个样本")

    fps_sub = fps[indices]
    if meta is not None and len(meta) == len(fps):
        meta = meta.iloc[indices].reset_index(drop=True)
    return fps_sub, meta, indices


# 兼容性函数
def load_fingerprints_from_csv(file_path, fingerprint_cols=None):
    """兼容性函数，内部调用优化版read_fps"""
    fps, fp_cols, meta = read_fps(file_path, fingerprint_cols=fingerprint_cols)
    if fps is not None:
        return fps, meta, fp_cols
    return None, None, None

def ensure_faiss_compatible(arr: np.ndarray) -> np.ndarray:
    """确保数组与FAISS兼容（C-contiguous + float32）"""
    if not arr.flags['C_CONTIGUOUS'] or arr.dtype != np.float32:
        return np.ascontiguousarray(arr.astype(np.float32))
    return arr

def knn_similarity(fps: np.ndarray, metric: str = "cosine", 
                   k: int = 30, use_gpu: bool = True) -> np.ndarray:
    """使用FAISS计算k最近邻相似度（不含自身）
    
    Args:
        fps: 指纹数组 (N, D)
        metric: 相似性度量方法
        k: 最近邻数量
        use_gpu: 是否使用GPU
    
    Returns:
        knn_sim: shape=(N, k) 的最近邻相似度矩阵
    """
    try:
        if not FAISS_AVAILABLE:
            # 回退到sklearn
            st.warning("FAISS不可用，使用sklearn计算k-NN（较慢）")
            from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
            
            if metric == "cosine":
                sim_matrix = cosine_similarity(fps)
            else:
                distances = euclidean_distances(fps)
                max_dist = np.max(distances)
                sim_matrix = 1 - (distances / max_dist)
            
            # 排除自身，取top-k
            np.fill_diagonal(sim_matrix, -1)  # 排除自身
            knn_indices = np.argsort(sim_matrix, axis=1)[:, -k:]
            knn_sim = np.array([sim_matrix[i, knn_indices[i]] for i in range(len(fps))])
            return knn_sim
        
        # 使用FAISS
        import faiss
        
        # 确保数组与FAISS兼容
        fps_copy = ensure_faiss_compatible(fps)
        
        if metric == "cosine":
            # 余弦相似性需要L2归一化
            faiss.normalize_L2(fps_copy)
            if use_gpu and faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(fps_copy.shape[1]))
            else:
                index = faiss.IndexFlatIP(fps_copy.shape[1])
        else:  # euclidean
            if use_gpu and faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(fps_copy.shape[1]))
            else:
                index = faiss.IndexFlatL2(fps_copy.shape[1])
        
        index.add(fps_copy)
        sim, _ = index.search(fps_copy, k + 1)  # 包含自身
        
        if metric == "euclidean":
            # L2距离转换为相似性
            max_dist = sim.max()
            sim = 1 - (sim / max_dist)
        
        return sim[:, 1:]  # 去掉自身（第一列）
        
    except Exception as e:
        st.error(f"k-NN计算出错: {str(e)}")
        return None

def sample_pairwise(fps: np.ndarray, n_pairs: int = 2_000_000, 
                   metric="cosine", rng=None) -> np.ndarray:
    """随机采样计算成对相似性
    
    Args:
        fps: 指纹数组
        n_pairs: 采样对数
        metric: 相似性度量
        rng: 随机数生成器
    
    Returns:
        pair_sim: 采样的成对相似性数组
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED)
    
    n_samples = len(fps)
    # 确保不超过总对数
    max_pairs = n_samples * (n_samples - 1) // 2
    n_pairs = min(n_pairs, max_pairs)
    
    # 随机采样索引对
    idx1 = rng.integers(0, n_samples, n_pairs, dtype=np.int64)
    idx2 = rng.integers(0, n_samples, n_pairs, dtype=np.int64)
    
    # 避免自身相比
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    
    if len(idx1) == 0:
        return np.array([])
    
    fps1, fps2 = fps[idx1], fps[idx2]
    
    if metric == "cosine":
        # 余弦相似性
        norms1 = np.linalg.norm(fps1, axis=1)
        norms2 = np.linalg.norm(fps2, axis=1)
        dot_products = (fps1 * fps2).sum(axis=1)
        
        # 避免除零
        norm_products = norms1 * norms2
        valid_mask = norm_products > 1e-10
        similarities = np.zeros(len(fps1))
        similarities[valid_mask] = dot_products[valid_mask] / norm_products[valid_mask]
        
        return similarities
    else:  # euclidean
        distances = np.linalg.norm(fps1 - fps2, axis=1)
        max_dist = distances.max() if len(distances) > 0 else 1.0
        return 1 - (distances / max_dist)

def compute_similarity_matrix_from_fingerprints(fingerprints, metric='cosine'):
    """兼容性函数：针对小数据集计算完整相似性矩阵"""
    n_samples = len(fingerprints)
    
    # 大数据集警告并推荐使用优化方法
    if n_samples > 50_000:
        st.warning(f"⚠️ 数据集较大({n_samples:,}个样本)，建议使用k-NN + 采样方法")
        if st.button("继续使用完整矩阵计算"):
            pass
        else:
            return None
    
    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text(f"计算 {n_samples}x{n_samples} 相似性矩阵...")
        
        if metric == 'cosine':
            similarity_matrix = cosine_similarity(fingerprints)
        elif metric == 'euclidean':
            distances = euclidean_distances(fingerprints)
            max_dist = np.max(distances)
            similarity_matrix = 1 - (distances / max_dist)
        elif metric == 'tanimoto':
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

def diversity_stats(knn_sim: np.ndarray, pair_sim: np.ndarray) -> dict[str, float]:
    """基于k-NN和采样相似性计算多样性指标
    
    Args:
        knn_sim: k最近邻相似性矩阵 (N, k)
        pair_sim: 随机采样的成对相似性数组
    
    Returns:
        stats: 多样性统计指标字典
    """
    if knn_sim is None or len(knn_sim) == 0:
        return {}
    
    # 最近邻统计
    nn_max = knn_sim.max(axis=1)  # 每个样本的最近邻相似性
    nn_mean = knn_sim.mean(axis=1)  # 每个样本的平均k-NN相似性
    
    stats = {
        # 最近邻指标
        'NN_Mean': nn_mean.mean(),
        'NN_Median': np.median(nn_mean),
        'NN_Std': nn_mean.std(),
        'NN_Min': nn_max.min(),
        'NN_Max': nn_max.max(),
        'NN_Q25': np.percentile(nn_max, 25),
        'NN_Q75': np.percentile(nn_max, 75),
    }
    
    # 如果有成对采样数据，添加全局统计
    if pair_sim is not None and len(pair_sim) > 0:
        # 过滤有效值
        valid_pairs = pair_sim[~np.isnan(pair_sim)]
        
        if len(valid_pairs) > 0:
            stats.update({
                'Pair_Mean': valid_pairs.mean(),
                'Pair_Median': np.median(valid_pairs),
                'Pair_Std': valid_pairs.std(),
                'Pair_Min': valid_pairs.min(),
                'Pair_Max': valid_pairs.max(),
            })
            
            # Shannon熵（基于直方图）
            try:
                hist, _ = np.histogram(valid_pairs, bins=100, range=(0, 1), density=True)
                p = hist / (hist.sum() + 1e-12)
                p = p[p > 1e-12]  # 避免log(0)
                shannon_entropy = -(p * np.log2(p)).sum()
                stats['Shannon_Entropy'] = shannon_entropy
            except:
                stats['Shannon_Entropy'] = 0.0
    
    return stats

def calculate_diversity_metrics(sim_matrix):
    """兼容性函数：从完整相似性矩阵计算多样性指标"""
    if sim_matrix is None:
        return {}
    
    # 排除对角线元素（自身相似性）
    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    off_diagonal = sim_matrix[mask]
    
    # 模拟k-NN数据用于新函数
    np.fill_diagonal(sim_matrix, -1)  # 排除自身
    knn_sim = np.sort(sim_matrix, axis=1)[:, -30:]  # top-30 NN
    
    # 采样成对相似性
    n_pairs = min(1_000_000, len(off_diagonal))
    rng = np.random.default_rng(RANDOM_SEED)
    sampled_pairs = rng.choice(off_diagonal, size=n_pairs, replace=False)
    
    return diversity_stats(knn_sim, sampled_pairs)

def embed_umap(fps: np.ndarray, n_pca=128, n_components=2) -> np.ndarray:
    """先PCA降维再UMAP嵌入的优化降维方法

    Args:
        fps: 指纹数组 (N, D)
        n_pca: PCA中间维度
        n_components: 最终输出维度
    
    Returns:
        embedding: 低维嵌入 (N, n_components)
    """
    try:
        if CUML_AVAILABLE and CUDA_AVAILABLE:
            st.info(f"🔧 使用 GPU PCA({n_pca}D) + UMAP({n_components}D) 降维...")
            import cupy as cp
            from cuml.decomposition import PCA as cuPCA
            from cuml.manifold import UMAP as cuUMAP

            n_pca_gpu = min(n_pca, fps.shape[1], fps.shape[0] - 1)
            fps_gpu = cp.asarray(fps.astype(np.float32), order='C')
            pca_gpu = cuPCA(n_components=n_pca_gpu, random_state=RANDOM_SEED)
            X_pca_gpu = pca_gpu.fit_transform(fps_gpu)
            reducer_gpu = cuUMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                random_state=RANDOM_SEED
            )
            embedding_gpu = reducer_gpu.fit_transform(X_pca_gpu)
            st.success(f"✅ GPU UMAP完成: {n_pca_gpu}D → {embedding_gpu.shape[1]}D")
            return cp.asnumpy(embedding_gpu)

        st.info(f"🔧 使用 IncrementalPCA({n_pca}D) + UMAP({n_components}D) 降维...")

        n_pca_cpu = min(n_pca, fps.shape[1], fps.shape[0] - 1)
        ipca = IncrementalPCA(n_components=n_pca_cpu, batch_size=min(10_000, fps.shape[0]))

        batch_size = min(10_000, fps.shape[0])
        for i in range(0, len(fps), batch_size):
            batch = fps[i:i + batch_size]
            ipca.partial_fit(batch)

        X_pca = ipca.transform(fps)
        st.info(f"✅ PCA完成: {fps.shape[1]}D → {X_pca.shape[1]}D")

        if not UMAP_AVAILABLE:
            st.warning("UMAP不可用，使用PCA降维到2D")
            if X_pca.shape[1] > n_components:
                final_pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
                return final_pca.fit_transform(X_pca)
            return X_pca[:, :n_components] if X_pca.shape[1] >= n_components else X_pca

        import umap
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=RANDOM_SEED,
            n_jobs=1
        )

        embedding = reducer.fit_transform(X_pca)
        st.success(f"✅ UMAP完成: {X_pca.shape[1]}D → {embedding.shape[1]}D")
        return embedding

    except Exception as e:
        st.error(f"降维失败: {str(e)}")
        st.info("回退到标准PCA降维...")
        try:
            pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            return pca.fit_transform(fps)
        except Exception as e2:
            st.error(f"PCA回退也失败: {str(e2)}")
            return None

def plot_nearest_neighbor_distribution(sim_matrix=None, knn_sim=None, title="Nearest Neighbor Distribution"):
    """绘制最近邻分布（支持完整矩阵或k-NN数据）"""
    if sim_matrix is not None:
        # 完整矩阵版本
        np.fill_diagonal(sim_matrix, 0)
        nearest_neighbors = np.max(sim_matrix, axis=1)
    elif knn_sim is not None:
        # k-NN版本
        nearest_neighbors = np.max(knn_sim, axis=1)
    else:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
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

def perform_optimized_clustering_analysis(fps: np.ndarray, n_clusters=5, eps=0.3, 
                                        min_samples=5, use_minibatch=True, force_device=None):
    """优化的聚类分析（直接基于指纹数据）"""
    np.random.seed(RANDOM_SEED)
    
    st.info("🔧 使用优化降维 + MiniBatch聚类...")
    
    # 使用优化降维
    coords = embed_umap(fps, n_pca=128, n_components=2)
    if coords is None:
        return None
    
    coords_gpu = None
    kmeans_clusters = None

    if CUML_AVAILABLE and CUDA_AVAILABLE:
        try:
            import cupy as cp
            from cuml.cluster import KMeans as cuKMeans

            coords_gpu = cp.asarray(coords.astype(np.float32))
            st.info("🚀 使用 cuML KMeans 聚类...")
            kmeans_gpu = cuKMeans(
                n_clusters=n_clusters,
                random_state=RANDOM_SEED,
                max_iter=300,
                tol=1e-4,
            )
            kmeans_clusters = cp.asnumpy(kmeans_gpu.fit_predict(coords_gpu))
        except Exception as exc:
            st.warning(f"cuML KMeans 聚类失败，回退到CPU: {exc}")
            coords_gpu = None

    if kmeans_clusters is None:
        if use_minibatch and len(fps) > 10_000:
            st.info("🚀 使用MiniBatchKMeans进行聚类...")
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(10_000, len(fps) // 10),
                random_state=RANDOM_SEED,
                n_init=3,
                max_iter=100
            )
        else:
            st.info("🔧 使用标准KMeans进行聚类...")
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=RANDOM_SEED,
                n_init=10,
                max_iter=300
            )

        kmeans_clusters = kmeans.fit_predict(coords)

    # DBSCAN基于降维结果
    dbscan_clusters = None
    if coords_gpu is not None:
        try:
            from cuml.cluster import DBSCAN as cuDBSCAN

            st.info("🚀 使用 cuML DBSCAN 聚类...")
            dbscan_gpu = cuDBSCAN(eps=eps, min_samples=min_samples)
            dbscan_clusters = cp.asnumpy(dbscan_gpu.fit_predict(coords_gpu))
        except Exception as exc:
            st.warning(f"cuML DBSCAN 聚类失败，回退到CPU: {exc}")

    if dbscan_clusters is None:
        st.info("🔧 使用DBSCAN进行密度聚类...")
        try:
            from sklearn.cluster import HDBSCAN
            dbscan = HDBSCAN(
                min_cluster_size=min_samples,
                min_samples=min_samples,
                metric='euclidean'
            )
            dbscan_clusters = dbscan.fit_predict(coords)
            st.info("✅ 使用HDBSCAN聚类")
        except ImportError:
            dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
                metric='euclidean'
            )
            dbscan_clusters = dbscan.fit_predict(coords)
            st.info("✅ 使用标准DBSCAN聚类")

    n_noise = np.sum(dbscan_clusters == -1)
    n_clusters_found = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
    st.info(f"📊 DBSCAN结果: {n_clusters_found}个簇, {n_noise}个噪声点")

    return {
        'coords': coords,
        'kmeans_clusters': kmeans_clusters,
        'dbscan_clusters': dbscan_clusters
    }

def perform_clustering_analysis(sim_matrix, n_clusters=5, eps=0.3, min_samples=5, perplexity=30.0, force_device=None):
    """兼容性函数：基于相似性矩阵的聚类分析"""
    if sim_matrix is None:
        return None
        
    n_samples = len(sim_matrix)
    
    # 大数据集推荐使用优化版本
    if n_samples > 50_000:
        st.warning("⚠️ 大数据集建议使用优化版聚类方法 (perform_optimized_clustering_analysis)")
    
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

        # 使用MiniBatch K-means
        if len(sim_matrix) > 10_000:
            kmeans_cpu = MiniBatchKMeans(
                n_clusters=n_clusters, 
                random_state=RANDOM_SEED, 
                batch_size=min(5000, len(sim_matrix) // 10),
                n_init=3,
                max_iter=100
            )
        else:
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
        # GPU版本保持不变...
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
            # 使用MiniBatch回退
            if len(sim_matrix) > 10_000:
                kmeans_cpu = MiniBatchKMeans(
                    n_clusters=n_clusters, 
                    random_state=RANDOM_SEED, 
                    batch_size=min(5000, len(sim_matrix) // 10)
                )
            else:
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

def plot_clustering_results(clustering_results, title="Clustering Results", method="PCA-UMAP"):
    """绘制聚类结果"""
    coords = clustering_results['coords']
    kmeans_clusters = clustering_results['kmeans_clusters']
    dbscan_clusters = clustering_results['dbscan_clusters']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 根据降维方法设置轴标签
    if "UMAP" in method:
        xlabel, ylabel = f"{method} Dimension 1", f"{method} Dimension 2"
    elif "t-SNE" in method:
        xlabel, ylabel = "t-SNE Dimension 1", "t-SNE Dimension 2"
    elif "PCA" in method:
        xlabel, ylabel = "PCA Dimension 1", "PCA Dimension 2"
    else:
        xlabel, ylabel = "Dimension 1", "Dimension 2"
    
    # K-means结果
    scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], c=kmeans_clusters, cmap='tab10', alpha=0.7)
    ax1.set_title("K-means Clustering")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    plt.colorbar(scatter1, ax=ax1, label='Cluster ID')
    
    # DBSCAN结果
    scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], c=dbscan_clusters, cmap='tab10', alpha=0.7)
    ax2.set_title("DBSCAN Clustering")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel)
    plt.colorbar(scatter2, ax=ax2, label='Cluster ID')
    
    if hasattr(fig, 'canvas'):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                fig.canvas.draw_idle()
        except Exception:
            pass
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig

def perform_dimensionality_reduction(similarity_matrix, method="t-SNE", perplexity=30, n_neighbors=15, min_dist=0.1, force_device=None):
    """执行降维操作（兼容模式专用）"""
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

def plot_distribution_comparison(coords_A, coords_B, metrics=None):
    """绘制分布对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 散点图
    scatter1 = ax1.scatter(coords_A[:, 0], coords_A[:, 1], c='blue', alpha=0.6, s=30, label='Dataset A')
    scatter2 = ax1.scatter(coords_B[:, 0], coords_B[:, 1], c='orange', alpha=0.6, s=30, label='Dataset B')
    ax1.set_title('Structure Distribution Scatter Plot')
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

            contour_a = ax2.contour(xx, yy, z_A, levels=5, colors='blue', alpha=0.5)
            contour_b = ax2.contour(xx, yy, z_B, levels=5, colors='orange', alpha=0.5)
            handles = [contour_a.collections[0], contour_b.collections[0]]
            labels = ['Dataset A', 'Dataset B']
            ax2.legend(handles, labels)
        except:
            ax2.text(0.5, 0.5, 'Density plot not available\n(insufficient data)', 
                    ha='center', va='center', transform=ax2.transAxes)

    ax2.set_title('Structure Density Contour Plot')
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
st.subheader("1. 选择包含指纹的数据文件")

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

# 添加分析模式选择
sample_ratio = 1.0
with st.expander("⚡ 分析模式选择", expanded=True):
    analysis_mode = st.selectbox(
        "选择分析模式",
        ["优化模式 (推荐)", "兼容模式"],
        help="优化模式：使用k-NN + 采样 + PCA-UMAP，适合大数据集；兼容模式：完整相似性矩阵，适合小数据集"
    )
    
    if analysis_mode == "优化模式 (推荐)":
        st.success("✅ 使用优化算法：流式读取 + k-NN相似性 + PCA-UMAP降维 + MiniBatch聚类")
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
            help="兼容模式下用于绘制聚类/分布图；优化模式固定为 IncrementalPCA→UMAP"
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

if st.button("🚀 开始指纹多样性分析", type="primary") and 'selected_fileA' in locals() and 'selected_fileB' in locals():
    with st.spinner("正在进行指纹多样性分析..."):
        # 显示内存使用情况
        mem_usage = monitor_memory_usage()
        st.sidebar.info(
            f"内存使用情况:\n"
            f"- RSS: {mem_usage['rss']:.1f} MB\n"
            f"- 内存占用: {mem_usage['percent']:.1f}%"
        )
        
        if analysis_mode == "优化模式 (推荐)":
            # 使用优化的流式加载
            st.info("🚀 使用优化模式进行分析...")
            fingerprints_A, fp_cols_A, meta_A = read_fps(fileA_path, fp_dtype=fp_dtype)
            fingerprints_B, fp_cols_B, meta_B = read_fps(fileB_path, fp_dtype=fp_dtype)
            fingerprints_A, meta_A, ratio_idx_A = subsample_by_ratio(fingerprints_A, sample_ratio, "数据集A", meta_A)
            fingerprints_B, meta_B, ratio_idx_B = subsample_by_ratio(fingerprints_B, sample_ratio, "数据集B", meta_B)
        else:
            # 使用兼容模式
            st.info("⚡ 使用兼容模式进行分析...")
            fingerprints_A, meta_A, fp_cols_A = load_fingerprints_from_csv(fileA_path)
            fingerprints_B, meta_B, fp_cols_B = load_fingerprints_from_csv(fileB_path)
            ratio_idx_A = ratio_idx_B = None
        
        if fingerprints_A is not None and fingerprints_B is not None:
            results_container.empty()
            
            with results_container:
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
                            # 确保数组与FAISS兼容
                            fingerprints_A = ensure_faiss_compatible(fingerprints_A)
                            fingerprints_B = ensure_faiss_compatible(fingerprints_B)
                            faiss.normalize_L2(fingerprints_A)
                            faiss.normalize_L2(fingerprints_B)
                        else:
                            fingerprints_A = fingerprints_A / (np.linalg.norm(fingerprints_A, axis=1, keepdims=True) + 1e-10)
                            fingerprints_B = fingerprints_B / (np.linalg.norm(fingerprints_B, axis=1, keepdims=True) + 1e-10)
                    
                    # 计算k-NN相似性
                    st.info(f"计算k-NN相似性 (k={k_neighbors})...")
                    try:
                        knn_A = knn_similarity(fingerprints_A, similarity_metric, k_neighbors, use_gpu=True)
                        knn_B = knn_similarity(fingerprints_B, similarity_metric, k_neighbors, use_gpu=True)
                    except Exception as e:
                        st.error(f"k-NN计算出错: {str(e)}")
                        st.info("回退到sklearn计算...")
                        # 回退到sklearn方法
                        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
                        if similarity_metric == "cosine":
                            sim_A = cosine_similarity(fingerprints_A)
                            sim_B = cosine_similarity(fingerprints_B)
                        else:
                            dist_A = euclidean_distances(fingerprints_A)
                            dist_B = euclidean_distances(fingerprints_B)
                            sim_A = 1 - (dist_A / dist_A.max())
                            sim_B = 1 - (dist_B / dist_B.max())
                        
                        # 提取k-NN
                        np.fill_diagonal(sim_A, -1)
                        np.fill_diagonal(sim_B, -1)
                        knn_A = np.sort(sim_A, axis=1)[:, -k_neighbors:]
                        knn_B = np.sort(sim_B, axis=1)[:, -k_neighbors:]
                        del sim_A, sim_B  # 清理内存
                        gc.collect()
                    
                    if knn_A is not None and knn_B is not None:
                        # 采样成对相似性
                        st.info(f"随机采样成对相似性 ({n_sample_pairs:,} 对)...")
                        rng = np.random.default_rng(RANDOM_SEED)
                        pair_A = sample_pairwise(fingerprints_A, n_sample_pairs, similarity_metric, rng)
                        pair_B = sample_pairwise(fingerprints_B, n_sample_pairs, similarity_metric, rng)
                        
                        # 计算多样性指标
                        metrics_A = diversity_stats(knn_A, pair_A)
                        metrics_B = diversity_stats(knn_B, pair_B)
                        
                        # 显示多样性指标
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**数据集A多样性指标**")
                            for key, value in metrics_A.items():
                                st.metric(key, f"{value:.4f}")
                        
                        with col2:
                            st.markdown("**数据集B多样性指标**")
                            for key, value in metrics_B.items():
                                st.metric(key, f"{value:.4f}")
                        
                        # 最近邻分布分析
                        st.markdown("### 🎯 最近邻分布")
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
                        
                        # 优化聚类分析
                        st.markdown("### 🔍 优化聚类分析")
                        with st.spinner("执行优化聚类分析..."):
                            clustering_resultsA = perform_optimized_clustering_analysis(
                                fingerprints_A, 
                                n_clusters=n_clusters,
                                eps=eps,
                                min_samples=min_samples,
                                use_minibatch=True
                            )
                            clustering_resultsB = perform_optimized_clustering_analysis(
                                fingerprints_B, 
                                n_clusters=n_clusters,
                                eps=eps,
                                min_samples=min_samples,
                                use_minibatch=True
                            )
                            
                            if clustering_resultsA and clustering_resultsB:
                                col1, col2 = st.columns(2)
                                with col1:
                                    fig = plot_clustering_results(clustering_resultsA, "Dataset A Optimized Clustering", "PCA-UMAP")
                                    st.pyplot(fig)
                                    plt.close(fig)
                                
                                with col2:
                                    fig = plot_clustering_results(clustering_resultsB, "Dataset B Optimized Clustering", "PCA-UMAP")
                                    st.pyplot(fig)
                                    plt.close(fig)
                        
                        # 结构分布分析（合并数据集）
                        st.markdown("### 📊 指纹空间分布分析")
                        
                        combined_fingerprints = np.vstack([fingerprints_A, fingerprints_B])
                        st.info(f"使用优化降维分析合并数据集 ({len(combined_fingerprints):,} 个样本)...")
                        
                        coords = embed_umap(combined_fingerprints, n_pca=128, n_components=2)
                        
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
                    fingerprints_A, sample_idx_A = subsample_fingerprints(fingerprints_A, max_matrix_samples, "数据集A")
                    fingerprints_B, sample_idx_B = subsample_fingerprints(fingerprints_B, max_matrix_samples, "数据集B")
                    sim_matrixA = compute_similarity_matrix_from_fingerprints(fingerprints_A, similarity_metric)
                    sim_matrixB = compute_similarity_matrix_from_fingerprints(fingerprints_B, similarity_metric)
                    
                    if sim_matrixA is not None and sim_matrixB is not None:
                        if sample_idx_A is not None or sample_idx_B is not None:
                            st.info("ℹ️ 已基于抽样子集计算兼容模式结果，可通过增大“最大相似性矩阵样本数”获取更多样本。")
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
                            fig = plot_nearest_neighbor_distribution(sim_matrix=sim_matrixA, title="Dataset A Nearest Neighbor Distribution")
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
                        
                        with col2:
                            fig = plot_nearest_neighbor_distribution(sim_matrix=sim_matrixB, title="Dataset B Nearest Neighbor Distribution")
                            if fig:
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
                                fig = plot_clustering_results(clustering_resultsA, "Dataset A Clustering Results", dim_reduction_method)
                                st.pyplot(fig)
                                plt.close(fig)
                            
                            with col2:
                                fig = plot_clustering_results(clustering_resultsB, "Dataset B Clustering Results", dim_reduction_method)
                                st.pyplot(fig)
                                plt.close(fig)
                        
                        # 结构分布分析
                        st.markdown("### 📊 指纹空间分布分析")
                        
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
