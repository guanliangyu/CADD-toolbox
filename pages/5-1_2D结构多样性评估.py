"""
CADD-Toolbox - 2D结构多样性评估页面
基于分子指纹进行结构多样性分析和比较
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
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

st.set_page_config(page_title="2D结构多样性评估", layout="wide")
st.title("📊 2D结构多样性评估")

st.markdown("""
基于分子指纹进行结构多样性分析，支持多种指纹类型和聚类算法。

🔬 **指纹类型**: Morgan、MACCS、拓扑、原子对指纹  
📈 **聚类分析**: K-means、DBSCAN聚类  
📊 **可视化**: t-SNE、UMAP、PCA降维  
⚡ **加速**: GPU加速计算（可选）  

> **💡 获得一致CPU/GPU结果的关键因素：**
> - 使用相同的随机种子
> - 确保算法参数完全一致
> - 处理数值精度差异
> - 统一的数据预处理方式
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
    """初始化CUDA设备并返回设备信息 (简化版)"""
    try:
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda" if cuda_available else "cpu")
        
        if cuda_available:
            # 清理CUDA缓存
            torch.cuda.empty_cache()
            
            # 获取GPU信息
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

# 工具函数
def load_molecules_from_csv(file_path, smiles_col="SMILES"):
    """从CSV文件加载分子
    
    Args:
        file_path: CSV文件路径（字符串）或上传的文件对象
        smiles_col: SMILES列名
    
    Returns:
        mols: RDKit分子对象列表
        df: 有效分子对应的DataFrame
    """
    try:
        # 处理文件路径或文件对象
        if isinstance(file_path, str):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path)
        
        if smiles_col not in df.columns:
            st.error(f"未找到SMILES列 '{smiles_col}'")
            st.info(f"可用列: {', '.join(df.columns.tolist())}")
            return None, None
        
        mols = []
        valid_indices = []
        invalid_count = 0
        
        for i, row in df.iterrows():
            try:
                smi = str(row[smiles_col]).strip()
                if smi and smi.lower() not in ['nan', 'none', '']:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        mols.append(mol)
                        valid_indices.append(i)
                    else:
                        invalid_count += 1
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1
                continue
        
        if invalid_count > 0:
            st.warning(f"跳过了 {invalid_count} 个无效的SMILES")
        
        return mols, df.iloc[valid_indices]
    except Exception as e:
        st.error(f"读取CSV文件时出错: {str(e)}")
        return None, None

def compute_fingerprint(mol, fp_type="morgan", radius=2, nBits=2048, use_features=False, **kwargs):
    """计算分子指纹
    
    Args:
        mol: RDKit分子对象
        fp_type: 指纹类型，可选:
            - "morgan": Morgan/ECFP指纹
            - "fcfp": Morgan特征指纹
            - "maccs": MACCS结构键
            - "topological": 拓扑指纹
            - "atom_pairs": 原子对指纹
            - "torsion": 扭转指纹
            - "layered": 分层指纹
        radius: Morgan指纹的半径
        nBits: 指纹位数
        use_features: 是否使用原子特征（用于FCFP）
        **kwargs: 其他参数
    """
    if fp_type == "morgan":
        # 使用新的MorganGenerator API
        morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits, useFeatures=use_features)
        return morgan_gen.GetFingerprint(mol)
    elif fp_type == "fcfp":
        # 使用新的MorganGenerator API，设置useFeatures=True
        morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits, useFeatures=True)
        return morgan_gen.GetFingerprint(mol)
    elif fp_type == "maccs":
        return AllChem.GetMACCSKeysFingerprint(mol)
    elif fp_type == "topological":
        # 使用新的RDKitFPGenerator API
        rdk_gen = GetRDKitFPGenerator(fpSize=nBits, **kwargs)
        return rdk_gen.GetFingerprint(mol)
    elif fp_type == "atom_pairs":
        return AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits)
    elif fp_type == "torsion":
        return AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits)
    elif fp_type == "layered":
        return Chem.LayeredFingerprint(mol, layerFlags=kwargs.get('layerFlags', 0xFFFFFFFF))
    else:
        raise ValueError(f"不支持的指纹类型: {fp_type}")

@st.cache_data
def compute_fingerprints_batch(smiles_list, fp_type="morgan", radius=2, nBits=2048, use_features=False, **kwargs):
    """批量计算分子指纹"""
    import torch
    
    # 创建进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    debug_info = st.empty()
    
    # 初始化CUDA
    cuda_available, device = initialize_cuda()
    start_time = time.time()
    
    # 创建指纹生成器
    if fp_type == "morgan":
        fp_gen = GetMorganGenerator(radius=radius, fpSize=nBits, useFeatures=use_features)
    elif fp_type == "fcfp":
        fp_gen = GetMorganGenerator(radius=radius, fpSize=nBits, useFeatures=True)
    elif fp_type == "topological":
        fp_gen = GetRDKitFPGenerator(fpSize=nBits, **kwargs)
    
    fps = []
    total_mols = len(smiles_list)
    
    # 计算指纹
    for i, smi in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                fps.append(np.zeros(nBits))
            else:
                if fp_type in ["morgan", "fcfp"]:
                    fp = np.array(fp_gen.GetFingerprint(mol).ToList())
                elif fp_type == "maccs":
                    fp = np.array(AllChem.GetMACCSKeysFingerprint(mol).ToList())
                elif fp_type == "topological":
                    fp = np.array(fp_gen.GetFingerprint(mol).ToList())
                elif fp_type == "atom_pairs":
                    fp = np.array(AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nBits).ToList())
                elif fp_type == "torsion":
                    fp = np.array(AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nBits).ToList())
                elif fp_type == "layered":
                    fp = np.array(Chem.LayeredFingerprint(mol, layerFlags=kwargs.get('layerFlags', 0xFFFFFFFF)).ToList())
                fps.append(fp)
        except Exception as e:
            st.warning(f"处理分子 {smi} 时出错: {str(e)}")
            fps.append(np.zeros(nBits))
        
        # 更新进度
        progress = (i + 1) / total_mols
        progress_bar.progress(progress)
        if (i + 1) % 100 == 0:
            status_text.text(f"已处理: {i + 1}/{total_mols} 个分子")
    
    fps_array = np.array(fps)
    
    # 如果有GPU，将数据转移到GPU
    if cuda_available:
        fps_tensor = torch.tensor(fps_array, dtype=torch.float32).to(device)
        debug_info.info(f"数据已转移到GPU，显存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
    
    # 显示性能统计
    total_time = time.time() - start_time
    avg_speed = total_mols / total_time
    debug_info.success(
        f"✅ 指纹计算完成:\n"
        f"- 总计算时间: {total_time:.1f} 秒\n"
        f"- 平均速度: {avg_speed:.1f} 分子/秒\n"
        f"- 计算设备: {'GPU' if cuda_available else 'CPU'}"
    )
    
    # 清除进度显示
    progress_bar.empty()
    status_text.empty()
    
    return fps_array

def compute_similarity_matrix(fps_list, progress_text="计算相似性矩阵"):
    """使用GPU加速计算相似性矩阵"""
    import torch
    debug_info = st.empty()
    start_time = time.time()
    
    # 初始化CUDA
    cuda_available, device = initialize_cuda()
    
    # 显示计算设备信息
    device_info = "GPU (CUDA)" if cuda_available else "CPU"
    debug_info.info(f"💻 使用 {device_info} {progress_text}")
    
    # 将指纹转换为二进制数组
    n_fps = len(fps_list)
    if n_fps == 0:
        return np.array([])
    
    # 获取指纹长度
    fp_length = len(fps_list[0].ToBitString())
    
    # 创建二进制矩阵
    fp_array = np.zeros((n_fps, fp_length), dtype=np.float32)
    for i, fp in enumerate(fps_list):
        fp_array[i] = np.array(list(fp.ToBitString())).astype(np.float32)
    
    if cuda_available:
        # 转换为PyTorch张量并移至GPU
        fp_tensor = torch.from_numpy(fp_array).to(device)
        
        # 计算点积
        dot_product = torch.mm(fp_tensor, fp_tensor.t())
        
        # 计算每个指纹的1的数量
        fp_sums = torch.sum(fp_tensor, dim=1, keepdim=True)
        
        # 计算并集
        union = fp_sums + fp_sums.t() - dot_product
        
        # 计算Tanimoto相似度
        similarity_matrix = (dot_product / union).cpu().numpy()
        
        # 显示GPU内存使用情况
        gpu_mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
        gpu_mem_cached = torch.cuda.memory_reserved(device) / 1024**2
        debug_info.info(
            f"GPU内存使用情况:\n"
            f"- 已分配: {gpu_mem_alloc:.1f} MB\n"
            f"- 缓存: {gpu_mem_cached:.1f} MB"
        )
    else:
        # CPU计算
        similarity_matrix = np.zeros((n_fps, n_fps))
        for i in range(n_fps):
            for j in range(i, n_fps):
                sim = DataStructs.TanimotoSimilarity(fps_list[i], fps_list[j])
                similarity_matrix[i,j] = sim
                similarity_matrix[j,i] = sim
    
    # 计算性能统计
    total_time = time.time() - start_time
    comparisons = (n_fps * (n_fps - 1)) / 2
    speed = comparisons / total_time if total_time > 0 else 0
    
    debug_info.success(
        f"✅ 相似性矩阵计算完成:\n"
        f"- 矩阵大小: {n_fps}x{n_fps}\n"
        f"- 比较次数: {comparisons:,.0f}\n"
        f"- 计算时间: {total_time:.1f} 秒\n"
        f"- 计算速度: {speed:,.0f} 比较/秒\n"
        f"- 计算设备: {device_info}"
    )
    
    return similarity_matrix

@st.cache_data(show_spinner=False)
def process_molecules_parallel(smiles_list, radius=2, nBits=2048):
    """处理分子并计算相似性矩阵"""
    # 初始化CUDA
    cuda_available, _ = initialize_cuda()
    
    st.info(
        f"🚀 开始处理:\n"
        f"- 分子数量: {len(smiles_list)}\n"
        f"- 指纹半径: {radius}\n"
        f"- 指纹位数: {nBits}\n"
        f"- GPU加速: {'可用' if cuda_available else '不可用'}"
    )
    
    # 计算分子指纹
    fps = compute_fingerprints_batch(smiles_list, radius=radius, nBits=nBits)
    
    # 计算相似性矩阵
    sim_matrix = compute_similarity_matrix(fps)
    
    return fps, sim_matrix

def calculate_diversity_metrics(sim_matrix):
    """计算多样性指标"""
    # 获取上三角矩阵的值（不包括对角线）
    triu_indices = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[triu_indices]
    
    metrics = {
        "Mean Similarity": np.mean(similarities),
        "Similarity Std": np.std(similarities),
        "Max Similarity": np.max(similarities),
        "Min Similarity": np.min(similarities),
        "Median Similarity": np.median(similarities)
    }
    
    return metrics

def get_bemis_murcko_scaffold(mol):
    """获取Bemis-Murcko骨架"""
    core = MurckoScaffold.GetScaffoldForMol(mol)
    if core:
        return Chem.MolToSmiles(core)
    return None

def calc_scaffold_stats(mols):
    """计算骨架统计信息"""
    scaff_dict = {}
    for mol in mols:
        scaf_smi = get_bemis_murcko_scaffold(mol)
        if scaf_smi:
            scaff_dict[scaf_smi] = scaff_dict.get(scaf_smi, 0) + 1
    
    if not scaff_dict:
        return 0, 0.0, 0.0, pd.DataFrame([])
    
    items = sorted(scaff_dict.items(), key=lambda x: x[1], reverse=True)
    scaffolds_freq_df = pd.DataFrame(items, columns=["Scaffold_SMILES", "Count"])
    n_scaffolds = len(items)
    n_mols = sum([count for _, count in items])
    
    scaffold_entropy = 0.0
    for _, c in items:
        p = c / n_mols
        scaffold_entropy -= p * math.log2(p)
    
    singletons = sum([1 for _, c in items if c == 1])
    fraction_singletons = singletons / n_scaffolds
    
    return n_scaffolds, scaffold_entropy, fraction_singletons, scaffolds_freq_df

def calc_F50(scaffolds_freq_df):
    """计算F50值"""
    if scaffolds_freq_df.empty:
        return 0.0
    total_mols = scaffolds_freq_df["Count"].sum()
    half_mols = total_mols * 0.5
    
    cover = 0
    used_scafs = 0
    for i, row in scaffolds_freq_df.iterrows():
        cover += row["Count"]
        used_scafs += 1
        if cover >= half_mols:
            break
    
    total_scaf = len(scaffolds_freq_df)
    f50 = used_scafs / total_scaf
    return f50

def plot_property_comparison(propsA, propsB, prop_name, prop_label):
    """Plot property distribution comparison"""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(data=propsA, x=prop_name, label="Library A", ax=ax, color='#2ecc71')
    sns.kdeplot(data=propsB, x=prop_name, label="Library B", ax=ax, color='#e74c3c')
    ax.set_title(f"Distribution of {prop_label}")
    ax.set_xlabel(prop_label)
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    return fig

def calc_property_stats(mols):
    """Calculate molecular properties"""
    records = []
    for mol in mols:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol)
        records.append((mw, logp, tpsa, hbd, hba, rotatable))
    
    columns = {
        "MW": "Molecular Weight",
        "LogP": "LogP",
        "TPSA": "TPSA",
        "HBD": "H-Bond Donors",
        "HBA": "H-Bond Acceptors",
        "RotBonds": "Rotatable Bonds"
    }
    
    df = pd.DataFrame(records, columns=list(columns.keys()))
    df.columns.name = "Property"
    return df, columns

def calculate_shannon_entropy(fp_array):
    """计算指纹比特的香农熵"""
    # 计算每个比特位的出现频率
    bit_frequencies = np.mean(fp_array, axis=0)
    # 计算香农熵
    entropy = 0
    for freq in bit_frequencies:
        if freq > 0 and freq < 1:  # 避免log(0)
            entropy -= freq * np.log2(freq) + (1-freq) * np.log2(1-freq)
    return entropy

def calculate_mean_nearest_neighbor(sim_matrix):
    """计算平均最近邻Tanimoto相似度"""
    # 对每个分子，找到与其最相似的其他分子（不包括自身）
    n = sim_matrix.shape[0]
    nearest_neighbors = []
    for i in range(n):
        # 创建掩码排除自身
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        # 找到最大相似度
        max_sim = np.max(sim_matrix[i][mask])
        nearest_neighbors.append(max_sim)
    return np.mean(nearest_neighbors)

def plot_similarity_distribution(sim_matrix):
    """绘制相似度分布直方图"""
    # 获取上三角矩阵的值（不包括对角线）
    triu_indices = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[triu_indices]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=similarities, bins=50, kde=True, ax=ax)
    ax.set_title("Distribution of Pairwise Tanimoto Similarities")
    ax.set_xlabel("Tanimoto Similarity")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig

def plot_similarity_threshold_stats(sim_matrix):
    """绘制相似度阈值统计图"""
    triu_indices = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[triu_indices]
    
    thresholds = np.arange(0, 1.1, 0.1)
    percentages = [np.mean(similarities >= t) * 100 for t in thresholds]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, percentages, marker='o')
    ax.set_title("Percentage of Pairs Above Similarity Threshold")
    ax.set_xlabel("Similarity Threshold")
    ax.set_ylabel("Percentage of Pairs (%)")
    ax.grid(True)
    plt.tight_layout()
    return fig

def calc_pairwise_similarity_stats(sim_matrix):
    """计算两两分子间相似度的统计信息"""
    # 获取上三角矩阵的值（不包括对角线）
    triu_indices = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[triu_indices]
    
    stats = {
        "mean": np.mean(similarities),
        "std": np.std(similarities),
        "median": np.median(similarities),
        "min": np.min(similarities),
        "max": np.max(similarities),
        "q1": np.percentile(similarities, 25),
        "q3": np.percentile(similarities, 75)
    }
    return stats

def calc_similarity_threshold_stats(sim_matrix, thresholds=None):
    """计算不同相似度阈值下的分子对比例"""
    if thresholds is None:
        thresholds = np.arange(0, 1.1, 0.1)
    
    triu_indices = np.triu_indices_from(sim_matrix, k=1)
    similarities = sim_matrix[triu_indices]
    
    stats = {}
    for t in thresholds:
        fraction = np.mean(similarities >= t)
        stats[t] = fraction
    return stats

def plot_bit_frequency(fps_list, title="Fingerprint Bit Frequency"):
    """Plot fingerprint bit frequency distribution"""
    bit_freqs = np.mean([np.array(list(fp.ToBitString())) == '1' for fp in fps_list], axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(bit_freqs)), bit_freqs, 'b-', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Bit Position")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_similarity_heatmap(sim_matrix, max_mols=100):
    """绘制相似度矩阵热图"""
    # 如果分子数量太多，随机采样
    n_mols = len(sim_matrix)
    if n_mols > max_mols:
        indices = np.random.choice(n_mols, max_mols, replace=False)
        sim_matrix = sim_matrix[indices][:, indices]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(sim_matrix, cmap='viridis', ax=ax)
    ax.set_title("Similarity Matrix Heatmap")
    plt.tight_layout()
    return fig

def plot_nearest_neighbor_distribution(sim_matrix):
    """绘制最近邻相似度分布"""
    n = sim_matrix.shape[0]
    nn_sims = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        nn_sims.append(np.max(sim_matrix[i][mask]))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=nn_sims, bins=50, kde=True, ax=ax)
    ax.set_title("Nearest Neighbor Similarity Distribution")
    ax.set_xlabel("Nearest Neighbor Similarity")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig

def perform_clustering_analysis(sim_matrix, n_clusters=5, eps=0.3, min_samples=5, perplexity=30.0, force_device=None):
    """进行聚类分析（改进版本，确保CPU/GPU结果一致性）"""
    # 设置本地随机种子
    np.random.seed(RANDOM_SEED)
    
    cuda_available, device = initialize_cuda()
    
    # 根据debug模式强制选择设备
    if force_device == "cpu":
        st.info("🔧 Debug模式：强制使用CPU计算")
        cuda_available = False
    elif force_device == "gpu":
        if not cuda_available:
            st.warning("🔧 Debug模式：强制GPU失败，回退到CPU")
        else:
            st.info("🔧 Debug模式：强制使用GPU计算")

    # 统一的距离矩阵计算（确保数值稳定性）
    dist_matrix = 1 - sim_matrix
    # 确保距离矩阵的数值稳定性
    dist_matrix = np.clip(dist_matrix, 0, 2)  # 防止负值或过大值
    dist_matrix = (dist_matrix + dist_matrix.T) / 2  # 确保对称性
    np.fill_diagonal(dist_matrix, 0)  # 确保对角线为0

    if not cuda_available or force_device == "cpu":
        st.warning("⚠️ 使用CPU进行计算")
        
        # CPU版本的t-SNE（统一参数）
        tsne_cpu = TSNE(
            n_components=2, 
            metric='precomputed', 
            random_state=RANDOM_SEED,
            init='random', 
            learning_rate='auto',
            perplexity=min(perplexity, (len(dist_matrix) - 1) // 3),  # 防止perplexity过大
            n_iter=1000,  # 固定迭代次数
            verbose=0
        )
        coords = tsne_cpu.fit_transform(dist_matrix)

        # CPU版本的K-means（统一参数）
        kmeans_cpu = KMeans(
            n_clusters=n_clusters, 
            random_state=RANDOM_SEED, 
            n_init=10,  # 固定初始化次数
            algorithm='lloyd',  # 使用lloyd算法确保一致性
            max_iter=300,  # 固定最大迭代次数
            tol=1e-4  # 固定收敛容差
        )
        clusters = kmeans_cpu.fit_predict(coords)

        # CPU版本的DBSCAN（统一参数）
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
    else: # GPU 路径
        import torch 
        import cupy as cp

        st.info("🚀 GPU计算路径")
        
        # 转换为GPU数组
        dist_matrix_gpu = cp.asarray(dist_matrix, dtype=cp.float32)

        # GPU的t-SNE：由于cuML不支持预计算矩阵，统一使用CPU版本确保一致性
        st.info("🔧 t-SNE使用CPU版本确保一致性...")
        tsne_cpu = TSNE(
            n_components=2,
            perplexity=min(perplexity, (len(dist_matrix) - 1) // 3),
            random_state=RANDOM_SEED,
            metric='precomputed',
            init='random',
            learning_rate='auto',
            n_iter=1000,  # 固定迭代次数
            verbose=0
        )
        coords = tsne_cpu.fit_transform(dist_matrix)

        # GPU版本的K-means
        st.info("🚀 使用GPU K-means...")
        try:
            # 设置CuPy随机种子
            cp.random.seed(RANDOM_SEED)
            
            kmeans_gpu = cuKMeans(
                n_clusters=n_clusters, 
                random_state=RANDOM_SEED, 
                n_init=10,  # 固定初始化次数
                max_iter=300,  # 固定最大迭代次数
                tol=1e-4  # 固定收敛容差
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

        # 显示GPU内存使用情况
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

def plot_clustering_results(clustering_results, title="Clustering Results"):
    """Plot clustering results"""
    coords = clustering_results['coords']
    kmeans_clusters = clustering_results['kmeans_clusters']
    dbscan_clusters = clustering_results['dbscan_clusters']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # K-means results
    scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], c=kmeans_clusters, cmap='tab10')
    ax1.set_title("K-means Clustering")
    ax1.set_xlabel("t-SNE Dimension 1")
    ax1.set_ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter1, ax=ax1, label='Cluster ID')
    
    # DBSCAN results
    scatter2 = ax2.scatter(coords[:, 0], coords[:, 1], c=dbscan_clusters, cmap='tab10')
    ax2.set_title("DBSCAN Clustering")
    ax2.set_xlabel("t-SNE Dimension 1")
    ax2.set_ylabel("t-SNE Dimension 2")
    plt.colorbar(scatter2, ax=ax2, label='Cluster ID')
    
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig

def monitor_memory_usage():
    """监控内存使用"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        'rss': memory_info.rss / 1024**2,  # RSS in MB
        'vms': memory_info.vms / 1024**2,  # VMS in MB
        'percent': process.memory_percent()
    }

def calculate_distribution_metrics(coords_A, coords_B):
    """计算两组点的分布差异指标"""
    from sklearn.neighbors import KernelDensity
    
    debug_info = st.empty()
    timing_info = st.empty()
    start_total = time.time()
    
    # 检查输入数组是否为空或太小
    if len(coords_A) < 2 or len(coords_B) < 2:
        debug_info.warning("⚠️ Warning: One or both datasets have insufficient samples (need at least 2 samples)")
        return {
            "Center Distance": np.nan,
            "Dataset A Dispersion": np.nan if len(coords_A) < 2 else 0,
            "Dataset B Dispersion": np.nan if len(coords_B) < 2 else 0,
            "Dataset A Avg Distance": np.nan if len(coords_A) < 2 else 0,
            "Dataset B Avg Distance": np.nan if len(coords_B) < 2 else 0,
            "X-axis KS Statistic": np.nan,
            "X-axis KS p-value": np.nan,
            "Y-axis KS Statistic": np.nan,
            "Y-axis KS p-value": np.nan,
            "Distribution Overlap": 0.0
        }
    
    try:
        debug_info.info("🔄 开始计算分布指标...")
        
        # 计算中心点
        debug_info.info("1️⃣ 计算中心点和离散度...")
        start_time = time.time()
        center_A = np.mean(coords_A, axis=0)
        center_B = np.mean(coords_B, axis=0)
        
        # 计算中心点之间的欧氏距离
        center_distance = np.linalg.norm(center_A - center_B)
        
        # 计算每组点的离散度（方差）
        dispersion_A = np.mean(np.linalg.norm(coords_A - center_A, axis=1))
        dispersion_B = np.mean(np.linalg.norm(coords_B - center_B, axis=1))
        timing_info.info(f"✓ 中心点和离散度计算完成 ({time.time() - start_time:.2f}秒)")
        
        # 计算每组的平均距离（使用GPU加速）
        debug_info.info("2️⃣ 计算组内平均距离...")
        start_time = time.time()
        
        try:
            import cupy as cp
            debug_info.info("   ↪ 使用GPU加速距离计算...")
            
            # 计算A组平均距离
            coords_A_gpu = cp.asarray(coords_A)
            diff_A = coords_A_gpu[:, None, :] - coords_A_gpu[None, :, :]
            dist_matrix_A = cp.sqrt(cp.sum(diff_A ** 2, axis=2))
            # 排除自身距离（对角线）
            mask_A = cp.ones_like(dist_matrix_A, dtype=bool)
            cp.fill_diagonal(mask_A, False)
            dist_matrix_A = dist_matrix_A[mask_A]
            dist_A = float(cp.mean(dist_matrix_A).get())
            
            # 计算B组平均距离
            coords_B_gpu = cp.asarray(coords_B)
            diff_B = coords_B_gpu[:, None, :] - coords_B_gpu[None, :, :]
            dist_matrix_B = cp.sqrt(cp.sum(diff_B ** 2, axis=2))
            # 排除自身距离（对角线）
            mask_B = cp.ones_like(dist_matrix_B, dtype=bool)
            cp.fill_diagonal(mask_B, False)
            dist_matrix_B = dist_matrix_B[mask_B]
            dist_B = float(cp.mean(dist_matrix_B).get())
            
            # 清理GPU内存
            del coords_A_gpu, coords_B_gpu, diff_A, diff_B, dist_matrix_A, dist_matrix_B
            cp.get_default_memory_pool().free_all_blocks()
            
            debug_info.success("   ✓ GPU加速距离计算完成")
            
        except (ImportError, Exception) as e:
            debug_info.warning(f"   ⚠️ GPU加速失败 ({str(e)})，使用CPU计算...")
            
            # 使用numpy的矢量化操作
            # 计算A组平均距离
            diff_A = coords_A[:, None, :] - coords_A[None, :, :]
            dist_matrix_A = np.sqrt(np.sum(diff_A ** 2, axis=2))
            # 排除自身距离（对角线）
            mask_A = np.ones_like(dist_matrix_A, dtype=bool)
            np.fill_diagonal(mask_A, False)
            dist_A = np.mean(dist_matrix_A[mask_A])
            
            # 计算B组平均距离
            diff_B = coords_B[:, None, :] - coords_B[None, :, :]
            dist_matrix_B = np.sqrt(np.sum(diff_B ** 2, axis=2))
            # 排除自身距离（对角线）
            mask_B = np.ones_like(dist_matrix_B, dtype=bool)
            np.fill_diagonal(mask_B, False)
            dist_B = np.mean(dist_matrix_B[mask_B])
        
        timing_info.info(f"✓ 平均距离计算完成 ({time.time() - start_time:.2f}秒)")
        
        # 进行双样本KS检验
        debug_info.info("3️⃣ 执行KS检验...")
        start_time = time.time()
        ks_statistic_x, p_value_x = stats.ks_2samp(coords_A[:, 0], coords_B[:, 0])
        ks_statistic_y, p_value_y = stats.ks_2samp(coords_A[:, 1], coords_B[:, 1])
        timing_info.info(f"✓ KS检验完成 ({time.time() - start_time:.2f}秒)")
        
        # 使用cuML的KernelDensity计算分布重叠度
        debug_info.info("4️⃣ 计算密度分布...")
        start_time = time.time()
        try:
            from cuml.neighbors import KernelDensity as cuKernelDensity
            import cupy as cp
            
            debug_info.info("🚀 使用GPU加速密度估计计算")
            
            # 确保数据是浮点型
            coords_A = coords_A.astype(np.float64)
            coords_B = coords_B.astype(np.float64)
            
            # 转移数据到GPU
            debug_info.info("   ↪ 转移数据到GPU...")
            coords_A_gpu = cp.asarray(coords_A)
            coords_B_gpu = cp.asarray(coords_B)
            
            # 使用GPU版本的KDE
            debug_info.info("   ↪ 训练KDE模型...")
            kde_A = cuKernelDensity(bandwidth=0.1).fit(coords_A_gpu)
            kde_B = cuKernelDensity(bandwidth=0.1).fit(coords_B_gpu)
            
            # 生成网格点
            debug_info.info("   ↪ 生成评估网格...")
            x_min = min(coords_A[:, 0].min(), coords_B[:, 0].min())
            x_max = max(coords_A[:, 0].max(), coords_B[:, 0].max())
            y_min = min(coords_A[:, 1].min(), coords_B[:, 1].min())
            y_max = max(coords_A[:, 1].max(), coords_B[:, 1].max())
            
            # 添加边距
            margin_x = (x_max - x_min) * 0.1
            margin_y = (y_max - y_min) * 0.1
            x_min -= margin_x
            x_max += margin_x
            y_min -= margin_y
            y_max += margin_y
            
            # 生成网格点
            xx, yy = np.mgrid[x_min:x_max:30j, y_min:y_max:30j]
            positions = cp.asarray(np.vstack([xx.ravel(), yy.ravel()]).T)
            
            # 计算两个分布的密度
            debug_info.info("   ↪ 计算密度分布...")
            log_dens_A = kde_A.score_samples(positions)
            log_dens_B = kde_B.score_samples(positions)
            
            # 转回CPU进行后续计算
            debug_info.info("   ↪ 转移结果回CPU...")
            density_A = cp.exp(log_dens_A).reshape(xx.shape)
            density_B = cp.exp(log_dens_B).reshape(xx.shape)
            density_A = cp.asnumpy(density_A)
            density_B = cp.asnumpy(density_B)
            
            # 计算分布重叠度
            debug_info.info("   ↪ 计算分布重叠度...")
            min_sum = np.minimum(density_A, density_B).sum()
            max_sum = max(density_A.sum(), density_B.sum())
            overlap = min_sum / max_sum if max_sum > 0 else 0
            
            timing_info.info(f"✓ GPU密度估计完成 ({time.time() - start_time:.2f}秒)")
            
        except Exception as e:
            debug_info.warning(f"⚠️ GPU加速失败，切换到CPU版本 (原因: {str(e)})")
            start_time = time.time()
            
            # 回退到CPU版本的KDE
            kde_A = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(coords_A)
            kde_B = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(coords_B)
            
            # 生成网格点
            xx, yy = np.mgrid[x_min:x_max:30j, y_min:y_max:30j]
            positions = np.vstack([xx.ravel(), yy.ravel()]).T
            
            # 计算密度
            log_dens_A = kde_A.score_samples(positions)
            log_dens_B = kde_B.score_samples(positions)
            density_A = np.exp(log_dens_A).reshape(xx.shape)
            density_B = np.exp(log_dens_B).reshape(xx.shape)
            
            # 计算重叠度
            min_sum = np.minimum(density_A, density_B).sum()
            max_sum = max(density_A.sum(), density_B.sum())
            overlap = min_sum / max_sum if max_sum > 0 else 0
            
            timing_info.info(f"✓ CPU密度估计完成 ({time.time() - start_time:.2f}秒)")
        
        total_time = time.time() - start_total
        debug_info.success(f"✅ 所有分布指标计算完成！总耗时: {total_time:.2f}秒")
        
        return {
            "Center Distance": center_distance,
            "Dataset A Dispersion": dispersion_A,
            "Dataset B Dispersion": dispersion_B,
            "Dataset A Avg Distance": dist_A,
            "Dataset B Avg Distance": dist_B,
            "X-axis KS Statistic": ks_statistic_x,
            "X-axis KS p-value": p_value_x,
            "Y-axis KS Statistic": ks_statistic_y,
            "Y-axis KS p-value": p_value_y,
            "Distribution Overlap": overlap
        }
        
    except Exception as e:
        debug_info.error(f"❌ 计算分布指标时出错: {str(e)}")
        return {
            "Center Distance": np.nan,
            "Dataset A Dispersion": np.nan,
            "Dataset B Dispersion": np.nan,
            "Dataset A Avg Distance": np.nan,
            "Dataset B Avg Distance": np.nan,
            "X-axis KS Statistic": np.nan,
            "X-axis KS p-value": np.nan,
            "Y-axis KS Statistic": np.nan,
            "Y-axis KS p-value": np.nan,
            "Distribution Overlap": 0.0
        }

def compare_cpu_gpu_results(sim_matrix, n_clusters=5, eps=0.3, min_samples=5, perplexity=30.0):
    """对比CPU和GPU计算结果"""
    if not (CUDA_AVAILABLE and TORCH_AVAILABLE):
        st.error("❌ GPU不可用，无法进行CPU/GPU对比")
        return None, None
    
    st.markdown("### 🔬 CPU vs GPU 计算结果对比")
    
    with st.spinner("正在进行CPU/GPU对比计算..."):
        # CPU计算
        st.info("🖥️ 开始CPU计算...")
        start_time_cpu = time.time()
        cpu_results = perform_clustering_analysis(sim_matrix, n_clusters, eps, min_samples, perplexity, force_device="cpu")
        cpu_time = time.time() - start_time_cpu
        
        # GPU计算
        st.info("🚀 开始GPU计算...")
        start_time_gpu = time.time()
        gpu_results = perform_clustering_analysis(sim_matrix, n_clusters, eps, min_samples, perplexity, force_device="gpu")
        gpu_time = time.time() - start_time_gpu
        
        # 计算结果差异（增强版本）
        coord_diff = np.mean(np.abs(cpu_results['coords'] - gpu_results['coords']))
        coord_max_diff = np.max(np.abs(cpu_results['coords'] - gpu_results['coords']))
        coord_std_diff = np.std(cpu_results['coords'] - gpu_results['coords'])
        
        # 聚类一致性分析
        kmeans_accuracy = np.mean(cpu_results['kmeans_clusters'] == gpu_results['kmeans_clusters'])
        dbscan_accuracy = np.mean(cpu_results['dbscan_clusters'] == gpu_results['dbscan_clusters'])
        
        # 计算聚类结果的调整兰德指数（Adjusted Rand Index）
        try:
            from sklearn.metrics import adjusted_rand_score
            kmeans_ari = adjusted_rand_score(cpu_results['kmeans_clusters'], gpu_results['kmeans_clusters'])
            dbscan_ari = adjusted_rand_score(cpu_results['dbscan_clusters'], gpu_results['dbscan_clusters'])
        except ImportError:
            kmeans_ari = kmeans_accuracy  # 如果没有sklearn，使用简单的准确性
            dbscan_ari = dbscan_accuracy
        
        # 显示对比结果
        st.markdown("**⏱️ 计算时间对比**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("CPU时间", f"{cpu_time:.2f}s")
        with col2:
            st.metric("GPU时间", f"{gpu_time:.2f}s")
        with col3:
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            st.metric("加速比", f"{speedup:.2f}x")
        
        st.markdown("**📊 结果一致性分析**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("坐标差异(平均绝对误差)", f"{coord_diff:.6f}")
            st.metric("坐标差异(最大值)", f"{coord_max_diff:.6f}")
        with col2:
            st.metric("K-means聚类一致性", f"{kmeans_accuracy:.3f}")
            st.metric("K-means ARI分数", f"{kmeans_ari:.3f}")
        with col3:
            st.metric("DBSCAN聚类一致性", f"{dbscan_accuracy:.3f}")
            st.metric("DBSCAN ARI分数", f"{dbscan_ari:.3f}")
        
        # 添加一致性评估
        st.markdown("**🔍 一致性评估**")
        if coord_diff < 1e-6:
            st.success("✅ CPU和GPU计算结果高度一致！（数值误差极小）")
        elif coord_diff < 1e-4:
            st.info("ℹ️ CPU和GPU计算结果基本一致（存在轻微数值误差）")
        elif coord_diff < 1e-2:
            st.warning("⚠️ CPU和GPU计算结果存在中等程度差异")
        else:
            st.error("❌ CPU和GPU计算结果存在显著差异，建议检查算法实现")
        
        # 聚类一致性评估
        if kmeans_accuracy > 0.95 and dbscan_accuracy > 0.95:
            st.success("✅ 聚类结果高度一致")
        elif kmeans_accuracy > 0.8 and dbscan_accuracy > 0.8:
            st.warning("⚠️ 聚类结果基本一致，存在部分差异")
        else:
            st.error("❌ 聚类结果差异较大，可能需要调整随机种子或算法参数")
        
        # 可视化对比
        st.markdown("**📈 可视化对比**")
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_clustering_results(cpu_results, "CPU Clustering Results")
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            fig = plot_clustering_results(gpu_results, "GPU Clustering Results")
            st.pyplot(fig)
            plt.close(fig)
        
        # 详细差异分析
        with st.expander("🔍 详细差异分析"):
            st.markdown("**坐标差异分布**")
            coord_diffs = np.abs(cpu_results['coords'] - gpu_results['coords'])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.hist(coord_diffs[:, 0], bins=30, alpha=0.7, label='X坐标差异')
            ax1.set_title('X Coordinate Differences')
            ax1.set_xlabel('Absolute Difference')
            ax1.set_ylabel('Frequency')
            
            ax2.hist(coord_diffs[:, 1], bins=30, alpha=0.7, label='Y坐标差异', color='orange')
            ax2.set_title('Y Coordinate Differences')
            ax2.set_xlabel('Absolute Difference')
            ax2.set_ylabel('Frequency')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # 聚类差异详情
            st.markdown("**聚类差异详情**")
            kmeans_diff_count = np.sum(cpu_results['kmeans_clusters'] != gpu_results['kmeans_clusters'])
            dbscan_diff_count = np.sum(cpu_results['dbscan_clusters'] != gpu_results['dbscan_clusters'])
            
            st.write(f"- K-means聚类不一致点数: {kmeans_diff_count}/{len(cpu_results['kmeans_clusters'])}")
            st.write(f"- DBSCAN聚类不一致点数: {dbscan_diff_count}/{len(cpu_results['dbscan_clusters'])}")

    
    return cpu_results, gpu_results

def plot_distribution_comparison(coords_A, coords_B, metrics):
    """绘制分布对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 散点图
    scatter1 = ax1.scatter(coords_A[:, 0], coords_A[:, 1], c='blue', alpha=0.6, s=30, label='Dataset A')
    scatter2 = ax1.scatter(coords_B[:, 0], coords_B[:, 1], c='orange', alpha=0.6, s=30, label='Dataset B')
    ax1.set_title('Distribution Scatter Plot')
    ax1.set_xlabel('Dimension 1')
    ax1.set_ylabel('Dimension 2')
    ax1.legend()
    
    # 密度等高线图
    x = np.concatenate([coords_A[:, 0], coords_B[:, 0]])
    y = np.concatenate([coords_A[:, 1], coords_B[:, 1]])
    
    # 创建网格
    xmin, xmax = x.min() - 1, x.max() + 1
    ymin, ymax = y.min() - 1, y.max() + 1
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # 计算KDE
    values_A = np.vstack([coords_A[:, 0], coords_A[:, 1]])
    values_B = np.vstack([coords_B[:, 0], coords_B[:, 1]])
    
    kernel_A = gaussian_kde(values_A)
    kernel_B = gaussian_kde(values_B)
    
    z_A = np.reshape(kernel_A(positions), xx.shape)
    z_B = np.reshape(kernel_B(positions), xx.shape)
    
    # 绘制等高线
    ax2.contour(xx, yy, z_A, levels=5, colors='blue', alpha=0.5, label='Dataset A')
    ax2.contour(xx, yy, z_B, levels=5, colors='orange', alpha=0.5, label='Dataset B')
    
    # 添加图例
    ax2.legend()
    ax2.set_title('Density Contour Plot')
    ax2.set_xlabel('Dimension 1')
    ax2.set_ylabel('Dimension 2')
    
    plt.tight_layout()
    return fig

def perform_dimensionality_reduction(similarity_matrix, method="t-SNE", perplexity=30, n_neighbors=15, min_dist=0.1, force_device=None):
    """执行降维操作（改进版本，确保一致性）"""
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 设置随机种子
    np.random.seed(RANDOM_SEED)
    
    debug_info = st.empty()
    start_time = time.time()
    
    # 初始化CUDA
    cuda_available, device = initialize_cuda()
    
    # 根据debug模式强制选择设备
    if force_device == "cpu":
        st.info("🔧 Debug模式：降维强制使用CPU")
        cuda_available = False
    elif force_device == "gpu":
        if not cuda_available:
            st.warning("🔧 Debug模式：降维强制GPU失败，回退到CPU")
        else:
            st.info("🔧 Debug模式：降维强制使用GPU")
    
    # 将相似性矩阵转换为距离矩阵（确保数值稳定性）
    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, 2)  # 防止负值或过大值
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # 确保对称性
    np.fill_diagonal(distance_matrix, 0)  # 确保对角线为0
    
    try:
        if method == "t-SNE":
            # 为了确保一致性，t-SNE统一使用CPU版本
            debug_info.info("🔧 使用CPU版本t-SNE确保结果一致性...")
            
            # 调整perplexity防止过大
            effective_perplexity = min(perplexity, (len(distance_matrix) - 1) // 3)
            
            tsne = TSNE(
                n_components=2,
                perplexity=effective_perplexity,
                random_state=RANDOM_SEED,
                metric='precomputed',
                init='random',
                learning_rate='auto',
                n_iter=1000,  # 固定迭代次数
                verbose=0
            )
            coords = tsne.fit_transform(distance_matrix)
        
        elif method == "UMAP":
            # 为了确保一致性，UMAP也统一参数
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
            # PCA降维
            debug_info.info("🔧 使用PCA进行降维...")
            pca = PCA(n_components=2, random_state=RANDOM_SEED)
            coords = pca.fit_transform(similarity_matrix)
        
        else:
            raise ValueError(f"不支持的降维方法: {method}")
        
        # 验证输出维度
        if coords.shape[1] != 2:
            raise ValueError(f"降维结果维度不正确: {coords.shape}")
        
        debug_info.success(f"✅ 降维完成 ({time.time() - start_time:.2f}秒)")
        return coords
    
    except Exception as e:
        debug_info.error(f"降维过程出错: {str(e)}")
        st.error(f"降维失败: {str(e)}")
        return None
    
    finally:
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# 文件选择界面
st.subheader("1. 选择输入文件")

# 获取可用文件夹
folders = list_data_folders()

if not folders:
    st.warning("data目录下没有找到任何文件夹")
    st.stop()

selected_folder = st.selectbox("选择数据文件夹:", folders)

if selected_folder:
    # 获取该文件夹中的CSV文件
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

# SMILES列名输入
smiles_col = st.text_input("SMILES列名", value="SMILES", help="指定CSV文件中包含SMILES字符串的列名")

# 参数设置界面
st.subheader("2. 参数设置")

# 添加帮助信息在界面最上方，默认折叠
with st.expander("💡 指纹类型和参数设置说明", expanded=False):
    st.markdown("""
    ### 分子指纹类型说明
    
    #### 1. MACCS Keys (166位)
    - 特点：固定长度166位的结构键指纹
    - 适用：快速相似性搜索，基础结构特征识别
    - 优势：计算速度快，结果易解释
    - 局限：仅包含预定义的166个结构特征
    
    #### 2. Morgan指纹（ECFP）
    - 特点：环境敏感的循环指纹，基于分子中原子环境
    - 参数说明：
        - 半径(Radius)：2-3最常用
            - 2: ECFP4，捕获直径4的环境
            - 3: ECFP6，捕获更大范围的结构特征
        - 位数(nBits)：512-2048常用
            - 较小(512)：更快的计算，可能有信息损失
            - 较大(2048)：更详细的结构信息，占用更多内存
    - 适用：药物发现，精确结构匹配
    
    #### 3. RDKit指纹
    - 特点：路径型指纹，基于分子中的原子路径
    - 参数说明：
        - 最小路径(minPath)：1-2常用
            - 1：包含单键信息
            - 2：从双键开始
        - 最大路径(maxPath)：5-7常用
            - 较小：关注局部结构
            - 较大：包含更多大范围结构信息
        - 位数(nBits)：1024-4096常用
    - 适用：通用相似性搜索，结构骨架分析
    
    #### 4. Atom Pairs指纹
    - 特点：基于原子对之间的拓扑距离
    - 参数说明：
        - 最大距离(maxDistance)：10-30常用
            - 较小：关注近距离原子关系
            - 较大：包含分子整体结构信息
    - 适用：构象无关的结构比较
    
    ### 聚类参数设置指南
    
    #### 1. K-Means聚类
    - 簇数(n_clusters)建议：
        - 小数据集(<1000): 3-10
        - 中等数据集(1000-10000): 10-50
        - 大数据集(>10000): 50-200
    - 特点：
        - 优势：快速，结果易理解
        - 局限：需要预先指定簇数，对异常值敏感
    
    #### 2. DBSCAN聚类
    - eps(邻域半径)建议：
        - Morgan/RDKit指纹: 0.2-0.4
        - MACCS: 0.3-0.5
        - Atom Pairs: 0.25-0.45
    - min_samples建议：
        - 小数据集: 3-5
        - 大数据集: 5-10
    - 特点：
        - 优势：可发现任意形状的簇，自动处理异常点
        - 局限：参数敏感，计算较慢
    
    ### 可视化降维方法选择
    
    #### 1. PCA
    - 优势：快速，保持全局结构
    - 适用：初步数据探索，线性关系显著的数据
    
    #### 2. t-SNE
    - 参数：perplexity (5-50)
        - 小数据集：5-15
        - 大数据集：30-50
    - 优势：保持局部结构，聚类可视化效果好
    - 适用：非线性数据，需要详细查看局部结构
    
    #### 3. UMAP
    - 参数：
        - n_neighbors：10-50
        - min_dist：0.1-0.5
    - 优势：保持全局和局部结构，速度快
    - 适用：大规模数据集，需要平衡全局和局部结构
    """)

# 参数配置区域
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**指纹设置**")
    fp_type = st.selectbox(
        "指纹类型",
        ["morgan", "maccs", "topological", "atom_pairs"],
        help="""
        - morgan: Morgan/ECFP指纹，适合药物发现
        - maccs: MACCS Keys，166位结构键指纹
        - topological: 拓扑指纹，基于分子图
        - atom_pairs: 原子对指纹，基于原子对距离
        """
    )
    
    if fp_type == "morgan":
        radius = st.number_input(
            "Morgan半径", 
            min_value=1, 
            max_value=4, 
            value=3, 
            step=1,
            help="推荐值：2(ECFP4)或3(ECFP6)，3能捕获更大范围的分子环境"
        )
        nBits = st.number_input(
            "指纹位数", 
            min_value=512, 
            max_value=8192, 
            value=2048, 
            step=512,
            help="推荐值：2048，平衡计算效率和信息完整性"
        )
    elif fp_type == "topological":
        nBits = st.number_input(
            "指纹位数", 
            min_value=512, 
            max_value=8192, 
            value=2048, 
            step=512,
            help="推荐值：2048，适合大多数分析任务"
        )
    elif fp_type == "atom_pairs":
        max_distance = st.number_input(
            "最大距离", 
            min_value=3, 
            max_value=50, 
            value=15, 
            step=1,
            help="推荐值：15，平衡局部和全局结构信息"
        )
        nBits = st.number_input(
            "指纹位数", 
            min_value=512, 
            max_value=8192, 
            value=2048, 
            step=512,
            help="推荐值：2048，适合原子对指纹分析"
        )
    else:  # maccs
        nBits = 166  # MACCS固定位数

with col2:
    st.markdown("**聚类设置**")
    n_clusters = st.number_input(
        "聚类数量(K-means)",
        min_value=2,
        max_value=50,
        value=8,
        step=1,
        help="推荐值：5-15，根据数据集大小调整。小数据集用5-8，大数据集用10-15"
    )
    
    eps = st.number_input(
        "DBSCAN邻域大小",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
        format="%.2f",
        help="推荐值：0.2-0.3。Morgan指纹用0.25，MACCS用0.35，值越小聚类越紧密"
    )
    
    min_samples = st.number_input(
        "DBSCAN最小样本数",
        min_value=2,
        max_value=30,
        value=5,
        step=1,
        help="推荐值：3-8。小数据集用3-5，大数据集用5-8，影响噪声点识别"
    )

with col3:
    st.markdown("**可视化设置**")
    dim_reduction_method = st.selectbox(
        "降维方法",
        ["t-SNE", "UMAP", "PCA"],
        help="""
        - t-SNE: 保持局部结构，适合聚类可视化
        - UMAP: 平衡全局和局部结构，速度快
        - PCA: 保持全局结构，适合初步探索
        """
    )
    
    if dim_reduction_method == "t-SNE":
        perplexity = st.number_input(
            "Perplexity",
            min_value=5,
            max_value=100,
            value=25,
            step=5,
            help="推荐值：20-30。小数据集(<1000)用15-25，大数据集用25-50"
        )
    elif dim_reduction_method == "UMAP":
        n_neighbors = st.number_input(
            "邻居数量",
            min_value=5,
            max_value=200,
            value=30,
            step=5,
            help="推荐值：15-50。值越大保持更多全局结构，值越小保持更多局部结构"
        )
        min_dist = st.number_input(
            "最小距离",
            min_value=0.01,
            max_value=0.99,
            value=0.1,
            step=0.05,
            format="%.2f",
            help="推荐值：0.1-0.2。控制嵌入点间最小距离，影响聚类紧密度"
        )

    # Debug模式设置
    st.markdown("**🔧 Debug模式**")
    debug_mode = st.checkbox("启用Debug模式", help="允许手动选择CPU/GPU并对比计算结果")
    
    if debug_mode:
        # 随机种子设置
        custom_seed = st.number_input(
            "自定义随机种子", 
            min_value=1, 
            max_value=99999, 
            value=RANDOM_SEED, 
            help="设置随机种子以确保结果可重现性"
        )
        if custom_seed != RANDOM_SEED:
            # 更新全局随机种子
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
    
    if force_device == "gpu" and not (CUDA_AVAILABLE and TORCH_AVAILABLE):
        st.warning("⚠️ GPU不可用，将回退到CPU计算")
        
    enable_comparison = st.checkbox(
        "启用CPU/GPU结果对比", 
        help="同时运行CPU和GPU计算，对比结果差异(需要GPU可用)"
    )

# GPU状态显示
if CUDA_AVAILABLE and TORCH_AVAILABLE:
    st.success(f"✅ GPU加速可用")
elif TORCH_AVAILABLE:
    st.info("ℹ️ PyTorch可用但CUDA不可用，将使用CPU")
else:
    st.info("ℹ️ 将使用CPU计算")

# 结果预览区域
st.subheader("3. 分析结果")

# 创建预览容器
results_container = st.container()
with results_container:
    if 'selected_fileA' not in locals() or 'selected_fileB' not in locals():
        st.info("请先选择要分析的CSV文件，然后点击开始评估按钮")
    else:
        # 显示即将分析的文件信息
        st.markdown("**准备分析的文件:**")
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"数据集A: {selected_fileA}")
        with col2:
            st.text(f"数据集B: {selected_fileB}")

if st.button("🚀 开始多样性评估", type="primary") and 'selected_fileA' in locals() and 'selected_fileB' in locals():
    with st.spinner("正在进行多样性评估..."):
        # 显示内存使用情况
        mem_usage = monitor_memory_usage()
        st.sidebar.info(
            f"内存使用情况:\n"
            f"- RSS: {mem_usage['rss']:.1f} MB\n"
            f"- 内存占用: {mem_usage['percent']:.1f}%"
        )
        
        # 加载分子
        molsA, dfA = load_molecules_from_csv(fileA_path, smiles_col)
        molsB, dfB = load_molecules_from_csv(fileB_path, smiles_col)
        
        if molsA and molsB:
            # 清空之前的结果容器并显示新结果
            results_container.empty()
            
            with results_container:
                st.success(f"✅ 成功加载: 数据集A {len(molsA)}个分子, 数据集B {len(molsB)}个分子")
                
                # 显示基本统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("数据集A分子数", len(molsA))
                with col2:
                    st.metric("数据集B分子数", len(molsB))
                with col3:
                    st.metric("选择比例", f"{len(molsB)/len(molsA)*100:.1f}%")
                
                # 1. 指纹多样性分析
                st.markdown("### 📊 指纹多样性分析")
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 计算指纹
                with st.spinner(f"正在计算{fp_type}指纹..."):
                    total_mols = len(molsA) + len(molsB)
                
                # 创建指纹生成器
                if fp_type == "morgan":
                    fp_gen = GetMorganGenerator(radius=int(radius), fpSize=int(nBits))
                elif fp_type == "topological":
                    fp_gen = GetRDKitFPGenerator(fpSize=int(nBits), minPath=1, maxPath=7)
                
                fpsA = []
                fpsB = []
                
                # 处理数据集A
                for i, mol in enumerate(molsA):
                    try:
                        if fp_type == "morgan":
                            fp = fp_gen.GetFingerprint(mol)
                        elif fp_type == "maccs":
                            fp = AllChem.GetMACCSKeysFingerprint(mol)
                        elif fp_type == "topological":
                            fp = fp_gen.GetFingerprint(mol)
                        elif fp_type == "atom_pairs":
                            fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=int(nBits))
                        fpsA.append(fp)
                    except Exception as e:
                        st.warning(f"处理分子时出错: {str(e)}")
                        continue
                    
                    # 更新进度
                    progress = min(1.0, (i + 1) / total_mols * 0.5)
                    progress_bar.progress(progress)
                    status_text.text(f"处理数据集A: {i+1}/{len(molsA)}")
                
                # 处理数据集B
                for i, mol in enumerate(molsB):
                    try:
                        if fp_type == "morgan":
                            fp = fp_gen.GetFingerprint(mol)
                        elif fp_type == "maccs":
                            fp = AllChem.GetMACCSKeysFingerprint(mol)
                        elif fp_type == "topological":
                            fp = fp_gen.GetFingerprint(mol)
                        elif fp_type == "atom_pairs":
                            fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=int(nBits))
                        fpsB.append(fp)
                    except Exception as e:
                        st.warning(f"处理分子时出错: {str(e)}")
                        continue
                    
                    # 更新进度
                    progress = min(1.0, 0.5 + (i + 1) / total_mols * 0.5)
                    progress_bar.progress(progress)
                    status_text.text(f"处理数据集B: {i+1}/{len(molsB)}")
                
                # 计算相似性矩阵（只计算一次）
                sim_matrixA = compute_similarity_matrix(fpsA, "计算数据集A相似性矩阵")
                sim_matrixB = compute_similarity_matrix(fpsB, "计算数据集B相似性矩阵")
                
                progress_bar.empty()
                status_text.empty()
                
                # 添加新的可视化
                st.markdown("### 📈 指纹位频率分布")
                col1, col2 = st.columns(2)
                with col1:
                    fig = plot_bit_frequency(fpsA, "Dataset A Bit Frequency")
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col2:
                    fig = plot_bit_frequency(fpsB, "Dataset B Bit Frequency")
                    st.pyplot(fig)
                    plt.close(fig)
                
                st.markdown("### 🔥 相似性矩阵热图")
                col1, col2 = st.columns(2)
                with col1:
                    fig = plot_similarity_heatmap(sim_matrixA)
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col2:
                    fig = plot_similarity_heatmap(sim_matrixB)
                    st.pyplot(fig)
                    plt.close(fig)
            
                st.markdown("### 🎯 最近邻分布")
                col1, col2 = st.columns(2)
                with col1:
                    fig = plot_nearest_neighbor_distribution(sim_matrixA)
                    st.pyplot(fig)
                    plt.close(fig)
                
                with col2:
                    fig = plot_nearest_neighbor_distribution(sim_matrixB)
                    st.pyplot(fig)
                    plt.close(fig)
            
                # Debug模式处理
                if debug_mode and 'enable_comparison' in locals() and enable_comparison and CUDA_AVAILABLE and TORCH_AVAILABLE:
                    # CPU/GPU对比模式
                    compare_cpu_gpu_results(
                        sim_matrixA, 
                        n_clusters=n_clusters,
                        eps=eps,
                        min_samples=min_samples,
                        perplexity=perplexity if dim_reduction_method == "t-SNE" else 30.0
                    )
                else:
                    # 聚类分析
                    st.markdown("### 🔍 聚类分析")
                    with st.spinner("执行聚类分析..."):
                        # 执行聚类分析
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
                            fig = plot_clustering_results(clustering_resultsA, "Dataset A Clustering Results")
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        with col2:
                            fig = plot_clustering_results(clustering_resultsB, "Dataset B Clustering Results")
                            st.pyplot(fig)
                            plt.close(fig)
            
                # 2. 骨架多样性
                st.markdown("### 🧬 骨架多样性")
                col1, col2 = st.columns(2)
                
                num_scafA, scaf_entropyA, frac_singletonsA, scaff_freq_A = calc_scaffold_stats(molsA)
                num_scafB, scaf_entropyB, frac_singletonsB, scaff_freq_B = calc_scaffold_stats(molsB)
                
                f50_A = calc_F50(scaff_freq_A)
                f50_B = calc_F50(scaff_freq_B)
                
                with col1:
                    st.markdown("**数据集A**")
                    st.write(f"骨架总数: {num_scafA}")
                    st.write(f"骨架熵: {scaf_entropyA:.3f}")
                    st.write(f"单例骨架比例: {frac_singletonsA:.3f}")
                    st.write(f"F50值: {f50_A:.3f}")
                    
                    if not scaff_freq_A.empty:
                        st.write("前10个最常见骨架:")
                        st.dataframe(scaff_freq_A.head(10))
                
                with col2:
                    st.markdown("**数据集B**")
                    st.write(f"骨架总数: {num_scafB}")
                    st.write(f"骨架熵: {scaf_entropyB:.3f}")
                    st.write(f"单例骨架比例: {frac_singletonsB:.3f}")
                    st.write(f"F50值: {f50_B:.3f}")
                    
                    if not scaff_freq_B.empty:
                        st.write("前10个最常见骨架:")
                        st.dataframe(scaff_freq_B.head(10))
            
                # 3. 理化性质分布
                st.markdown("### 🧪 理化性质分布分析")
                
                propsA, prop_labels = calc_property_stats(molsA)
                propsB, prop_labels = calc_property_stats(molsB)
                
                # 显示统计摘要
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**数据集A 统计信息**")
                    summary_A = propsA.describe()
                    summary_A.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                    st.dataframe(summary_A.round(2))
                
                with col2:
                    st.markdown("**数据集B 统计信息**")
                    summary_B = propsB.describe()
                    summary_B.index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
                    st.dataframe(summary_B.round(2))
            
                # 绘制分布比较图
                st.markdown("**性质分布比较**")
                properties = list(prop_labels.keys())
                for i in range(0, len(properties), 2):
                    col1, col2 = st.columns(2)
                    
                    # 第一个图
                    with col1:
                        prop = properties[i]
                        fig = plot_property_comparison(propsA, propsB, prop, prop_labels[prop])
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # 第二个图 (如果存在)
                    with col2:
                        if i + 1 < len(properties):
                            prop = properties[i + 1]
                            fig = plot_property_comparison(propsA, propsB, prop, prop_labels[prop])
                            st.pyplot(fig)
                            plt.close(fig)
            
                # 导出评估报告
                st.markdown("### 📄 导出评估报告")
                
                # 更新报告数据
                report_data = {
                    "Metric": [
                        "Molecule Count",
                        "Mean Similarity",
                        "Similarity Std",
                        "Max Similarity",
                        "Min Similarity",
                        "Median Similarity"
                    ],
                    "Dataset A": [
                        len(molsA),
                        calculate_diversity_metrics(sim_matrixA)['Mean Similarity'],
                        calculate_diversity_metrics(sim_matrixA)['Similarity Std'],
                        calculate_diversity_metrics(sim_matrixA)['Max Similarity'],
                        calculate_diversity_metrics(sim_matrixA)['Min Similarity'],
                        calculate_diversity_metrics(sim_matrixA)['Median Similarity']
                    ],
                    "Dataset B": [
                        len(molsB),
                        calculate_diversity_metrics(sim_matrixB)['Mean Similarity'],
                        calculate_diversity_metrics(sim_matrixB)['Similarity Std'],
                        calculate_diversity_metrics(sim_matrixB)['Max Similarity'],
                        calculate_diversity_metrics(sim_matrixB)['Min Similarity'],
                        calculate_diversity_metrics(sim_matrixB)['Median Similarity']
                    ]
                }
                
                # 导出评估报告
                report_df = pd.DataFrame(report_data)
                csv = report_df.to_csv(index=False)
                st.download_button(
                    "下载评估报告",
                    csv,
                    "diversity_assessment_report.csv",
                    "text/csv",
                    key='download-report'
                )
            
                # 4. 结构分布分析
                st.markdown("### 📊 结构分布分析")
                
                debug_info = st.empty()
                timing_info = st.empty()

                # 获取数据集A和B的SMILES集合
                debug_info.info("开始处理数据集...")
                smiles_A = [Chem.MolToSmiles(mol) for mol in molsA]
                smiles_B = set([Chem.MolToSmiles(mol) for mol in molsB])
                debug_info.info(f"原始数据集A包含 {len(smiles_A)} 个分子")
                debug_info.info(f"选中的数据集B包含 {len(smiles_B)} 个分子")

                # 创建两个独立的数据集用于比较
                dataset_A = []  # 原始完整数据集
                dataset_B = []  # 被选中的子集
                sources_A = []  # 原始数据集的标记
                sources_B = []  # 子集的标记

                # 处理数据集A（完整数据集）
                for mol in molsA:
                    dataset_A.append(mol)
                    sources_A.append('A')

                # 处理数据集B（选中的子集）
                for mol in molsB:
                    dataset_B.append(mol)
                    sources_B.append('B')

                # 统计信息
                debug_info.info(f"处理完成:")
                debug_info.info(f"- 原始数据集大小: {len(dataset_A)}个分子")
                debug_info.info(f"- 选中子集大小: {len(dataset_B)}个分子")

                # 计算两个数据集的指纹
                debug_info.info("开始计算指纹...")
                start_time = time.time()

                # 计算数据集A的指纹
                fps_A = []
                for i, mol in enumerate(dataset_A):
                    try:
                        if fp_type == "morgan":
                            fp = fp_gen.GetFingerprint(mol)
                        elif fp_type == "maccs":
                            fp = AllChem.GetMACCSKeysFingerprint(mol)
                        elif fp_type == "topological":
                            fp = fp_gen.GetFingerprint(mol)
                        elif fp_type == "atom_pairs":
                            fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=int(nBits))
                        fps_A.append(fp)
                        
                        if (i + 1) % 100 == 0:
                            debug_info.info(f"已处理数据集A: {i+1}/{len(dataset_A)} 个分子的指纹")
                    except Exception as e:
                        st.warning(f"处理数据集A第{i+1}个分子指纹时出错: {str(e)}")
                        continue

                # 计算数据集B的指纹
                fps_B = []
                for i, mol in enumerate(dataset_B):
                    try:
                        if fp_type == "morgan":
                            fp = fp_gen.GetFingerprint(mol)
                        elif fp_type == "maccs":
                            fp = AllChem.GetMACCSKeysFingerprint(mol)
                        elif fp_type == "topological":
                            fp = fp_gen.GetFingerprint(mol)
                        elif fp_type == "atom_pairs":
                            fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=int(nBits))
                        fps_B.append(fp)
                        
                        if (i + 1) % 100 == 0:
                            debug_info.info(f"已处理数据集B: {i+1}/{len(dataset_B)} 个分子的指纹")
                    except Exception as e:
                        st.warning(f"处理数据集B第{i+1}个分子指纹时出错: {str(e)}")
                        continue

                timing_info.text(f"指纹计算耗时: {time.time() - start_time:.2f}秒")
                debug_info.info(f"成功计算的指纹数量 - 数据集A: {len(fps_A)}, 数据集B: {len(fps_B)}")

                # 计算相似度矩阵
                debug_info.info("开始计算相似度矩阵...")
                start_time = time.time()

                # 合并所有指纹并计算相似度矩阵
                fps_combined = fps_A + fps_B
                sources_combined = sources_A + sources_B
                sim_matrix_combined = compute_similarity_matrix(fps_combined, "计算合并数据集相似性矩阵")
                timing_info.text(f"相似度矩阵计算耗时: {time.time() - start_time:.2f}秒")

                # 降维
                debug_info.info(f"开始使用{dim_reduction_method}进行降维...")
                start_time = time.time()

                # 添加输入矩阵的形状信息
                debug_info.info(f"输入相似性矩阵形状: {sim_matrix_combined.shape}")

                coords = perform_dimensionality_reduction(
                    sim_matrix_combined,
                    method=dim_reduction_method,
                    perplexity=perplexity if dim_reduction_method == "t-SNE" else None,
                    n_neighbors=n_neighbors if dim_reduction_method == "UMAP" else None,
                    min_dist=min_dist if dim_reduction_method == "UMAP" else None,
                    force_device=force_device if debug_mode else None
                )

                # 添加降维结果的形状信息
                debug_info.info(f"降维后的坐标形状: {coords.shape}")
                if coords.shape[1] != 2:
                    debug_info.error(f"错误：降维结果维度不正确，期望2维但得到{coords.shape[1]}维")
                    st.error("降维过程出错：结果维度不正确") 
                    st.stop()

                timing_info.text(f"降维计算耗时: {time.time() - start_time:.2f}秒")

                # 分离坐标
                debug_info.info("分离数据集坐标...")
                coords_A = coords[:len(dataset_A)]  # 原始数据集的坐标
                coords_B = coords[len(dataset_A):]  # 选中子集的坐标

                debug_info.info(f"分离后的坐标形状:")
                debug_info.info(f"- 原始数据集(A): {coords_A.shape}")
                debug_info.info(f"- 选中子集(B): {coords_B.shape}")

                # 验证坐标数据的有效性
                if len(coords_A) < 2 or len(coords_B) < 2:
                    debug_info.error("错误：一个或两个数据集样本数量不足（需要至少2个样本）")
                    st.error("无法进行分布分析：样本数量不足")
                elif np.isnan(coords_A).any() or np.isnan(coords_B).any():
                    debug_info.error("错误：坐标中包含NaN值")
                    st.error("无法进行分布分析：坐标包含无效值")
                else:
                    # 计算分布指标
                    debug_info.info("开始计算分布指标...")
                    start_time = time.time()
                    distribution_metrics = calculate_distribution_metrics(coords_A, coords_B)
                    timing_info.text(f"分布指标计算耗时: {time.time() - start_time:.2f}秒")

                    # 可视化结果
                    debug_info.info("开始生成可视化结果...")
                    start_time = time.time()
                    st.markdown("**结构分布对比** ({})".format(dim_reduction_method))
                    st.markdown("""
                    - 蓝色点：原始完整数据集（数据集A）
                    - 橙色点：被选中的子集（数据集B）
                    """)
                    fig = plot_distribution_comparison(coords_A, coords_B, distribution_metrics)
                    st.pyplot(fig)
                    plt.close(fig)
                    timing_info.text(f"可视化生成耗时: {time.time() - start_time:.2f}秒")

                    # 显示GPU内存使用情况（如果可用）
                    if torch.cuda.is_available():
                        gpu_mem_alloc = torch.cuda.memory_allocated() / 1024**2
                        gpu_mem_cached = torch.cuda.memory_reserved() / 1024**2
                        debug_info.info(
                            f"GPU内存使用情况:\n"
                            f"- 已分配: {gpu_mem_alloc:.1f} MB\n"
                            f"- 已缓存: {gpu_mem_cached:.1f} MB"
                        )

                    # 显示详细的统计信息
                    st.markdown("**分布统计指标**")
                    metrics_df = pd.DataFrame({
                        "Metric": list(distribution_metrics.keys()),
                        "Value": list(distribution_metrics.values())
                    })
                    st.dataframe(metrics_df.style.format({
                        "Value": lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
                    })) 