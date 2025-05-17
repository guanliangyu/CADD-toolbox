"""
聚类和多样性选择工具 - 实现多种聚类算法和代表分子选择
"""
import numpy as np
import logging
from typing import List, Dict, Tuple, Union, Optional, Any, Callable
import time
from tqdm import tqdm

from rdkit import Chem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker

from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 引入GPU工具
try:
    from .gpu_utils import check_gpu_availability, gpu_distance_matrix, GPUKMeans
    GPU_TOOLS_AVAILABLE = True
except ImportError:
    GPU_TOOLS_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# 设置日志
logger = logging.getLogger(__name__)


def calculate_distance_matrix(fps: List[Any], metric: str = 'tanimoto', 
                             config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    计算距离矩阵
    
    参数:
        fps: 指纹或特征向量列表
        metric: 距离度量方法
        config: 配置字典（可选，用于GPU配置）
        
    返回:
        距离矩阵（方阵）
    """
    n = len(fps)
    
    # 检查是否可以使用GPU加速
    use_gpu = False
    if config is not None and GPU_TOOLS_AVAILABLE:
        gpu_config = config.get('gpu', {})
        if gpu_config.get('enabled', False):
            # 检查GPU可用性
            if gpu_config.get('auto_detect', True):
                gpu_status = check_gpu_availability()
                use_gpu = gpu_status['any_gpu'] and gpu_config.get('features', {}).get('distances', True)
            else:
                use_gpu = True
    
    # 首先尝试将指纹转换为NumPy数组，便于后续处理
    try:
        # 转换RDKit指纹为NumPy数组
        if hasattr(fps[0], '__len__'):  # 检查是否可迭代
            features = np.array([list(fp) for fp in fps if fp is not None], dtype=np.float32)
            
            # 如果支持GPU加速并且是欧几里得距离，使用GPU计算
            if use_gpu and metric in ['euclidean', 'tanimoto']:
                logger.info(f"使用GPU计算{n}x{n}距离矩阵")
                return gpu_distance_matrix(features, metric=metric)
            
    except Exception as e:
        logger.warning(f"转换为NumPy数组失败: {e}，使用原始指纹")
    
    # 如果无法使用GPU或转换失败，回退到CPU计算
    logger.info(f"使用CPU计算{n}x{n}距离矩阵")
    distances = np.zeros((n, n), dtype=np.float32)
    
    # 对于大型数据集，这可能非常耗时，因此添加进度条
    for i in tqdm(range(n), desc='计算距离矩阵'):
        for j in range(i+1, n):
            if metric == 'tanimoto':
                # 如果是RDKit指纹对象
                if hasattr(DataStructs, 'TanimotoSimilarity'):
                    try:
                        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                        d = 1.0 - sim
                    except:
                        # 如果是NumPy数组
                        a_bits = np.array(fps[i], dtype=bool)
                        b_bits = np.array(fps[j], dtype=bool)
                        sim = np.sum(a_bits & b_bits) / np.sum(a_bits | b_bits)
                        d = 1.0 - sim
                else:
                    # 如果是NumPy数组
                    a_bits = np.array(fps[i], dtype=bool)
                    b_bits = np.array(fps[j], dtype=bool)
                    sim = np.sum(a_bits & b_bits) / np.sum(a_bits | b_bits)
                    d = 1.0 - sim
            elif metric == 'euclidean':
                d = np.linalg.norm(fps[i] - fps[j])
            else:
                raise ValueError(f"不支持的距离度量: {metric}")
                
            distances[i, j] = d
            distances[j, i] = d
    
    return distances


def butina_clustering(fps: List[Any], cutoff: float = 0.4, 
                     distance_metric: str = 'tanimoto',
                     config: Optional[Dict[str, Any]] = None) -> List[List[int]]:
    """
    使用Butina算法进行聚类
    
    参数:
        fps: 指纹或特征向量列表
        cutoff: 距离阈值（当distance_metric为tanimoto时，实际为1-相似度阈值）
        distance_metric: 距离度量方法
        config: 配置字典（可选，用于GPU配置）
        
    返回:
        聚类结果，每个簇是分子索引列表
    """
    n = len(fps)
    
    # 检查是否可以使用GPU加速
    use_gpu = False
    if config is not None and GPU_TOOLS_AVAILABLE:
        gpu_config = config.get('gpu', {})
        if gpu_config.get('enabled', False):
            # 检查GPU可用性
            if gpu_config.get('auto_detect', True):
                gpu_status = check_gpu_availability()
                use_gpu = gpu_status['any_gpu'] and gpu_config.get('features', {}).get('distances', True)
            else:
                use_gpu = True
                
    # 如果可以使用GPU加速且指纹可转换为NumPy数组
    if use_gpu:
        try:
            # 计算完整距离矩阵
            distance_matrix = calculate_distance_matrix(fps, metric=distance_metric, config=config)
            
            # 提取下三角矩阵（一维形式）
            dists = []
            for i in range(n):
                for j in range(i+1, n):
                    dists.append(distance_matrix[i, j])
                    
            # 执行Butina聚类
            clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
            logger.info(f"使用GPU加速的Butina聚类完成，共{len(clusters)}个簇")
            return clusters
        except Exception as e:
            logger.warning(f"GPU加速Butina聚类失败: {e}，回退到CPU计算")
    
    # 使用CPU计算（原始实现）
    logger.info("使用CPU进行Butina聚类")
    # 计算距离矩阵（一维压缩格式）
    dists = []
    for i in tqdm(range(n), desc='计算Butina聚类距离'):
        for j in range(i+1, n):
            if distance_metric == 'tanimoto':
                try:
                    sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    d = 1.0 - sim
                except:
                    # 如果是NumPy数组
                    a_bits = np.array(fps[i], dtype=bool)
                    b_bits = np.array(fps[j], dtype=bool)
                    sim = np.sum(a_bits & b_bits) / np.sum(a_bits | b_bits)
                    d = 1.0 - sim
            elif distance_metric == 'euclidean':
                d = np.linalg.norm(np.array(fps[i]) - np.array(fps[j]))
            else:
                raise ValueError(f"不支持的距离度量: {distance_metric}")
                
            dists.append(d)
    
    # 执行Butina聚类
    clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
    logger.info(f"使用CPU的Butina聚类完成，共{len(clusters)}个簇")
    return clusters


def kmeans_clustering(features: np.ndarray, n_clusters: int, 
                     batch_size: int = 1000, max_iter: int = 100,
                     config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用K-means算法进行聚类
    
    参数:
        features: 特征矩阵，每行是一个分子的特征向量
        n_clusters: 聚类数量
        batch_size: 小批量K-means的批量大小
        max_iter: 最大迭代次数
        config: 配置字典（可选，用于GPU配置）
        
    返回:
        (簇标签数组, 簇中心数组)
    """
    # 检查是否可以使用GPU加速
    use_gpu = False
    if config is not None and GPU_TOOLS_AVAILABLE:
        gpu_config = config.get('gpu', {})
        if gpu_config.get('enabled', False):
            # 检查GPU可用性
            if gpu_config.get('auto_detect', True):
                gpu_status = check_gpu_availability()
                use_gpu = gpu_status['any_gpu'] and gpu_config.get('features', {}).get('kmeans', True)
            else:
                use_gpu = True
    
    # 如果可以使用GPU加速
    if use_gpu:
        try:
            # 使用GPU KMeans
            logger.info(f"使用GPU进行K-means聚类, k={n_clusters}")
            gpu_kmeans = GPUKMeans(
                n_clusters=n_clusters,
                random_state=42,
                max_iter=max_iter
            )
            
            # 执行聚类
            labels = gpu_kmeans.fit_predict(features)
            centers = gpu_kmeans.cluster_centers_
            
            logger.info(f"GPU K-means聚类完成")
            return labels, centers
        except Exception as e:
            logger.warning(f"GPU K-means聚类失败: {e}，回退到CPU计算")
    
    # 如果无法使用GPU，使用CPU KMeans
    if len(features) < 10000:
        logger.info(f"使用标准K-means, k={n_clusters}")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=max_iter, n_init=10)
    else:
        logger.info(f"使用MiniBatch K-means, 批量大小={batch_size}, k={n_clusters}")
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, 
                                 batch_size=batch_size, max_iter=max_iter)
    
    # 执行聚类
    labels = kmeans.fit_predict(features)
    centers = kmeans.cluster_centers_
    
    logger.info(f"CPU K-means聚类完成")
    return labels, centers


def maxmin_selection(fps: List[Any], num_to_select: int, 
                    distance_metric: str = 'tanimoto', 
                    seed_idx: int = -1,
                    config: Optional[Dict[str, Any]] = None) -> List[int]:
    """
    使用MaxMin算法选择具有最大多样性的子集
    
    参数:
        fps: 指纹或特征向量列表
        num_to_select: 要选择的分子数量
        distance_metric: 距离度量方法
        seed_idx: 初始种子索引，-1表示随机选择
        config: 配置字典（可选，用于GPU配置）
        
    返回:
        选择的分子索引列表
    """
    n = len(fps)
    if num_to_select >= n:
        return list(range(n))
    
    # 检查是否可以使用GPU加速
    use_gpu = False
    if config is not None and GPU_TOOLS_AVAILABLE:
        gpu_config = config.get('gpu', {})
        if gpu_config.get('enabled', False):
            # 检查GPU可用性
            if gpu_config.get('auto_detect', True):
                gpu_status = check_gpu_availability()
                use_gpu = gpu_status['any_gpu'] and gpu_config.get('features', {}).get('distances', True)
            else:
                use_gpu = True
                
    # 如果可以使用GPU加速，预先计算完整的距离矩阵
    if use_gpu:
        try:
            # 计算完整距离矩阵
            logger.info("使用GPU预计算MaxMin距离矩阵")
            distance_matrix = calculate_distance_matrix(fps, metric=distance_metric, config=config)
            
            # 定义使用预计算距离的距离函数
            def distance_fn(i, j):
                return distance_matrix[i, j]
                
            # 使用MaxMinPicker
            picker = MaxMinPicker()
            if seed_idx < 0:
                seed_idx = np.random.randint(0, n)
                
            logger.info(f"开始GPU加速的MaxMin选择，目标数量={num_to_select}")
            selected_indices = list(picker.LazyPick(
                distance_fn, n, num_to_select, seed_idx
            ))
            
            logger.info(f"GPU MaxMin选择完成，选择了{len(selected_indices)}个分子")
            return selected_indices
        except Exception as e:
            logger.warning(f"GPU MaxMin选择失败: {e}，回退到CPU计算")
    
    # 使用CPU计算
    logger.info(f"使用CPU进行MaxMin选择，目标数量={num_to_select}")
    # 定义距离函数
    def distance_fn(i, j):
        if distance_metric == 'tanimoto':
            try:
                # 如果是RDKit指纹对象
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                return 1.0 - sim
            except:
                # 如果是NumPy数组
                a_bits = np.array(fps[i], dtype=bool)
                b_bits = np.array(fps[j], dtype=bool)
                sim = np.sum(a_bits & b_bits) / np.sum(a_bits | b_bits)
                return 1.0 - sim
        elif distance_metric == 'euclidean':
            return np.linalg.norm(np.array(fps[i]) - np.array(fps[j]))
        else:
            raise ValueError(f"不支持的距离度量: {distance_metric}")
    
    # 使用MaxMinPicker
    picker = MaxMinPicker()
    if seed_idx < 0:
        seed_idx = np.random.randint(0, n)
    
    selected_indices = list(picker.LazyBitVectorPick(
        distance_fn, n, num_to_select, seed_idx
    ))
    
    logger.info(f"CPU MaxMin选择完成，选择了{len(selected_indices)}个分子")
    return selected_indices


def hdbscan_clustering(features: np.ndarray, min_cluster_size: int = 5, 
                      min_samples: int = 5,
                      config: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """
    使用HDBSCAN算法进行基于密度的聚类
    
    参数:
        features: 特征矩阵，每行是一个分子的特征向量
        min_cluster_size: 最小簇大小
        min_samples: 核心点的最小样本数
        config: 配置字典（可选，用于GPU配置）
        
    返回:
        簇标签数组（-1表示噪声点）
    """
    if not HDBSCAN_AVAILABLE:
        logger.warning("HDBSCAN库不可用，无法执行基于密度的聚类")
        # 返回所有点都是噪声的标签
        return np.array([-1] * len(features))
        
    # 检查是否可以使用RAPIDS cuML加速
    use_gpu = False
    if config is not None and GPU_TOOLS_AVAILABLE:
        gpu_config = config.get('gpu', {})
        if gpu_config.get('enabled', False):
            # 检查GPU可用性
            if gpu_config.get('auto_detect', True):
                gpu_status = check_gpu_availability()
                use_gpu = gpu_status['cuml'] and gpu_config.get('features', {}).get('kmeans', True)
            else:
                use_gpu = True
                
    # 尝试使用GPU加速的HDBSCAN
    if use_gpu:
        try:
            import cuml.cluster
            
            # 使用cuML的HDBSCAN实现
            logger.info(f"使用GPU进行HDBSCAN聚类")
            clusterer = cuml.cluster.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            
            # 执行聚类
            labels = clusterer.fit_predict(features)
            
            # 转换回CPU
            if hasattr(labels, 'to_numpy'):
                labels = labels.to_numpy()
            else:
                labels = np.array(labels)
                
            logger.info(f"GPU HDBSCAN聚类完成")
            return labels
        except (ImportError, Exception) as e:
            logger.warning(f"GPU HDBSCAN聚类失败: {e}，回退到CPU计算")
    
    # 使用CPU HDBSCAN
    logger.info(f"使用CPU进行HDBSCAN聚类")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                               min_samples=min_samples,
                               metric='euclidean',
                               cluster_selection_method='eom')
    
    # 执行聚类
    labels = clusterer.fit_predict(features)
    
    logger.info(f"CPU HDBSCAN聚类完成")
    return labels


def select_cluster_representatives(clusters: List[List[int]], fps: List[Any], 
                                 method: str = 'centroid',
                                 config: Optional[Dict[str, Any]] = None) -> List[int]:
    """
    从每个簇中选择代表分子
    
    参数:
        clusters: 聚类结果，每个簇是分子索引列表
        fps: 指纹或特征向量列表
        method: 选择方法 ('centroid', 'random', 'first')
        config: 配置字典（可选，用于GPU配置）
        
    返回:
        代表分子索引列表
    """
    representatives = []
    
    # 检查是否可以使用GPU加速
    use_gpu = False
    if method == 'centroid' and config is not None and GPU_TOOLS_AVAILABLE:
        gpu_config = config.get('gpu', {})
        if gpu_config.get('enabled', False):
            # 检查GPU可用性
            if gpu_config.get('auto_detect', True):
                gpu_status = check_gpu_availability()
                use_gpu = gpu_status['any_gpu'] and gpu_config.get('features', {}).get('distances', True)
            else:
                use_gpu = True
                
    # 如果使用GPU加速且选择方法是centroid
    distance_matrix = None
    if use_gpu and method == 'centroid':
        try:
            # 先转换指纹为支持的格式
            if hasattr(fps[0], '__len__'):
                features = np.array([list(fp) for fp in fps if fp is not None], dtype=np.float32)
                
                # 计算完整距离矩阵
                logger.info("使用GPU预计算簇代表的距离矩阵")
                distance_matrix = calculate_distance_matrix(features, metric='tanimoto', config=config)
        except Exception as e:
            logger.warning(f"GPU距离矩阵计算失败: {e}，使用逐对计算")
    
    logger.info(f"开始从{len(clusters)}个簇中选择代表分子")
    for cluster in clusters:
        if not cluster:
            continue
            
        if method == 'first':
            # 简单选择簇中第一个
            representatives.append(cluster[0])
        elif method == 'random':
            # 随机选择一个
            representatives.append(np.random.choice(cluster))
        elif method == 'centroid':
            # 选择最接近簇中心的分子
            
            if len(cluster) == 1:
                representatives.append(cluster[0])
                continue
            
            # 如果有预计算的距离矩阵，使用它
            if distance_matrix is not None:
                # 提取子矩阵
                sub_matrix = distance_matrix[np.ix_(cluster, cluster)]
                
                # 计算每个点到其他所有点的平均距离
                avg_dists = np.mean(sub_matrix, axis=1)
                
                # 选择平均距离最小的点
                best_idx = cluster[np.argmin(avg_dists)]
                representatives.append(best_idx)
                continue
                
            # 否则，使用传统方法计算
            best_idx = None
            best_similarity = -1
            
            for i, idx in enumerate(cluster):
                total_sim = 0
                for j, other_idx in enumerate(cluster):
                    if i != j:
                        try:
                            sim = DataStructs.TanimotoSimilarity(fps[idx], fps[other_idx])
                        except:
                            # 如果是NumPy数组
                            a_bits = np.array(fps[idx], dtype=bool)
                            b_bits = np.array(fps[other_idx], dtype=bool)
                            sim = np.sum(a_bits & b_bits) / np.sum(a_bits | b_bits)
                        total_sim += sim
                
                avg_sim = total_sim / (len(cluster) - 1)
                if avg_sim > best_similarity:
                    best_similarity = avg_sim
                    best_idx = idx
            
            representatives.append(best_idx)
    
    logger.info(f"簇代表选择完成，共{len(representatives)}个代表分子")
    return representatives


def select_representatives_from_kmeans(features: np.ndarray, labels: np.ndarray, 
                                     centers: np.ndarray,
                                     config: Optional[Dict[str, Any]] = None) -> List[int]:
    """
    从K-means聚类结果中选择最接近聚类中心的代表分子
    
    参数:
        features: 特征矩阵，每行是一个分子的特征向量
        labels: 簇标签数组
        centers: 簇中心数组
        config: 配置字典（可选，用于GPU配置）
        
    返回:
        代表分子索引列表
    """
    n_clusters = len(centers)
    representatives = []
    
    # 检查是否可以使用GPU加速
    use_gpu = False
    if config is not None and GPU_TOOLS_AVAILABLE:
        gpu_config = config.get('gpu', {})
        if gpu_config.get('enabled', False):
            # 检查GPU可用性
            if gpu_config.get('auto_detect', True):
                gpu_status = check_gpu_availability()
                use_gpu = gpu_status['any_gpu'] and (
                    gpu_status['faiss_gpu'] or gpu_status['cupy'] or gpu_status['torch']
                )
            else:
                use_gpu = True
    
    # 如果可以使用GPU加速，使用批处理计算
    if use_gpu:
        try:
            import faiss
            
            # 将中心转换为float32格式
            centers_float32 = centers.astype(np.float32)
            
            # 创建GPU资源
            res = faiss.StandardGpuResources()
            
            # 创建FAISS索引
            index = faiss.IndexFlatL2(centers_float32.shape[1])
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
            
            # 添加聚类中心
            gpu_index.add(centers_float32)
            
            logger.info("使用GPU批量计算K-means代表分子")
            
            # 为每个簇找到最接近中心的分子
            for i in range(n_clusters):
                # 找出属于当前簇的所有点
                cluster_indices = np.where(labels == i)[0]
                
                if len(cluster_indices) == 0:
                    continue  # 跳过空簇
                
                # 提取簇中的点
                cluster_points = features[cluster_indices].astype(np.float32)
                
                # 使用FAISS查询中心i的最近邻
                _, indices = gpu_index.search(cluster_points, 1)
                
                # 找出最接近中心i的点
                if indices[0, 0] == i:  # 确认匹配的是正确的中心
                    closest_point_idx = cluster_indices[0]  # 默认使用第一个
                    
                    # 如果有多个点，计算距离
                    if len(cluster_indices) > 1:
                        # 计算到中心的距离
                        distances = np.linalg.norm(cluster_points - centers[i], axis=1)
                        closest_point_idx = cluster_indices[np.argmin(distances)]
                    
                    representatives.append(closest_point_idx)
            
            logger.info(f"GPU计算完成，找到{len(representatives)}个代表分子")
            return representatives
        except Exception as e:
            logger.warning(f"GPU K-means代表计算失败: {e}，回退到CPU计算")
    
    # 使用CPU计算
    logger.info("使用CPU计算K-means代表分子")
    for i in range(n_clusters):
        # 找出属于当前簇的所有点
        cluster_indices = np.where(labels == i)[0]
        
        if len(cluster_indices) == 0:
            continue  # 跳过空簇
            
        # 计算到中心的距离
        cluster_points = features[cluster_indices]
        distances = np.linalg.norm(cluster_points - centers[i], axis=1)
        
        # 选择最近的点
        closest_point_idx = cluster_indices[np.argmin(distances)]
        representatives.append(closest_point_idx)
    
    logger.info(f"CPU计算完成，找到{len(representatives)}个代表分子")
    return representatives


def perform_clustering(fps: List[Any], method: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行聚类分析
    
    参数:
        fps: 指纹或特征向量列表
        method: 聚类方法 ('butina', 'kmeans', 'maxmin')
        config: 配置字典
        
    返回:
        包含聚类结果的字典
    """
    logger.info(f"开始执行{method}聚类")
    
    # 提取有效指纹
    valid_fps = [fp for fp in fps if fp is not None]
    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
    
    if not valid_fps:
        return {
            'status': 'error',
            'message': '没有有效的指纹数据',
            'labels': None,
            'centers': None,
            'valid_indices': valid_indices
        }
    
    try:
        if method == 'butina':
            # 执行Butina聚类
            cutoff = config.get('clustering', {}).get('butina', {}).get('cutoff', 0.4)
            clusters = butina_clustering(valid_fps, cutoff=cutoff, config=config)
            
            # 转换为标签形式
            labels = np.zeros(len(valid_fps), dtype=int)
            for i, cluster in enumerate(clusters):
                for idx in cluster:
                    labels[idx] = i
            
            return {
                'status': 'success',
                'message': f'Butina聚类完成，生成{len(clusters)}个簇',
                'labels': labels,
                'centers': None,
                'valid_indices': valid_indices,
                'clusters': clusters
            }
            
        elif method == 'kmeans':
            # 转换为特征矩阵
            features = np.array([list(fp) for fp in valid_fps], dtype=np.float32)
            
            # 获取配置参数
            kmeans_config = config.get('clustering', {}).get('kmeans', {})
            use_ratio = kmeans_config.get('use_ratio', True)
            subset_ratio = config.get('subset_ratio', 1.0)
            
            if use_ratio:
                n_clusters = int(len(features) * subset_ratio / 100)
                n_clusters = max(2, min(n_clusters, len(features) - 1))
            else:
                n_clusters = kmeans_config.get('n_clusters', 100)
            
            max_iter = kmeans_config.get('max_iter', 100)
            batch_size = config.get('data', {}).get('batching', {}).get('batch_size', 1000)
            
            # 执行K-means聚类
            labels, centers = kmeans_clustering(
                features, 
                n_clusters=n_clusters,
                batch_size=batch_size,
                max_iter=max_iter,
                config=config
            )
            
            return {
                'status': 'success',
                'message': f'K-means聚类完成，生成{n_clusters}个簇',
                'labels': labels,
                'centers': centers,
                'valid_indices': valid_indices
            }
            
        elif method == 'maxmin':
            # 获取配置参数
            subset_ratio = config.get('subset_ratio', 1.0)
            num_to_select = int(len(valid_fps) * subset_ratio / 100)
            num_to_select = max(2, min(num_to_select, len(valid_fps) - 1))
            
            init_method = config.get('clustering', {}).get('maxmin', {}).get('init_method', 'random')
            seed_idx = 0 if init_method == 'first' else -1
            
            # 执行MaxMin选择
            selected_indices = maxmin_selection(
                valid_fps,
                num_to_select=num_to_select,
                seed_idx=seed_idx,
                config=config
            )
            
            # 创建标签数组（-1表示未选择的点）
            labels = np.full(len(valid_fps), -1)
            for i, idx in enumerate(selected_indices):
                labels[idx] = i
            
            return {
                'status': 'success',
                'message': f'MaxMin选择完成，选择{len(selected_indices)}个代表点',
                'labels': labels,
                'centers': None,
                'valid_indices': valid_indices,
                'selected_indices': selected_indices
            }
        
        else:
            return {
                'status': 'error',
                'message': f'不支持的聚类方法: {method}',
                'labels': None,
                'centers': None,
                'valid_indices': valid_indices
            }
            
    except Exception as e:
        logger.error(f"聚类过程出错: {str(e)}")
        return {
            'status': 'error',
            'message': f'聚类过程出错: {str(e)}',
            'labels': None,
            'centers': None,
            'valid_indices': valid_indices
        }

def evaluate_clustering(features: np.ndarray, labels: np.ndarray, method: str) -> Dict[str, float]:
    """
    评估聚类结果的质量
    
    参数:
        features: 特征矩阵
        labels: 聚类标签
        method: 聚类方法
        
    返回:
        包含评估指标的字典
    """
    metrics = {}
    
    try:
        # 计算轮廓系数
        if len(np.unique(labels)) > 1:  # 至少有两个簇才能计算
            sil_score = silhouette_score(features, labels)
            metrics['silhouette_score'] = float(sil_score)
            
            # 计算Davies-Bouldin指数
            db_score = davies_bouldin_score(features, labels)
            metrics['davies_bouldin_score'] = float(db_score)
            
            # 计算Calinski-Harabasz指数
            ch_score = calinski_harabasz_score(features, labels)
            metrics['calinski_harabasz_score'] = float(ch_score)
        
        # 计算簇的统计信息
        unique_labels = np.unique(labels[labels >= 0])  # 排除噪声点（标签为-1）
        n_clusters = len(unique_labels)
        cluster_sizes = [np.sum(labels == label) for label in unique_labels]
        
        metrics.update({
            'n_clusters': n_clusters,
            'avg_cluster_size': float(np.mean(cluster_sizes)),
            'min_cluster_size': float(np.min(cluster_sizes)),
            'max_cluster_size': float(np.max(cluster_sizes)),
            'std_cluster_size': float(np.std(cluster_sizes))
        })
        
        # 方法特定的指标
        if method == 'kmeans':
            # 计算簇内距离平方和
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(features)
            metrics['inertia'] = float(kmeans.inertia_)
            
        elif method == 'butina':
            # 计算平均簇内相似度
            avg_intra_sim = 0.0
            n_pairs = 0
            
            for label in unique_labels:
                cluster_points = features[labels == label]
                if len(cluster_points) > 1:
                    for i in range(len(cluster_points)):
                        for j in range(i+1, len(cluster_points)):
                            sim = 1.0 - np.linalg.norm(cluster_points[i] - cluster_points[j])
                            avg_intra_sim += sim
                            n_pairs += 1
            
            if n_pairs > 0:
                avg_intra_sim /= n_pairs
                metrics['avg_intra_cluster_similarity'] = float(avg_intra_sim)
            
        elif method == 'maxmin':
            # 计算覆盖率
            selected_points = features[labels >= 0]
            if len(selected_points) > 0:
                coverage = np.sum(labels >= 0) / len(labels)
                metrics['coverage_ratio'] = float(coverage)
                
                # 计算平均最小距离
                min_distances = []
                for point in features:
                    min_dist = np.min([np.linalg.norm(point - sel_point) for sel_point in selected_points])
                    min_distances.append(min_dist)
                
                metrics.update({
                    'avg_min_distance': float(np.mean(min_distances)),
                    'max_min_distance': float(np.max(min_distances)),
                    'std_min_distance': float(np.std(min_distances))
                })
        
    except Exception as e:
        logger.error(f"评估聚类结果时出错: {str(e)}")
        metrics['error'] = str(e)
    
    return metrics

def cluster_quality_metrics(features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    计算聚类质量指标
    
    参数:
        features: 特征矩阵
        labels: 聚类标签
        
    返回:
        包含质量指标的字典
    """
    metrics = {}
    
    try:
        if len(np.unique(labels)) > 1:  # 至少有两个簇才能计算
            # 计算轮廓系数
            sil_score = silhouette_score(features, labels)
            metrics['silhouette_score'] = float(sil_score)
            
            # 计算Davies-Bouldin指数
            db_score = davies_bouldin_score(features, labels)
            metrics['davies_bouldin_score'] = float(db_score)
            
            # 计算Calinski-Harabasz指数
            ch_score = calinski_harabasz_score(features, labels)
            metrics['calinski_harabasz_score'] = float(ch_score)
            
            # 计算簇的统计信息
            unique_labels = np.unique(labels[labels >= 0])
            cluster_sizes = [np.sum(labels == label) for label in unique_labels]
            
            metrics.update({
                'n_clusters': len(unique_labels),
                'avg_cluster_size': float(np.mean(cluster_sizes)),
                'min_cluster_size': float(np.min(cluster_sizes)),
                'max_cluster_size': float(np.max(cluster_sizes)),
                'std_cluster_size': float(np.std(cluster_sizes))
            })
    
    except Exception as e:
        logger.error(f"计算聚类质量指标时出错: {str(e)}")
        metrics['error'] = str(e)
    
    return metrics 