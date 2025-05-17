"""
GPU加速工具模块 - 提供GPU相关功能和检测
"""
import os
import logging
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any, Callable

# 设置日志
logger = logging.getLogger(__name__)

# 全局变量，用于记录GPU可用性
_GPU_AVAILABLE = None
_FAISS_GPU_AVAILABLE = None
_CUML_AVAILABLE = None
_TORCH_AVAILABLE = None
_CUPY_AVAILABLE = None

def check_gpu_availability() -> Dict[str, bool]:
    """
    检查各种GPU库的可用性
    
    返回:
        包含各GPU库可用性的字典
    """
    global _GPU_AVAILABLE, _FAISS_GPU_AVAILABLE, _CUML_AVAILABLE, _TORCH_AVAILABLE, _CUPY_AVAILABLE
    
    if _GPU_AVAILABLE is not None:
        return {
            'any_gpu': _GPU_AVAILABLE,
            'faiss_gpu': _FAISS_GPU_AVAILABLE,
            'cuml': _CUML_AVAILABLE,
            'torch': _TORCH_AVAILABLE,
            'cupy': _CUPY_AVAILABLE
        }
    
    # 默认假设不可用
    _GPU_AVAILABLE = False
    _FAISS_GPU_AVAILABLE = False
    _CUML_AVAILABLE = False
    _TORCH_AVAILABLE = False
    _CUPY_AVAILABLE = False
    
    # 检查CUDA环境变量
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible == '-1':
        logger.info('CUDA_VISIBLE_DEVICES设置为-1，禁用GPU')
        return {
            'any_gpu': False,
            'faiss_gpu': False,
            'cuml': False,
            'torch': False,
            'cupy': False
        }
    
    # 检查FAISS GPU
    try:
        import faiss
        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            _FAISS_GPU_AVAILABLE = True
            _GPU_AVAILABLE = True
            logger.info(f"FAISS-GPU可用，GPU数量: {faiss.get_num_gpus()}")
        else:
            logger.info("FAISS GPU不可用")
    except ImportError:
        logger.info("FAISS未安装")
    except Exception as e:
        logger.warning(f"检查FAISS GPU时出错: {e}")
    
    # 检查CUML
    try:
        import cuml
        _CUML_AVAILABLE = True
        _GPU_AVAILABLE = True
        logger.info("CUML (GPU机器学习库) 可用")
    except ImportError:
        logger.info("CUML未安装")
    except Exception as e:
        logger.warning(f"检查CUML时出错: {e}")
    
    # 检查PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            _TORCH_AVAILABLE = True
            _GPU_AVAILABLE = True
            logger.info(f"PyTorch GPU可用，设备: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("PyTorch GPU不可用")
    except ImportError:
        logger.info("PyTorch未安装")
    except Exception as e:
        logger.warning(f"检查PyTorch GPU时出错: {e}")
    
    # 检查CuPy
    try:
        import cupy as cp
        _CUPY_AVAILABLE = True
        _GPU_AVAILABLE = True
        logger.info(f"CuPy可用，CUDA版本: {cp.cuda.runtime.runtimeGetVersion()}")
    except ImportError:
        logger.info("CuPy未安装")
    except Exception as e:
        logger.warning(f"检查CuPy时出错: {e}")
    
    return {
        'any_gpu': _GPU_AVAILABLE,
        'faiss_gpu': _FAISS_GPU_AVAILABLE,
        'cuml': _CUML_AVAILABLE,
        'torch': _TORCH_AVAILABLE,
        'cupy': _CUPY_AVAILABLE
    }


def init_faiss_gpu(gpu_id: int = 0) -> None:
    """
    初始化FAISS GPU资源
    
    参数:
        gpu_id: 要使用的GPU ID
    """
    if not _FAISS_GPU_AVAILABLE:
        logger.warning("FAISS GPU不可用，跳过初始化")
        return
    
    try:
        import faiss
        res = faiss.StandardGpuResources()  # 默认GPU资源
        logger.info(f"FAISS GPU资源已初始化，使用GPU {gpu_id}")
        return res
    except Exception as e:
        logger.warning(f"初始化FAISS GPU时出错: {e}")
        return None


def copy_to_gpu(data: np.ndarray) -> Any:
    """
    将NumPy数组复制到GPU内存
    
    参数:
        data: NumPy数组
        
    返回:
        GPU上的数组（取决于可用库）
    """
    if _TORCH_AVAILABLE:
        import torch
        return torch.from_numpy(data).cuda()
    elif _CUPY_AVAILABLE:
        import cupy as cp
        return cp.array(data)
    else:
        logger.warning("没有可用的GPU库来复制数据")
        return data


def copy_to_cpu(data: Any) -> np.ndarray:
    """
    将GPU数组复制回CPU内存
    
    参数:
        data: GPU上的数组
        
    返回:
        NumPy数组
    """
    if hasattr(data, 'cpu') and callable(getattr(data, 'cpu')):  # PyTorch Tensor
        return data.cpu().numpy()
    elif hasattr(data, 'get') and callable(getattr(data, 'get')):  # CuPy Array
        return data.get()
    else:
        return np.array(data)


class GPUKMeans:
    """使用GPU加速的K-means聚类实现"""
    
    def __init__(self, n_clusters: int, random_state: int = 42, max_iter: int = 100):
        """
        初始化GPU K-means聚类器
        
        参数:
            n_clusters: 聚类数量
            random_state: 随机种子
            max_iter: 最大迭代次数
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.cluster_centers_ = None
        self.labels_ = None
        self._backend = None
        
        # 选择可用的后端
        if _CUML_AVAILABLE:
            try:
                import cuml.cluster
                self._backend = 'cuml'
                self._model = cuml.cluster.KMeans(
                    n_clusters=n_clusters,
                    random_state=random_state,
                    max_iter=max_iter
                )
                logger.info("使用CUML后端进行K-means聚类")
            except:
                self._backend = None
        
        if self._backend is None and _FAISS_GPU_AVAILABLE:
            try:
                import faiss
                self._backend = 'faiss'
                self._res = faiss.StandardGpuResources()
                logger.info("使用FAISS GPU后端进行K-means聚类")
            except:
                self._backend = None
                
        if self._backend is None:
            from sklearn.cluster import KMeans
            self._backend = 'sklearn'
            self._model = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                max_iter=max_iter
            )
            logger.info("没有可用的GPU后端，回退到CPU K-means聚类")
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        拟合数据并预测聚类标签
        
        参数:
            X: 特征矩阵，shape为(n_samples, n_features)
            
        返回:
            聚类标签数组
        """
        if self._backend == 'cuml':
            # CUML后端
            self._model.fit(X)
            self.cluster_centers_ = copy_to_cpu(self._model.cluster_centers_)
            self.labels_ = copy_to_cpu(self._model.labels_)
        
        elif self._backend == 'faiss':
            # FAISS后端
            import faiss
            
            n_samples, n_features = X.shape
            X_float32 = X.astype(np.float32)
            
            # 初始化聚类器
            kmeans = faiss.Kmeans(
                n_features, 
                self.n_clusters, 
                niter=self.max_iter, 
                gpu=True, 
                seed=self.random_state
            )
            
            # 执行聚类
            kmeans.train(X_float32)
            
            # 获取结果
            self.cluster_centers_ = kmeans.centroids
            
            # 计算标签
            _, self.labels_ = kmeans.index.search(X_float32, 1)
            self.labels_ = self.labels_.reshape(-1)
        
        else:
            # Scikit-learn后端
            self.labels_ = self._model.fit_predict(X)
            self.cluster_centers_ = self._model.cluster_centers_
        
        return self.labels_


def gpu_pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用GPU加速的PCA实现
    
    参数:
        X: 输入数据矩阵
        n_components: 主成分数量
        
    返回:
        (降维后的数据, 主成分, 解释方差比)
    """
    if _CUML_AVAILABLE:
        try:
            import cuml
            
            # 使用CUML的PCA
            pca = cuml.PCA(n_components=n_components)
            reduced_data = pca.fit_transform(X)
            
            # 转换回CPU
            reduced_data_cpu = copy_to_cpu(reduced_data)
            components_cpu = copy_to_cpu(pca.components_)
            explained_variance_ratio_cpu = copy_to_cpu(pca.explained_variance_ratio_)
            
            logger.info(f"使用CUML GPU完成PCA降维，解释方差: {sum(explained_variance_ratio_cpu):.2%}")
            return reduced_data_cpu, components_cpu, explained_variance_ratio_cpu
        except Exception as e:
            logger.warning(f"GPU PCA失败: {e}，回退到CPU PCA")
    
    # 回退到CPU PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(X)
    
    logger.info(f"使用CPU完成PCA降维，解释方差: {sum(pca.explained_variance_ratio_):.2%}")
    return reduced_data, pca.components_, pca.explained_variance_ratio_


def gpu_distance_matrix(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    使用GPU计算距离矩阵
    
    参数:
        X: 输入数据矩阵，形状为(n_samples, n_features)
        metric: 距离度量方法
        
    返回:
        距离矩阵，形状为(n_samples, n_samples)
    """
    if metric == 'euclidean':
        # 尝试使用FAISS计算欧几里得距离
        if _FAISS_GPU_AVAILABLE:
            try:
                import faiss
                
                n_samples = X.shape[0]
                X_float32 = X.astype(np.float32)
                
                # 创建GPU资源
                res = faiss.StandardGpuResources()
                
                # 创建内积内积计算器
                inner_product = faiss.GpuDistanceToIpMetric(
                    res, X_float32, faiss.METRIC_L2, True, 256
                )
                
                # 计算L2距离
                distances = np.zeros((n_samples, n_samples), dtype=np.float32)
                batch_size = 1024  # 可根据GPU内存调整
                
                for i in range(0, n_samples, batch_size):
                    end_i = min(i + batch_size, n_samples)
                    batch_distances = inner_product.pairwise_distances(X_float32[i:end_i])
                    distances[i:end_i] = np.sqrt(batch_distances)
                
                logger.info(f"使用FAISS GPU完成{n_samples}x{n_samples}距离矩阵计算")
                return distances
            except Exception as e:
                logger.warning(f"GPU距离矩阵计算失败: {e}，回退到其他方法")
    
    # 尝试使用CuPy
    if _CUPY_AVAILABLE and metric == 'euclidean':
        try:
            import cupy as cp
            from cupyx.scipy.spatial.distance import cdist
            
            # 转移数据到GPU
            X_gpu = cp.array(X.astype(np.float32))
            
            # 计算距离矩阵
            distances_gpu = cdist(X_gpu, X_gpu, metric=metric)
            
            # 转回CPU
            distances = cp.asnumpy(distances_gpu)
            
            logger.info(f"使用CuPy完成{X.shape[0]}x{X.shape[0]}距离矩阵计算")
            return distances
        except Exception as e:
            logger.warning(f"CuPy距离矩阵计算失败: {e}，回退到CPU计算")
    
    # 回退到CPU计算
    from scipy.spatial.distance import pdist, squareform
    
    if metric == 'euclidean':
        # 使用高效的SciPy实现
        pairwise_dists = pdist(X, metric='euclidean')
        distances = squareform(pairwise_dists)
    else:
        # 自定义距离度量
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if metric == 'tanimoto':
                    # 二进制向量的Tanimoto距离
                    a = X[i].astype(bool)
                    b = X[j].astype(bool)
                    intersection = np.sum(a & b)
                    union = np.sum(a | b)
                    sim = intersection / union if union > 0 else 0
                    d = 1.0 - sim
                else:
                    # 默认使用欧几里得距离
                    d = np.linalg.norm(X[i] - X[j])
                
                distances[i, j] = d
                distances[j, i] = d
    
    logger.info(f"使用CPU完成{X.shape[0]}x{X.shape[0]}距离矩阵计算")
    return distances 