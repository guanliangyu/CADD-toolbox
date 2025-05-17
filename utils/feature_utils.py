"""
特征处理和降维工具 - 提供特征组合、降维和标准化功能
"""
import numpy as np
import logging
from typing import List, Dict, Tuple, Union, Optional, Any

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

# 引入GPU工具
try:
    from .gpu_utils import check_gpu_availability, gpu_pca
    GPU_TOOLS_AVAILABLE = True
except ImportError:
    GPU_TOOLS_AVAILABLE = False

# 设置日志
logger = logging.getLogger(__name__)

# 用于高维特征可视化的降维
class DimensionalityReducer:
    """特征降维类，提供多种降维方法"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化降维器
        
        参数:
            config: 配置字典，包含降维参数
        """
        self.config = config
        self.dr_config = config.get('features', {}).get('dimensionality_reduction', {})
        self.method = self.dr_config.get('method', 'pca')
        self.n_components = self.dr_config.get('n_components', 50)
        self.scaler_type = self.dr_config.get('scaler', 'standard')
        self.variance_ratio = self.dr_config.get('variance_ratio', 0.95)
        
        # 检查是否启用GPU加速
        self.use_gpu = False
        gpu_config = config.get('gpu', {})
        if gpu_config.get('enabled', False) and GPU_TOOLS_AVAILABLE:
            # 检查GPU可用性
            if gpu_config.get('auto_detect', True):
                gpu_status = check_gpu_availability()
                self.use_gpu = gpu_status['any_gpu'] and gpu_config.get('features', {}).get('pca', True)
            else:
                self.use_gpu = True
                
            if self.use_gpu:
                logger.info("降维将使用GPU加速")
            else:
                logger.info("降维将使用CPU")
        
        # 初始化缩放器
        self.scaler = self._get_scaler()
        # 初始化降维模型
        self.model = None
        
    def _get_scaler(self):
        """获取特征缩放器"""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'minmax':
            return MinMaxScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:
            return None
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        拟合特征并执行降维
        
        参数:
            features: 输入特征矩阵
            
        返回:
            降维后的特征矩阵
        """
        # 检查是否需要执行降维
        if not self.dr_config.get('enabled', True):
            return features
            
        # 先执行缩放
        if self.scaler is not None:
            features = self.scaler.fit_transform(features)
        
        # 执行降维
        if self.method == 'pca':
            return self._fit_transform_pca(features)
        elif self.method == 'umap':
            return self._fit_transform_umap(features)
        elif self.method == 'tsne':
            return self._fit_transform_tsne(features)
        else:
            logger.warning(f"不支持的降维方法: {self.method}，返回原始特征")
            return features
    
    def _fit_transform_pca(self, features: np.ndarray) -> np.ndarray:
        """PCA降维，支持GPU加速"""
        # 如果指定了方差比率，则重新确定组分数量
        if self.variance_ratio < 1.0:
            # 先创建一个PCA模型
            temp_pca = PCA(n_components=min(features.shape[1], features.shape[0]))
            temp_pca.fit(features)
            # 计算需要多少组分才能达到目标方差比率
            explained_var = np.cumsum(temp_pca.explained_variance_ratio_)
            # 找到第一个超过目标方差比率的索引
            n_components = np.argmax(explained_var >= self.variance_ratio) + 1
            self.n_components = min(n_components, self.n_components)
        
        # 根据GPU可用性选择PCA实现
        if self.use_gpu:
            try:
                # 使用GPU实现的PCA
                reduced_features, components, explained_variance_ratio = gpu_pca(
                    features, n_components=self.n_components
                )
                
                # 保存模型相关属性，用于后续transform
                class DummyPCA:
                    def __init__(self, components, explained_variance_ratio):
                        self.components_ = components
                        self.explained_variance_ratio_ = explained_variance_ratio
                        
                    def transform(self, X):
                        return X @ self.components_.T
                
                self.model = DummyPCA(components, explained_variance_ratio)
                
                logger.info(f"使用GPU进行PCA: 选择了{self.n_components}个主成分，解释了{sum(explained_variance_ratio):.2%}的方差")
                return reduced_features
            except Exception as e:
                logger.warning(f"GPU PCA失败: {e}，回退到CPU实现")
        
        # 使用CPU实现的PCA
        self.model = PCA(n_components=min(self.n_components, features.shape[1]))
        reduced_features = self.model.fit_transform(features)
        
        # 打印方差解释率
        explained_var_ratio = self.model.explained_variance_ratio_.sum()
        logger.info(f"CPU PCA降维: 选择了{self.n_components}个主成分，解释了{explained_var_ratio:.2%}的方差")
        
        return reduced_features
    
    def _fit_transform_umap(self, features: np.ndarray) -> np.ndarray:
        """UMAP降维"""
        # 注意: 目前UMAP不支持GPU，但未来可能会支持
        self.model = umap.UMAP(n_components=self.n_components, random_state=42)
        return self.model.fit_transform(features)
    
    def _fit_transform_tsne(self, features: np.ndarray) -> np.ndarray:
        """t-SNE降维 (通常用于可视化，而不是实际降维)"""
        self.model = TSNE(n_components=min(self.n_components, 3), random_state=42)
        return self.model.fit_transform(features)
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        使用已拟合的模型对新特征进行转换
        
        参数:
            features: 输入特征矩阵
            
        返回:
            降维后的特征矩阵
        """
        if not self.dr_config.get('enabled', True) or self.model is None:
            return features
            
        # 先执行缩放
        if self.scaler is not None:
            features = self.scaler.transform(features)
            
        # 执行降维
        if hasattr(self.model, 'transform'):
            return self.model.transform(features)
        else:
            # 对于不支持transform的模型（如t-SNE），再次执行fit_transform
            return self.model.fit_transform(features)


class FeatureCombiner:
    """特征组合类，用于将多种特征组合成单一特征向量"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化特征组合器
        
        参数:
            config: 配置字典，包含特征计算参数
        """
        self.config = config
        self.features_config = config.get('features', {})
        
        # 检查是否启用GPU加速
        self.use_gpu = False
        gpu_config = config.get('gpu', {})
        if gpu_config.get('enabled', False) and GPU_TOOLS_AVAILABLE:
            # 检查GPU可用性
            if gpu_config.get('auto_detect', True):
                gpu_status = check_gpu_availability()
                self.use_gpu = gpu_status['any_gpu']
            else:
                self.use_gpu = True
                
            if self.use_gpu:
                logger.info("特征处理将使用GPU加速")
            else:
                logger.info("特征处理将使用CPU")
        
    def combine_features(self, fp_features: Optional[List[np.ndarray]] = None, 
                        basic_features: Optional[List[Dict[str, float]]] = None,
                        shape_features: Optional[List[Dict[str, float]]] = None,
                        charges_features: Optional[List[List[float]]] = None) -> np.ndarray:
        """
        组合多种特征成为单一特征矩阵
        
        参数:
            fp_features: 指纹特征列表，每个元素是一个分子的指纹NumPy数组
            basic_features: 基本描述符字典列表，每个元素是一个分子的描述符字典
            shape_features: 形状描述符字典列表
            charges_features: 电荷特征列表，每个元素是一个分子的原子电荷列表
            
        返回:
            组合后的特征矩阵，每行代表一个分子
        """
        # 确定分子数量（取第一个非空特征的长度）
        n_mols = 0
        if fp_features is not None and fp_features:
            n_mols = len(fp_features)
        elif basic_features is not None and basic_features:
            n_mols = len(basic_features)
        elif shape_features is not None and shape_features:
            n_mols = len(shape_features)
        elif charges_features is not None and charges_features:
            n_mols = len(charges_features)
            
        if n_mols == 0:
            raise ValueError("所有特征列表都为空或None")
            
        # 初始化用于存放组合特征的列表
        combined_features = []
        
        # 处理每个分子
        for i in range(n_mols):
            mol_features = []
            
            # 添加指纹特征
            if fp_features is not None and i < len(fp_features) and fp_features[i] is not None:
                mol_features.extend(fp_features[i])
                
            # 添加基本描述符特征
            if basic_features is not None and i < len(basic_features) and basic_features[i]:
                # 获取配置中指定的描述符
                props_config = self.features_config.get('properties', {})
                if props_config.get('enabled', True):
                    selected_props = props_config.get('descriptors', [])
                    for prop in selected_props:
                        if prop in basic_features[i]:
                            mol_features.append(basic_features[i][prop])
                        else:
                            mol_features.append(0.0)  # 如果没有该描述符，填充0
                            
            # 添加形状描述符特征
            if shape_features is not None and i < len(shape_features) and shape_features[i]:
                shape_config = self.features_config.get('shape', {})
                if shape_config.get('enabled', True):
                    selected_shape = shape_config.get('descriptors', [])
                    for shape_prop in selected_shape:
                        if shape_prop == 'moments' and 'principal_moment_1' in shape_features[i]:
                            mol_features.append(shape_features[i]['principal_moment_1'])
                            mol_features.append(shape_features[i]['principal_moment_2'])
                            mol_features.append(shape_features[i]['principal_moment_3'])
                        elif shape_prop in shape_features[i]:
                            mol_features.append(shape_features[i][shape_prop])
                        else:
                            mol_features.append(0.0)  # 如果没有该描述符，填充0
                            
            # 添加电荷特征
            if charges_features is not None and i < len(charges_features) and charges_features[i]:
                elec_config = self.features_config.get('electrostatics', {})
                if elec_config.get('enabled', True):
                    selected_elec = elec_config.get('descriptors', [])
                    charges = charges_features[i]
                    
                    if 'charges_stats' in selected_elec:
                        # 计算电荷统计量
                        if charges:
                            mol_features.append(np.mean(charges))  # 平均电荷
                            mol_features.append(np.std(charges))   # 电荷标准差
                            mol_features.append(np.min(charges))   # 最小电荷
                            mol_features.append(np.max(charges))   # 最大电荷
                        else:
                            mol_features.extend([0.0, 0.0, 0.0, 0.0])
            
            # 将分子特征添加到组合特征列表中
            combined_features.append(mol_features)
            
        # 将列表转换为NumPy数组
        # 注意可能需要填充，因为不同分子的特征长度可能不同
        max_len = max(len(f) for f in combined_features)
        padded_features = []
        for f in combined_features:
            if len(f) < max_len:
                padded_features.append(f + [0.0] * (max_len - len(f)))
            else:
                padded_features.append(f)
        
        # 转换为numpy数组
        features_array = np.array(padded_features)
        
        # 如果使用GPU加速，可以在这里处理数据
        if self.use_gpu:
            # 这里我们不直接转到GPU，因为后续操作会根据需要处理
            logger.info(f"组合特征完成: 形状 {features_array.shape}")
        else:
            logger.info(f"组合特征完成: 形状 {features_array.shape}")
            
        return features_array
    
    
class ReferenceSetMapper:
    """参考集映射器，用于将指纹映射到参考空间"""
    
    def __init__(self, reference_fps: List[Any], metric: str = 'tanimoto'):
        """
        初始化参考集映射器
        
        参数:
            reference_fps: 参考分子指纹列表
            metric: 相似度度量方法
        """
        self.reference_fps = reference_fps
        self.metric = metric
        
        # 检查GPU可用性
        self.use_gpu = False
        if GPU_TOOLS_AVAILABLE:
            gpu_status = check_gpu_availability()
            self.use_gpu = gpu_status['any_gpu'] and (
                gpu_status['faiss_gpu'] or gpu_status['cupy']
            )
            
            if self.use_gpu:
                logger.info("参考集映射将使用GPU加速")
                
                # 预处理参考指纹，转换为适合GPU的格式
                if gpu_status['faiss_gpu']:
                    try:
                        import faiss
                        # 如果是二进制指纹，转换为float32数组
                        if hasattr(reference_fps[0], '__len__'):
                            self.reference_array = np.array([list(fp) for fp in reference_fps], dtype=np.float32)
                            # 创建FAISS索引
                            self.index = faiss.IndexFlatL2(self.reference_array.shape[1])
                            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)
                            self.index.add(self.reference_array)
                            logger.info(f"FAISS GPU索引已创建: {self.reference_array.shape}")
                    except Exception as e:
                        logger.warning(f"创建FAISS GPU索引失败: {e}")
                        self.use_gpu = False
        
    def map_to_reference_space(self, fps: List[Any]) -> np.ndarray:
        """
        将指纹映射到参考空间（每个维度是与参考分子的相似度）
        
        参数:
            fps: 待映射的指纹列表
            
        返回:
            映射后的特征矩阵，每行是一个分子与所有参考分子的相似度
        """
        n_mols = len(fps)
        n_refs = len(self.reference_fps)
        
        # 初始化结果矩阵
        similarity_matrix = np.zeros((n_mols, n_refs), dtype=np.float32)
        
        # 如果使用GPU加速且为FAISS格式
        if self.use_gpu and hasattr(self, 'index'):
            try:
                # 转换查询指纹为float32数组
                query_array = np.array([list(fp) for fp in fps], dtype=np.float32)
                
                # 如果是Tanimoto相似度，需要预处理数据
                if self.metric == 'tanimoto':
                    # 对于Tanimoto相似度，先执行归一化
                    import faiss
                    faiss.normalize_L2(query_array)
                    faiss.normalize_L2(self.reference_array)
                    
                    # 使用内积计算相似度（归一化后内积~相似度）
                    _, distances = self.index.search(query_array, n_refs)
                    # 转换距离为相似度
                    similarity_matrix = 1.0 - distances / 2.0
                else:
                    # 对于其他度量，直接计算距离
                    _, distances = self.index.search(query_array, n_refs)
                    # 转换距离为相似度（这里简单处理，实际可能需要更复杂的转换）
                    max_dist = np.max(distances)
                    similarity_matrix = 1.0 - distances / max_dist
                
                return similarity_matrix
            except Exception as e:
                logger.warning(f"GPU参考集映射失败: {e}，回退到CPU计算")
        
        # CPU计算
        for i, fp in enumerate(fps):
            for j, ref_fp in enumerate(self.reference_fps):
                if self.metric == 'tanimoto':
                    try:
                        sim = DataStructs.TanimotoSimilarity(fp, ref_fp)
                    except:
                        # 如果是NumPy数组
                        a_bits = np.array(fp, dtype=bool)
                        b_bits = np.array(ref_fp, dtype=bool)
                        sim = np.sum(a_bits & b_bits) / np.sum(a_bits | b_bits)
                else:
                    raise ValueError(f"不支持的相似度度量: {self.metric}")
                    
                similarity_matrix[i, j] = sim
                
        return similarity_matrix 