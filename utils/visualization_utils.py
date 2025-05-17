"""
可视化工具模块 - 实现各种数据可视化功能
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from typing import List, Optional, Union, Tuple, Dict
import pandas as pd
import logging
import umap
import warnings

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 忽略特定警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Configure matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 设置后端为非交互式
plt.switch_backend('agg')

def plot_property_distributions(
    df: pd.DataFrame,
    properties: List[str],
    subset_df: Optional[pd.DataFrame] = None,
    n_cols: int = 2
) -> plt.Figure:
    """
    绘制分子属性的分布对比图
    
    参数:
        df: 原始数据集的DataFrame
        properties: 要绘制的属性列表
        subset_df: 子集的DataFrame（可选）
        n_cols: 每行的图形数量
        
    返回:
        matplotlib图形对象
    """
    try:
        # 设置绘图样式
        plt.style.use('seaborn')
        
        # 计算行数
        n_rows = (len(properties) + n_cols - 1) // n_cols
        
        # 创建子图
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.reshape(-1, 1) if n_cols == 1 else axes.reshape(1, -1)
        
        # 绘制每个属性的分布
        for idx, prop in enumerate(properties):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            try:
                # 绘制原始数据集的分布
                if prop in df.columns:
                    sns.histplot(
                        data=df,
                        x=prop,
                        stat="density",
                        alpha=0.5,
                        label="原始数据集",
                        ax=ax,
                        color="blue"
                    )
                    
                    # 如果有子集数据，绘制子集的分布
                    if subset_df is not None and prop in subset_df.columns:
                        sns.histplot(
                            data=subset_df,
                            x=prop,
                            stat="density",
                            alpha=0.5,
                            label="选择的子集",
                            ax=ax,
                            color="red"
                        )
                    
                    # 添加KDE曲线
                    sns.kdeplot(
                        data=df,
                        x=prop,
                        ax=ax,
                        color="blue",
                        linestyle="--"
                    )
                    if subset_df is not None and prop in subset_df.columns:
                        sns.kdeplot(
                            data=subset_df,
                            x=prop,
                            ax=ax,
                            color="red",
                            linestyle="--"
                        )
                    
                    # 设置图形属性
                    ax.set_title(f"{prop}的分布")
                    ax.legend()
                else:
                    ax.text(0.5, 0.5, f"属性 {prop} 不存在",
                           ha='center', va='center')
                    ax.set_axis_off()
            
            except Exception as e:
                logger.error(f"绘制属性 {prop} 的分布时出错: {str(e)}")
                ax.text(0.5, 0.5, f"绘制失败: {str(e)}",
                       ha='center', va='center')
                ax.set_axis_off()
        
        # 隐藏多余的子图
        for idx in range(len(properties), n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_axis_off()
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        logger.error(f"绘制属性分布图失败: {str(e)}")
        # 返回错误提示图
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"绘制属性分布图失败: {str(e)}",
                ha='center', va='center')
        return fig

def plot_fps_pca(
    features: Union[List[np.ndarray], np.ndarray],
    labels: np.ndarray,
    selected_indices: Optional[List[int]] = None,
    n_components: int = 2
) -> plt.Figure:
    """
    使用PCA降维并绘制聚类结果
    
    参数:
        features: 特征矩阵或特征向量列表
        labels: 聚类标签
        selected_indices: 选中点的索引
        n_components: PCA组分数量
        
    返回:
        matplotlib图形对象
    """
    try:
        # 设置绘图样式
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 转换特征为numpy数组
        if isinstance(features, list):
            features = np.array(features)
        
        # 执行PCA降维
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(features)
        
        # 计算解释方差比
        explained_var_ratio = pca.explained_variance_ratio_
        
        # 创建新图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        
        # 绘制每个簇的点
        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                reduced_features[mask, 0],
                reduced_features[mask, 1],
                alpha=0.6,
                label=f'簇 {label}'
            )
        
        # 标记选中的点
        if selected_indices is not None:
            ax.scatter(
                reduced_features[selected_indices, 0],
                reduced_features[selected_indices, 1],
                color='red',
                marker='*',
                s=200,
                alpha=0.8,
                label='选中的代表点'
            )
        
        # 设置图形属性
        ax.set_title('PCA投影的聚类结果')
        ax.set_xlabel(f'第一主成分 ({explained_var_ratio[0]:.1%})')
        ax.set_ylabel(f'第二主成分 ({explained_var_ratio[1]:.1%})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"PCA可视化失败: {str(e)}")
        # 返回错误提示图
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'PCA可视化失败: {str(e)}', 
                ha='center', va='center')
        return fig

def plot_fps_tsne(
    features: Union[List[np.ndarray], np.ndarray],
    labels: np.ndarray,
    selected_indices: Optional[List[int]] = None,
    perplexity: Optional[float] = None,
    n_iter: int = 1000
) -> plt.Figure:
    """
    使用t-SNE降维并绘制聚类结果
    
    参数:
        features: 特征矩阵或特征向量列表
        labels: 聚类标签
        selected_indices: 选中点的索引
        perplexity: t-SNE困惑度参数
        n_iter: 最大迭代次数
        
    返回:
        matplotlib图形对象
    """
    try:
        # 设置绘图样式
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 转换特征为numpy数组
        if isinstance(features, list):
            features = np.array(features)
        
        # 如果没有指定困惑度，根据数据量自动设置
        if perplexity is None:
            perplexity = min(30, max(5, len(features) // 100))
        
        # 执行t-SNE降维
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=42
        )
        reduced_features = tsne.fit_transform(features)
        
        # 创建新图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        
        # 绘制每个簇的点
        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                reduced_features[mask, 0],
                reduced_features[mask, 1],
                alpha=0.6,
                label=f'簇 {label}'
            )
        
        # 标记选中的点
        if selected_indices is not None:
            ax.scatter(
                reduced_features[selected_indices, 0],
                reduced_features[selected_indices, 1],
                color='red',
                marker='*',
                s=200,
                alpha=0.8,
                label='选中的代表点'
            )
        
        # 设置图形属性
        ax.set_title('t-SNE投影的聚类结果')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"t-SNE可视化失败: {str(e)}")
        # 返回错误提示图
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f't-SNE可视化失败: {str(e)}', 
                ha='center', va='center')
        return fig

def plot_fps_umap(
    features: Union[List[np.ndarray], np.ndarray],
    labels: np.ndarray,
    selected_indices: Optional[List[int]] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean'
) -> plt.Figure:
    """
    使用UMAP降维并绘制聚类结果
    
    参数:
        features: 特征矩阵或特征向量列表
        labels: 聚类标签
        selected_indices: 选中点的索引
        n_neighbors: UMAP的邻居数量参数
        min_dist: UMAP的最小距离参数
        metric: 距离度量方法
        
    返回:
        matplotlib图形对象
    """
    try:
        # 设置绘图样式
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 转换特征为numpy数组
        if isinstance(features, list):
            features = np.array(features)
        
        # 执行UMAP降维
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        reduced_features = reducer.fit_transform(features)
        
        # 创建新图形
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 获取唯一标签
        unique_labels = np.unique(labels)
        
        # 绘制每个簇的点
        for label in unique_labels:
            mask = labels == label
            ax.scatter(
                reduced_features[mask, 0],
                reduced_features[mask, 1],
                alpha=0.6,
                label=f'簇 {label}'
            )
        
        # 标记选中的点
        if selected_indices is not None:
            ax.scatter(
                reduced_features[selected_indices, 0],
                reduced_features[selected_indices, 1],
                color='red',
                marker='*',
                s=200,
                alpha=0.8,
                label='选中的代表点'
            )
        
        # 设置图形属性
        ax.set_title('UMAP投影的聚类结果')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        return fig
    
    except Exception as e:
        logger.error(f"UMAP可视化失败: {str(e)}")
        # 返回错误提示图
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f'UMAP可视化失败: {str(e)}', 
                ha='center', va='center')
        return fig

def plot_dimensionality_reduction(similarity_matrix: np.ndarray,
                                method: str = 'tsne',
                                perplexity: int = 30,
                                n_components: int = 2,
                                random_state: int = 42,
                                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Perform dimensionality reduction and plot results
    
    Args:
        similarity_matrix: Similarity matrix to reduce
        method: Reduction method ('tsne', 'umap', or 'pca')
        perplexity: t-SNE perplexity parameter
        n_components: Number of components for reduction
        random_state: Random seed
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    try:
        # Convert to numpy array if needed
        if not isinstance(similarity_matrix, np.ndarray):
            similarity_matrix = np.array(similarity_matrix)
            
        # Handle NaN values
        similarity_matrix = np.nan_to_num(similarity_matrix)
        
        # Perform dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, 
                         perplexity=min(perplexity, len(similarity_matrix) - 1),
                         random_state=random_state)
            embedding = reducer.fit_transform(similarity_matrix)
            method_name = 't-SNE'
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=n_components,
                              random_state=random_state)
            embedding = reducer.fit_transform(similarity_matrix)
            method_name = 'UMAP'
        else:  # PCA
            reducer = PCA(n_components=n_components)
            embedding = reducer.fit_transform(similarity_matrix)
            method_name = 'PCA'
        
        # Create visualization
        fig = plt.figure(figsize=figsize)
        
        if n_components >= 3:
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                               c=range(len(embedding)), cmap='viridis')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')
        else:
            ax = fig.add_subplot(111)
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                               c=range(len(embedding)), cmap='viridis')
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
        
        plt.colorbar(scatter, label='Molecule Index')
        ax.set_title(f'{method_name} Visualization')
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        logger.error(f"Dimensionality reduction failed: {str(e)}")
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'Visualization failed: {str(e)}',
                ha='center', va='center')
        return fig

def plot_similarity_heatmap(similarity_matrix: np.ndarray,
                          figsize: Tuple[int, int] = (10, 8),
                          cmap: str = 'viridis') -> plt.Figure:
    """
    Plot similarity matrix as heatmap
    
    Args:
        similarity_matrix: Similarity matrix to visualize
        figsize: Figure size
        cmap: Colormap to use
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(similarity_matrix, cmap=cmap)
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Similarity Score')
    
    ax.set_title('Molecular Similarity Heatmap')
    ax.set_xlabel('Molecule Index')
    ax.set_ylabel('Molecule Index')
    
    plt.tight_layout()
    return fig

def plot_similarity_distribution(similarity_matrix: np.ndarray,
                               figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot distribution of similarity scores
    
    Args:
        similarity_matrix: Similarity matrix
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Get upper triangle values (excluding diagonal)
    triu_indices = np.triu_indices(len(similarity_matrix), k=1)
    similarity_values = similarity_matrix[triu_indices]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    sns.histplot(similarity_values, kde=True, ax=ax)
    
    # Add statistics
    mean_sim = np.mean(similarity_values)
    median_sim = np.median(similarity_values)
    std_sim = np.std(similarity_values)
    
    stats_text = (f"Mean: {mean_sim:.3f}\n"
                 f"Median: {median_sim:.3f}\n"
                 f"Std Dev: {std_sim:.3f}")
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_title('Distribution of Similarity Scores')
    ax.set_xlabel('Similarity Score')
    ax.set_ylabel('Frequency')
    
    plt.tight_layout()
    return fig 