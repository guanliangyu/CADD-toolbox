"""
子集验证工具 - 提供各种验证方法评估子集的代表性
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Union, Optional, Any

from rdkit import DataStructs
from rdkit import Chem
from sklearn.decomposition import PCA
from tqdm import tqdm

# Configure matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def calculate_nearest_neighbor_distance(query_fps: List[Any], 
                                      ref_fps: List[Any], 
                                      metric: str = 'tanimoto') -> Tuple[np.ndarray, float, float, float]:
    """
    计算查询集中每个分子到参考集的最近邻距离
    
    参数:
        query_fps: 查询分子的指纹列表
        ref_fps: 参考分子的指纹列表
        metric: 距离度量方法
        
    返回:
        (距离数组, 平均距离, 最大距离, 中位数距离)
    """
    n_query = len(query_fps)
    n_ref = len(ref_fps)
    
    nn_distances = []
    
    for i, q_fp in enumerate(tqdm(query_fps, desc='计算最近邻距离')):
        min_dist = float('inf')
        
        for j, r_fp in enumerate(ref_fps):
            if metric == 'tanimoto':
                try:
                    sim = DataStructs.TanimotoSimilarity(q_fp, r_fp)
                    d = 1.0 - sim
                except:
                    # 如果是NumPy数组
                    a_bits = np.array(q_fp, dtype=bool)
                    b_bits = np.array(r_fp, dtype=bool)
                    sim = np.sum(a_bits & b_bits) / np.sum(a_bits | b_bits)
                    d = 1.0 - sim
            elif metric == 'euclidean':
                d = np.linalg.norm(np.array(q_fp) - np.array(r_fp))
            else:
                raise ValueError(f"不支持的距离度量: {metric}")
                
            min_dist = min(min_dist, d)
            
        nn_distances.append(min_dist)
    
    nn_distances = np.array(nn_distances)
    
    # 计算统计值
    mean_dist = np.mean(nn_distances)
    max_dist = np.max(nn_distances)
    median_dist = np.median(nn_distances)
    
    return nn_distances, mean_dist, max_dist, median_dist


def plot_property_distributions(full_props: List[Dict[str, float]], 
                              subset_props: List[Dict[str, float]], 
                              prop_names: List[str],
                              figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Compare property distributions between full library and subset
    
    Args:
        full_props: List of property dictionaries for full library
        subset_props: List of property dictionaries for subset
        prop_names: List of property names to compare
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Determine subplot layout
    n_props = len(prop_names)
    n_cols = min(3, n_props)
    n_rows = (n_props + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_props == 1:
        axes = [axes]
    axes = axes.flatten() if hasattr(axes, 'flatten') else axes
    
    for i, prop in enumerate(prop_names):
        if i < len(axes):
            # Extract property values
            full_values = [p.get(prop, 0) for p in full_props if p and prop in p]
            subset_values = [p.get(prop, 0) for p in subset_props if p and prop in p]
            
            # Remove invalid values
            full_values = [v for v in full_values if not np.isnan(v) and not np.isinf(v)]
            subset_values = [v for v in subset_values if not np.isnan(v) and not np.isinf(v)]
            
            # Plot histograms
            ax = axes[i]
            sns.histplot(full_values, ax=ax, alpha=0.5, label='Full Library', color='blue')
            sns.histplot(subset_values, ax=ax, alpha=0.5, label='Subset', color='red')
            
            ax.set_title(f'{prop} Distribution')
            ax.set_xlabel(prop)
            ax.set_ylabel('Frequency')
            ax.legend()
            
            # Add statistics
            full_mean = np.mean(full_values)
            subset_mean = np.mean(subset_values)
            full_std = np.std(full_values)
            subset_std = np.std(subset_values)
            
            stats_text = (f"Full: μ={full_mean:.2f}, σ={full_std:.2f}\n"
                         f"Subset: μ={subset_mean:.2f}, σ={subset_std:.2f}")
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Hide unused subplots
    for i in range(n_props, len(axes)):
        axes[i].set_visible(False)
        
    plt.tight_layout()
    return fig


def plot_nearest_neighbor_histogram(nn_distances: np.ndarray, 
                                  figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot histogram of nearest neighbor distances
    
    Args:
        nn_distances: Array of nearest neighbor distances
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate statistics
    mean_dist = np.mean(nn_distances)
    max_dist = np.max(nn_distances)
    median_dist = np.median(nn_distances)
    
    # Plot histogram
    sns.histplot(nn_distances, ax=ax, kde=True)
    
    # Add mean and median lines
    ax.axvline(mean_dist, color='red', linestyle='--', label=f'Mean: {mean_dist:.3f}')
    ax.axvline(median_dist, color='green', linestyle='--', label=f'Median: {median_dist:.3f}')
    
    ax.set_title('Nearest Neighbor Distance Distribution')
    ax.set_xlabel('Distance to Nearest Representative')
    ax.set_ylabel('Frequency')
    ax.legend()
    
    # Add statistics text box
    stats_text = (f"Mean Distance: {mean_dist:.3f}\n"
                 f"Median Distance: {median_dist:.3f}\n"
                 f"Max Distance: {max_dist:.3f}")
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    return fig


def plot_pca_visualization(full_features: np.ndarray, 
                         subset_features: np.ndarray, 
                         n_components: int = 2,
                         figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Visualize full library and subset distribution in feature space using PCA
    
    Args:
        full_features: Feature matrix of full library
        subset_features: Feature matrix of subset
        n_components: Number of PCA components
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Combine features for PCA fitting
    combined_features = np.vstack([full_features, subset_features])
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    combined_pca = pca.fit_transform(combined_features)
    
    # Split PCA results
    full_pca = combined_pca[:len(full_features)]
    subset_pca = combined_pca[len(full_features):]
    
    # Create visualization
    fig = plt.figure(figsize=figsize)
    
    if n_components >= 3:
        # 3D scatter plot
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot full library (with sampling for performance)
        sample_size = min(5000, len(full_pca))
        sample_idx = np.random.choice(len(full_pca), sample_size, replace=False)
        ax.scatter(full_pca[sample_idx, 0], full_pca[sample_idx, 1], full_pca[sample_idx, 2], 
                  alpha=0.2, s=10, label='Full Library')
        
        # Plot subset
        ax.scatter(subset_pca[:, 0], subset_pca[:, 1], subset_pca[:, 2], 
                  color='red', s=20, label='Representative Subset')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
        
    else:
        # 2D scatter plot
        ax = fig.add_subplot(111)
        
        # Plot full library (with sampling for performance)
        sample_size = min(5000, len(full_pca))
        sample_idx = np.random.choice(len(full_pca), sample_size, replace=False)
        ax.scatter(full_pca[sample_idx, 0], full_pca[sample_idx, 1], 
                  alpha=0.2, s=10, label='Full Library')
        
        # Plot subset
        ax.scatter(subset_pca[:, 0], subset_pca[:, 1], 
                  color='red', s=20, label='Representative Subset')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    
    ax.set_title('PCA Visualization: Full Library vs Subset Coverage')
    ax.legend()
    
    plt.tight_layout()
    return fig


def calculate_coverage_metrics(full_fps: List[Any], subset_fps: List[Any], 
                             distance_threshold: float = 0.4,
                             metric: str = 'tanimoto') -> Dict[str, float]:
    """
    计算子集覆盖度评价指标
    
    参数:
        full_fps: 全库分子指纹列表
        subset_fps: 子集分子指纹列表
        distance_threshold: 距离阈值，小于该阈值认为已覆盖
        metric: 距离度量方法
        
    返回:
        覆盖度指标字典
    """
    # 计算全库中每个分子到子集最近分子的距离
    nn_distances, mean_dist, max_dist, median_dist = calculate_nearest_neighbor_distance(
        full_fps, subset_fps, metric=metric
    )
    
    # 计算处于阈值内的分子百分比
    covered_count = np.sum(nn_distances <= distance_threshold)
    coverage_ratio = covered_count / len(full_fps)
    
    # 计算有效覆盖半径（包含90%分子的最小距离）
    sorted_distances = np.sort(nn_distances)
    radius_90 = sorted_distances[int(0.9 * len(sorted_distances))]
    radius_95 = sorted_distances[int(0.95 * len(sorted_distances))]
    
    # 计算混合覆盖指标
    hybrid_score = coverage_ratio * (1 - mean_dist / max_dist)
    
    return {
        'coverage_ratio': coverage_ratio,
        'mean_distance': mean_dist,
        'max_distance': max_dist,
        'median_distance': median_dist,
        'radius_90': radius_90,
        'radius_95': radius_95,
        'hybrid_score': hybrid_score
    } 