#!/usr/bin/env python
"""
分子库代表性子集选择主脚本
处理流程：
1. 数据读取和过滤
2. 分子处理和特征计算
3. 特征降维
4. 聚类和子集选择
5. 验证和评估
"""
import os
import sys
import time
import argparse
import yaml
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import pickle
import datetime

# 添加项目根目录到路径，确保能导入utils模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

from utils.molecular_utils import MoleculeProcessor
from utils.clustering_utils import (
    butina_clustering, kmeans_clustering, maxmin_selection, 
    hdbscan_clustering, select_cluster_representatives,
    select_representatives_from_kmeans
)
from utils.feature_utils import DimensionalityReducer, FeatureCombiner
from utils.validation_utils import (
    calculate_nearest_neighbor_distance, plot_property_distributions,
    plot_nearest_neighbor_histogram, plot_pca_visualization,
    calculate_coverage_metrics
)

# 尝试导入GPU工具
try:
    from utils.gpu_utils import check_gpu_availability
    GPU_TOOLS_AVAILABLE = True
except ImportError:
    GPU_TOOLS_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('molecular_subset.log')
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_file):
    """
    加载YAML配置文件
    
    参数:
        config_file: 配置文件路径
        
    返回:
        配置字典
    """
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_data(input_file):
    """
    从文件加载分子数据
    
    参数:
        input_file: 输入文件路径
        
    返回:
        包含分子数据的DataFrame
    """
    ext = os.path.splitext(input_file)[1].lower()
    
    if ext == '.csv':
        df = pd.read_csv(input_file)
    elif ext == '.sdf':
        df = PandasTools.LoadSDF(input_file)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
        
    logger.info(f"加载了 {len(df)} 条记录")
    return df


def process_molecule_batch(batch_data, config, include_3d=True, include_charges=True):
    """
    处理一批分子并计算特征
    
    参数:
        batch_data: (smiles_list, indices) 元组
        config: 配置字典
        include_3d: 是否生成3D构象
        include_charges: 是否计算电荷
        
    返回:
        处理结果字典
    """
    smiles_list, indices = batch_data
    processor = MoleculeProcessor(config)
    
    # 结果容器
    results = {
        'indices': indices,
        'mols': [],
        'fps': [],
        'basic_desc': [],
        'shape_desc': [],
        'charges': []
    }
    
    for smiles in smiles_list:
        # 1. 准备分子
        mol = processor.prepare_molecule(smiles)
        
        if mol is None:
            # 记录空值
            results['mols'].append(None)
            results['fps'].append(None)
            results['basic_desc'].append({})
            results['shape_desc'].append({})
            results['charges'].append(None)
            continue
            
        # 2. 计算指纹
        fp_config = config.get('features', {}).get('fingerprints', {})
        fp_type = fp_config.get('types', ['morgan'])[0]
        radius = fp_config.get('morgan_radius', 2)
        n_bits = fp_config.get('morgan_bits', 1024)
        
        fp = processor.compute_fingerprint(mol, radius=radius, nBits=n_bits, fp_type=fp_type)
        fp_array = processor.fp_to_numpy(fp)
        
        # 3. 计算基本描述符
        basic_desc = processor.compute_basic_descriptors(mol)
        
        # 4. 可选：生成3D构象
        mol_3d = None
        shape_desc = {}
        charges = None
        
        if include_3d:
            mol_3d = processor.generate_3d_conformer(mol)
            
            if mol_3d:
                # 计算形状描述符
                shape_desc = processor.compute_shape_descriptors(mol_3d)
                
                # 可选：计算电荷
                if include_charges:
                    charges = processor.compute_gasteiger_charges(mol_3d)
        
        # 添加到结果
        results['mols'].append(mol)
        results['fps'].append(fp_array)
        results['basic_desc'].append(basic_desc)
        results['shape_desc'].append(shape_desc)
        results['charges'].append(charges)
    
    return results


def batch_process_molecules(df, config, smiles_col='SMILES'):
    """
    批量处理分子
    
    参数:
        df: 包含分子数据的DataFrame
        config: 配置字典
        smiles_col: SMILES列名
        
    返回:
        处理结果字典
    """
    batch_size = config.get('data', {}).get('batching', {}).get('batch_size', 1000)
    n_jobs = config.get('data', {}).get('batching', {}).get('n_jobs', -1)
    
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    n_jobs = min(mp.cpu_count(), n_jobs)
    
    include_3d = config.get('data', {}).get('conformers', {}).get('enabled', True)
    include_charges = config.get('data', {}).get('charges', {}).get('enabled', True)
    
    logger.info(f"开始批量处理分子: 批量大小={batch_size}, CPU核心数={n_jobs}, 3D={include_3d}, 电荷={include_charges}")
    
    # 准备批次数据
    smiles_list = df[smiles_col].tolist()
    total = len(smiles_list)
    batches = []
    
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batches.append((smiles_list[i:end], list(range(i, end))))
    
    # 启动并行处理
    results = {
        'mols': [None] * total,
        'fps': [None] * total,
        'basic_desc': [{}] * total,
        'shape_desc': [{}] * total,
        'charges': [None] * total
    }
    
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for batch_result in tqdm(
            executor.map(lambda x: process_molecule_batch(x, config, include_3d, include_charges), batches),
            total=len(batches),
            desc="处理分子批次"
        ):
            # 合并结果
            indices = batch_result['indices']
            for i, idx in enumerate(indices):
                results['mols'][idx] = batch_result['mols'][i]
                results['fps'][idx] = batch_result['fps'][i]
                results['basic_desc'][idx] = batch_result['basic_desc'][i]
                results['shape_desc'][idx] = batch_result['shape_desc'][i]
                results['charges'][idx] = batch_result['charges'][i]
    
    duration = time.time() - start_time
    
    # 统计有效分子数量
    valid_count = sum(1 for m in results['mols'] if m is not None)
    logger.info(f"分子处理完成: 总计={total}, 有效={valid_count}, 耗时={duration:.1f}秒")
    
    return results


def combine_features_and_reduce(processed_results, config):
    """
    组合特征并进行降维
    
    参数:
        processed_results: 处理结果字典
        config: 配置字典
        
    返回:
        (原始组合特征, 降维后特征)
    """
    logger.info("组合特征并降维")
    
    feature_combiner = FeatureCombiner(config)
    
    # 合并特征
    combined_features = feature_combiner.combine_features(
        fp_features=processed_results['fps'],
        basic_features=processed_results['basic_desc'],
        shape_features=processed_results['shape_desc'],
        charges_features=processed_results['charges']
    )
    
    # 降维
    reducer = DimensionalityReducer(config)
    reduced_features = reducer.fit_transform(combined_features)
    
    # 记录特征维度
    logger.info(f"特征维度: 原始={combined_features.shape}, 降维后={reduced_features.shape}")
    
    return combined_features, reduced_features


def select_representative_subset(features, mols, config):
    """
    选择代表性子集
    
    参数:
        features: 特征矩阵
        mols: 分子列表
        config: 配置字典
        
    返回:
        代表性分子的索引列表
    """
    clustering_config = config.get('clustering', {})
    method = clustering_config.get('method', 'butina')
    
    logger.info(f"使用 {method} 方法选择代表性子集")
    
    # 移除无效分子
    valid_idx = [i for i, m in enumerate(mols) if m is not None]
    valid_features = features[valid_idx]
    
    # 计算目标选择数量（默认约1%）
    total_valid = len(valid_idx)
    target_count = int(total_valid * 0.01)  # 默认1%
    
    # 根据聚类方法选择代表
    if method == 'butina':
        # Butina算法需要指纹，而不是降维后的特征
        cutoff = clustering_config.get('butina', {}).get('cutoff', 0.4)
        valid_fps = [processor.fp_to_numpy(mols[i]) for i in valid_idx]
        clusters = butina_clustering(valid_fps, cutoff=cutoff)
        
        # 选择每个簇的代表分子
        rep_indices = select_cluster_representatives(clusters, valid_fps, method='centroid')
        
        # 将局部索引映射回全局索引
        global_indices = [valid_idx[i] for i in rep_indices]
        
    elif method == 'kmeans':
        # K-means直接使用降维后的特征
        n_clusters = clustering_config.get('kmeans', {}).get('n_clusters', target_count)
        batch_size = clustering_config.get('kmeans', {}).get('batch_size', 1000)
        max_iter = clustering_config.get('kmeans', {}).get('max_iter', 100)
        
        labels, centers = kmeans_clustering(
            valid_features, n_clusters=n_clusters, 
            batch_size=batch_size, max_iter=max_iter
        )
        
        # 选择最接近簇中心的分子
        rep_local_indices = select_representatives_from_kmeans(valid_features, labels, centers)
        
        # 映射回全局索引
        global_indices = [valid_idx[i] for i in rep_local_indices]
        
    elif method == 'maxmin':
        # MaxMin选择器
        num_to_select = target_count
        distance_metric = clustering_config.get('maxmin', {}).get('distance_measure', 'euclidean')
        
        # MaxMin直接返回选择的索引
        rep_local_indices = maxmin_selection(
            valid_features, num_to_select=num_to_select, 
            distance_metric=distance_metric
        )
        
        # 映射回全局索引
        global_indices = [valid_idx[i] for i in rep_local_indices]
        
    elif method == 'hdbscan':
        # 基于密度的聚类
        min_cluster_size = clustering_config.get('hdbscan', {}).get('min_cluster_size', 5)
        min_samples = clustering_config.get('hdbscan', {}).get('min_samples', 5)
        
        labels = hdbscan_clustering(
            valid_features, 
            min_cluster_size=min_cluster_size,
            min_samples=min_samples
        )
        
        # 计算簇数量
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"HDBSCAN聚类: 发现{n_clusters}个簇")
        
        # 为每个簇选择代表
        rep_local_indices = []
        for cluster_id in set(labels):
            if cluster_id == -1:  # 噪声点
                continue
                
            # 找出属于当前簇的所有点
            cluster_points = np.where(labels == cluster_id)[0]
            
            # 选择簇中心点（到其他点平均距离最小的点）
            if len(cluster_points) == 1:
                rep_local_indices.append(cluster_points[0])
            else:
                cluster_features = valid_features[cluster_points]
                center = np.mean(cluster_features, axis=0)
                distances = np.linalg.norm(cluster_features - center, axis=1)
                closest_idx = cluster_points[np.argmin(distances)]
                rep_local_indices.append(closest_idx)
                
        # 映射回全局索引
        global_indices = [valid_idx[i] for i in rep_local_indices]
    else:
        raise ValueError(f"不支持的聚类方法: {method}")
    
    logger.info(f"选择了{len(global_indices)}个代表性分子")
    return global_indices


def validate_subset(full_results, subset_indices, config):
    """
    验证子集的代表性
    
    参数:
        full_results: 全库处理结果
        subset_indices: 子集分子索引
        config: 配置字典
        
    返回:
        验证结果字典
    """
    validation_config = config.get('validation', {})
    
    logger.info("验证子集代表性")
    
    # 准备子集数据
    subset_mols = [full_results['mols'][i] for i in subset_indices]
    subset_fps = [full_results['fps'][i] for i in subset_indices]
    subset_basic_desc = [full_results['basic_desc'][i] for i in subset_indices]
    
    # 准备有效分子索引（过滤空值）
    valid_full_idx = [i for i, m in enumerate(full_results['mols']) if m is not None]
    valid_full_fps = [full_results['fps'][i] for i in valid_full_idx]
    
    validation_results = {}
    
    # 1. 计算覆盖度指标
    metrics = calculate_coverage_metrics(valid_full_fps, subset_fps)
    validation_results['coverage_metrics'] = metrics
    
    logger.info(f"覆盖度指标: 覆盖率={metrics['coverage_ratio']:.2%}, 平均距离={metrics['mean_distance']:.3f}")
    
    # 2. 如果配置了属性分布比较
    if validation_config.get('property_coverage', True):
        # 选择要比较的属性
        prop_names = ['mw', 'logp', 'tpsa', 'hba', 'hbd', 'rotatable_bonds']
        
        # 过滤有效属性
        valid_full_props = [full_results['basic_desc'][i] for i in valid_full_idx]
        
        # 绘制属性分布对比图
        prop_fig = plot_property_distributions(valid_full_props, subset_basic_desc, prop_names)
        validation_results['property_distribution_fig'] = prop_fig
    
    # 3. 如果配置了最近邻分析
    if validation_config.get('nearest_neighbor_analysis', True):
        # 计算全库到子集的最近邻距离
        nn_distances, mean_dist, max_dist, median_dist = calculate_nearest_neighbor_distance(
            valid_full_fps, subset_fps
        )
        
        # 绘制最近邻距离直方图
        nn_fig = plot_nearest_neighbor_histogram(nn_distances)
        validation_results['nn_distribution_fig'] = nn_fig
        validation_results['nn_distances'] = nn_distances
    
    return validation_results


def save_results(df, subset_indices, processed_results, validation_results, output_dir, config):
    """
    保存处理结果
    
    参数:
        df: 原始数据DataFrame
        subset_indices: 子集分子索引
        processed_results: 处理结果字典
        validation_results: 验证结果字典
        output_dir: 输出目录
        config: 配置字典
    """
    output_config = config.get('output', {})
    formats = output_config.get('formats', ['csv', 'sdf'])
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 提取子集数据
    subset_df = df.iloc[subset_indices].copy().reset_index(drop=True)
    subset_mols = [processed_results['mols'][i] for i in subset_indices]
    
    # 导出子集
    if 'csv' in formats:
        csv_path = os.path.join(output_dir, f"representative_subset_{timestamp}.csv")
        subset_df.to_csv(csv_path, index=False)
        logger.info(f"已保存CSV至 {csv_path}")
    
    if 'sdf' in formats:
        sdf_path = os.path.join(output_dir, f"representative_subset_{timestamp}.sdf")
        with Chem.SDWriter(sdf_path) as w:
            for mol in subset_mols:
                if mol is not None:
                    w.write(mol)
        logger.info(f"已保存SDF至 {sdf_path}")
    
    # 保存验证结果图表
    if validation_results:
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        if 'property_distribution_fig' in validation_results:
            prop_fig_path = os.path.join(plots_dir, f"property_distribution_{timestamp}.png")
            validation_results['property_distribution_fig'].savefig(prop_fig_path, dpi=300)
            logger.info(f"已保存属性分布图至 {prop_fig_path}")
        
        if 'nn_distribution_fig' in validation_results:
            nn_fig_path = os.path.join(plots_dir, f"nearest_neighbor_distribution_{timestamp}.png")
            validation_results['nn_distribution_fig'].savefig(nn_fig_path, dpi=300)
            logger.info(f"已保存最近邻距离图至 {nn_fig_path}")
    
    # 保存指标结果
    if 'coverage_metrics' in validation_results:
        metrics_path = os.path.join(output_dir, f"coverage_metrics_{timestamp}.txt")
        with open(metrics_path, 'w') as f:
            for k, v in validation_results['coverage_metrics'].items():
                f.write(f"{k}: {v}\n")
        logger.info(f"已保存覆盖度指标至 {metrics_path}")
    
    # 如果配置保存中间结果
    if output_config.get('save_intermediates', False):
        # 保存特征和处理结果
        pickle_path = os.path.join(output_dir, f"processed_results_{timestamp}.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'processed_results': processed_results,
                'subset_indices': subset_indices,
                'validation_results': validation_results
            }, f)
        logger.info(f"已保存中间结果至 {pickle_path}")
    
    logger.info("所有结果保存完成")


def main():
    parser = argparse.ArgumentParser(description='分子库代表性子集选择工具')
    parser.add_argument('--input', required=True, help='输入文件路径(.csv或.sdf)')
    parser.add_argument('--output', default='output', help='输出目录')
    parser.add_argument('--config', default='configs/default_config.yml', help='配置文件路径')
    parser.add_argument('--smiles_col', default='SMILES', help='SMILES列名')
    parser.add_argument('--use_gpu', action='store_true', help='启用GPU加速')
    parser.add_argument('--gpu_id', type=int, default=0, help='使用的GPU设备ID')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)
    
    # 加载配置
    config = load_config(args.config)
    
    # 如果命令行指定了GPU选项，覆盖配置文件
    if args.use_gpu:
        if 'gpu' not in config:
            config['gpu'] = {}
        config['gpu']['enabled'] = True
        config['gpu']['device_id'] = args.gpu_id
        
        # 设置CUDA设备
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        logger.info(f"命令行启用GPU，设备ID: {args.gpu_id}")
    
    # 检查GPU可用性
    if GPU_TOOLS_AVAILABLE and config.get('gpu', {}).get('enabled', False):
        gpu_status = check_gpu_availability()
        if gpu_status['any_gpu']:
            logger.info(f"检测到可用的GPU加速: {gpu_status}")
        else:
            logger.warning("未检测到可用的GPU，将使用CPU计算")
    
    # 加载数据
    df = load_data(args.input)
    
    # 批量处理分子
    processed_results = batch_process_molecules(df, config, smiles_col=args.smiles_col)
    
    # 特征组合和降维
    _, reduced_features = combine_features_and_reduce(processed_results, config)
    
    # 选择代表性子集
    subset_indices = select_representative_subset(reduced_features, processed_results['mols'], config)
    
    # 验证子集代表性
    validation_results = validate_subset(processed_results, subset_indices, config)
    
    # 保存结果
    save_results(df, subset_indices, processed_results, validation_results, args.output, config)
    
    logger.info("处理完成!")


if __name__ == '__main__':
    main() 