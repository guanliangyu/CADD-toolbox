"""
分子库代表性子集选择系统 - Streamlit交互式应用
"""
import os
import sys
import yaml
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io
from pathlib import Path
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, PandasTools
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目根目录到路径，确保能导入utils模块
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from utils.molecular_utils import MoleculeProcessor
from utils.clustering_utils import (
    butina_clustering, kmeans_clustering, maxmin_selection, 
    select_cluster_representatives, select_representatives_from_kmeans
)
from utils.feature_utils import DimensionalityReducer, FeatureCombiner
from utils.validation_utils import (
    plot_property_distributions, plot_nearest_neighbor_histogram,
    plot_pca_visualization, calculate_coverage_metrics
)

# 导入GPU工具
try:
    from utils.gpu_utils import check_gpu_availability
    GPU_TOOLS_AVAILABLE = True
except ImportError:
    GPU_TOOLS_AVAILABLE = False

# 设置页面标题和配置
st.set_page_config(
    page_title="分子库代表性子集选择系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Streamlit应用标题
st.title("分子库代表性子集选择系统")

st.markdown("""
本应用实现了从大型分子库中提取代表性子集的流程，主要功能包括：
- 分子过滤和标准化
- 2D指纹和理化性质计算
- 3D构象生成和形状特征计算
- 多种聚类算法支持
- 子集代表性验证与评价
""")

# 侧边栏配置区域
st.sidebar.header("配置参数")

# 文件上传区域
uploaded_file = st.sidebar.file_uploader("上传SMILES文件 (CSV)", type=["csv"])

# 参数设置
st.sidebar.subheader("基本参数")
smiles_col = st.sidebar.text_input("SMILES列名", "SMILES")
subset_ratio = st.sidebar.slider("子集比例 (%)", 0.1, 10.0, 1.0, 0.1)

# 聚类方法选择
st.sidebar.subheader("聚类方法")
clustering_method = st.sidebar.selectbox(
    "选择聚类/选择算法",
    ["butina", "kmeans", "maxmin"]
)

# 根据聚类方法显示不同参数
if clustering_method == "butina":
    cutoff = st.sidebar.slider("相似度阈值", 0.0, 1.0, 0.6, 0.05)
elif clustering_method == "kmeans":
    use_fixed_clusters = st.sidebar.checkbox("使用固定簇数量", value=False)
    if use_fixed_clusters:
        n_clusters = st.sidebar.number_input("簇数量", 10, 10000, 100)
    else:
        st.sidebar.info(f"将使用子集比例 ({subset_ratio}%) 计算簇数量")
    kmeans_iterations = st.sidebar.slider("最大迭代次数", 10, 1000, 100)
elif clustering_method == "maxmin":
    init_method = st.sidebar.selectbox("初始点选择", ["random", "first"])

# 特征计算选项
st.sidebar.subheader("特征计算")
fps_type = st.sidebar.selectbox("指纹类型", ["morgan", "rdkit", "maccs"])
if fps_type == "morgan":
    morgan_radius = st.sidebar.slider("Morgan半径", 1, 4, 2)
    morgan_bits = st.sidebar.selectbox("Morgan位数", [512, 1024, 2048])

include_3d = st.sidebar.checkbox("生成3D构象", value=False)
if include_3d:
    include_charges = st.sidebar.checkbox("计算Gasteiger电荷", value=False)

# 并行处理选项
st.sidebar.subheader("计算参数")
batch_size = st.sidebar.slider("批处理大小", 100, 5000, 1000, 100)
n_jobs = st.sidebar.slider("并行作业数", 1, mp.cpu_count(), min(4, mp.cpu_count()))

# GPU加速设置
if GPU_TOOLS_AVAILABLE:
    st.sidebar.subheader("GPU加速")
    use_gpu = st.sidebar.checkbox("启用GPU加速", value=True)
    
    # 检查可用GPU
    if use_gpu:
        gpu_status = check_gpu_availability()
        if gpu_status['any_gpu']:
            st.sidebar.success("检测到可用GPU")
            
            # 显示可用的GPU库
            gpu_libs = [k for k, v in gpu_status.items() if v and k != 'any_gpu']
            if gpu_libs:
                st.sidebar.info(f"可用GPU库: {', '.join(gpu_libs)}")
            
            # GPU设置选项
            gpu_id = st.sidebar.number_input("GPU设备ID", 0, 7, 0, 1)
            
            # 启用的功能
            st.sidebar.subheader("GPU加速功能")
            gpu_kmeans = st.sidebar.checkbox("K-means聚类", value=True)
            gpu_distances = st.sidebar.checkbox("距离计算", value=True)
            gpu_pca = st.sidebar.checkbox("PCA降维", value=True)
        else:
            st.sidebar.warning("未检测到可用GPU，将使用CPU计算")
            use_gpu = False
else:
    use_gpu = False

# 主内容区域
tab1, tab2, tab3 = st.tabs(["数据处理", "聚类与子集选择", "验证与下载"])

def process_batch(batch_data, processor, config, include_3d=True, include_charges=True):
    """处理一批分子数据
    
    参数:
        batch_data: (smiles_batch, indices) 元组
        processor: MoleculeProcessor实例
        config: 配置字典
        include_3d: 是否包含3D特征
        include_charges: 是否包含电荷
    """
    smiles_batch, indices = batch_data
    batch_results = {
        'indices': indices,
        'mols': [],
        'fps': [],
        'fps_binary': [],
        'basic_desc': [],
        'shape_desc': [],
        'charges': []
    }
    
    fp_config = config.get('features', {}).get('fingerprints', {})
    fp_type = fp_config.get('types', ['morgan'])[0]
    radius = fp_config.get('morgan_radius', 2)
    n_bits = fp_config.get('morgan_bits', 1024)
    
    for smiles in smiles_batch:
        # 准备分子
        mol = processor.prepare_molecule(smiles)
        
        if mol is None:
            batch_results['mols'].append(None)
            batch_results['fps'].append(None)
            batch_results['fps_binary'].append(None)
            batch_results['basic_desc'].append({})
            batch_results['shape_desc'].append({})
            batch_results['charges'].append(None)
            continue
        
        # 计算指纹
        fp = processor.compute_fingerprint(mol, radius=radius, nBits=n_bits, fp_type=fp_type)
        fp_array = processor.fp_to_numpy(fp)
        
        # 计算基本描述符
        basic_desc = processor.compute_basic_descriptors(mol)
        
        # 可选：3D构象和形状特征
        shape_desc = {}
        charges = None
        
        if include_3d:
            mol_3d = processor.generate_3d_conformer(mol)
            
            if mol_3d:
                shape_desc = processor.compute_shape_descriptors(mol_3d)
                
                if include_charges:
                    charges = processor.compute_gasteiger_charges(mol_3d)
        
        # 添加结果
        batch_results['mols'].append(mol)
        batch_results['fps'].append(fp_array)
        batch_results['fps_binary'].append(fp)  # 保存原始指纹对象
        batch_results['basic_desc'].append(basic_desc)
        batch_results['shape_desc'].append(shape_desc)
        batch_results['charges'].append(charges)
    
    return batch_results

def process_molecules(df, config):
    """处理分子并计算指纹和特征"""
    processor = MoleculeProcessor(config)
    
    # 创建结果容器
    results = {
        'mols': [],
        'fps': [],
        'fps_binary': [],  # 用于传递给Butina聚类
        'basic_desc': [],
        'shape_desc': [],
        'charges': []
    }
    
    # 进度条
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(df)
    batch_size = config.get('data', {}).get('batching', {}).get('batch_size', 1000)
    n_jobs = config.get('data', {}).get('batching', {}).get('n_jobs', -1)
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    include_3d = config.get('data', {}).get('conformers', {}).get('enabled', True)
    include_charges = config.get('data', {}).get('charges', {}).get('enabled', True)
    
    # 准备批次
    smiles_list = df[smiles_col].tolist()
    batches = []
    
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        batches.append((smiles_list[i:end], list(range(i, end))))
    
    # 初始化结果列表
    results['mols'] = [None] * total
    results['fps'] = [None] * total
    results['fps_binary'] = [None] * total
    results['basic_desc'] = [{}] * total
    results['shape_desc'] = [{}] * total
    results['charges'] = [None] * total
    
    # 处理所有批次
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [
            executor.submit(process_batch, batch, processor, config, include_3d, include_charges)
            for batch in batches
        ]
        
        for i, future in enumerate(as_completed(futures)):
            try:
                batch_result = future.result()
                
                # 更新进度
                progress = (i + 1) / len(batches)
                progress_bar.progress(progress)
                
                # 更新状态文本
                elapsed = time.time() - start_time
                estimated_total = elapsed / progress if progress > 0 else 0
                remaining = estimated_total - elapsed
                status_text.text(f"处理中... {i+1}/{len(batches)} 批次 | 已用时间: {elapsed:.1f}秒 | 剩余时间: {remaining:.1f}秒")
                
                # 合并结果
                indices = batch_result['indices']
                for j, idx in enumerate(indices):
                    results['mols'][idx] = batch_result['mols'][j]
                    results['fps'][idx] = batch_result['fps'][j]
                    results['fps_binary'][idx] = batch_result['fps_binary'][j]
                    results['basic_desc'][idx] = batch_result['basic_desc'][j]
                    results['shape_desc'][idx] = batch_result['shape_desc'][j]
                    results['charges'][idx] = batch_result['charges'][j]
            except Exception as e:
                logger.error(f"处理批次时出错: {str(e)}")
                continue
    
    duration = time.time() - start_time
    
    # 清理进度条和状态文本
    progress_bar.empty()
    
    # 计算有效分子数量
    valid_count = sum(1 for m in results['mols'] if m is not None)
    status_text.text(f"处理完成: {valid_count}/{total} 个有效分子, 耗时 {duration:.1f} 秒")
    
    return results


def cluster_and_select(processed_results, config):
    """根据所选算法聚类并选择代表分子"""
    # 创建进度条
    cluster_progress = st.progress(0)
    cluster_status = st.empty()
    
    cluster_status.text("准备聚类数据...")
    
    # 过滤有效分子
    valid_idx = [i for i, m in enumerate(processed_results['mols']) if m is not None]
    
    if not valid_idx:
        st.error("没有有效分子可供聚类")
        return []
    
    # 聚类方法
    method = config.get('clustering', {}).get('method', 'butina')
    subset_ratio = config.get('subset_ratio', 1.0)
    
    # 计算目标选择数量
    total_valid = len(valid_idx)
    target_count = max(1, int(total_valid * subset_ratio / 100.0))
    
    cluster_status.text(f"使用 {method} 方法聚类中...")
    start_time = time.time()
    
    # 根据方法执行聚类
    if method == 'butina':
        cluster_progress.progress(0.1)
        
        # Butina聚类用原始指纹
        valid_fps = [processed_results['fps_binary'][i] for i in valid_idx]
        cutoff = config.get('clustering', {}).get('butina', {}).get('cutoff', 0.4)
        
        # 计算距离矩阵
        cluster_status.text("计算距离矩阵...")
        n = len(valid_fps)
        dists = []
        
        # 分批计算距离，以便更新进度
        batch_size = 1000
        total_pairs = n * (n - 1) // 2
        pairs_processed = 0
        
        for i in range(n):
            for j in range(i+1, n):
                if valid_fps[i] is not None and valid_fps[j] is not None:
                    try:
                        sim = DataStructs.TanimotoSimilarity(valid_fps[i], valid_fps[j])
                        d = 1.0 - sim
                    except:
                        d = 1.0  # 如果计算失败，假设最大距离
                else:
                    d = 1.0  # 无效指纹
                
                dists.append(d)
                
                # 更新进度
                pairs_processed += 1
                if pairs_processed % batch_size == 0:
                    progress = 0.1 + 0.5 * (pairs_processed / total_pairs)
                    cluster_progress.progress(progress)
                    cluster_status.text(f"计算距离矩阵... {pairs_processed}/{total_pairs} 对")
        
        # 执行聚类
        cluster_progress.progress(0.7)
        cluster_status.text("执行Butina聚类...")
        clusters = Butina.ClusterData(dists, n, cutoff, isDistData=True)
        
        cluster_progress.progress(0.9)
        cluster_status.text(f"选择簇代表... 共 {len(clusters)} 个簇")
        
        # 选择代表分子
        local_indices = []
        for cluster in clusters:
            # 从每个簇选择一个代表（这里简单地选第一个）
            if cluster:  # 确保簇非空
                local_indices.append(cluster[0])
        
        # 映射回全局索引
        representatives = [valid_idx[i] for i in local_indices]
        
    elif method == 'kmeans':
        cluster_progress.progress(0.1)
        
        # 提取特征
        valid_features = [processed_results['fps'][i] for i in valid_idx]
        valid_features = np.array([f for f in valid_features if f is not None])
        
        if len(valid_features) == 0:
            st.error("没有有效特征可供K-means聚类")
            return []
        
        # K-means参数
        if config.get('clustering', {}).get('kmeans', {}).get('use_ratio', True):
            # 根据子集比例计算簇数量
            n_clusters = target_count
        else:
            # 使用固定的簇数量
            n_clusters = config.get('clustering', {}).get('kmeans', {}).get('n_clusters', target_count)
        
        max_iter = config.get('clustering', {}).get('kmeans', {}).get('max_iter', 100)
        
        # 执行K-means聚类
        cluster_status.text(f"执行K-means聚类, k={n_clusters}...")
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=max_iter)
        cluster_progress.progress(0.3)
        
        labels = kmeans.fit_predict(valid_features)
        centers = kmeans.cluster_centers_
        
        cluster_progress.progress(0.7)
        cluster_status.text("选择簇代表...")
        
        # 选择最接近簇中心的点
        representatives = []
        for i in range(n_clusters):
            # 找出属于当前簇的所有点
            cluster_indices = np.where(labels == i)[0]
            
            if len(cluster_indices) == 0:
                continue  # 跳过空簇
            
            # 计算到中心的距离
            cluster_points = valid_features[cluster_indices]
            distances = np.linalg.norm(cluster_points - centers[i], axis=1)
            
            # 选择最近的点
            closest_point_idx = cluster_indices[np.argmin(distances)]
            
            # 映射回原始索引
            representatives.append(valid_idx[closest_point_idx])
        
    elif method == 'maxmin':
        cluster_progress.progress(0.1)
        
        # 提取特征
        valid_features = [processed_results['fps'][i] for i in valid_idx]
        valid_features = np.array([f for f in valid_features if f is not None])
        
        if len(valid_features) == 0:
            st.error("没有有效特征可供MaxMin选择")
            return []
        
        # MaxMin参数
        num_to_select = min(target_count, len(valid_features))
        
        # 执行MaxMin选择
        cluster_status.text(f"执行MaxMin选择, 目标数量={num_to_select}...")
        
        # 定义距离函数
        def distance_fn(i, j):
            return np.linalg.norm(valid_features[i] - valid_features[j])
        
        from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
        
        # 初始种子点
        init_method = config.get('clustering', {}).get('maxmin', {}).get('init_method', 'random')
        if init_method == 'random':
            seed_idx = np.random.randint(0, len(valid_features))
        else:
            seed_idx = 0
        
        # MaxMin选择
        picker = MaxMinPicker()
        
        # 分批更新进度
        n_steps = 20
        step_size = num_to_select // n_steps
        
        # 使用小函数逐步执行MaxMin，以便更新进度
        selected = []
        current_selection = [seed_idx]
        remaining_to_select = num_to_select - 1
        
        for step in range(n_steps):
            n_to_pick = min(step_size, remaining_to_select)
            if n_to_pick <= 0:
                break
            
            # 选择下一批点
            new_selections = list(picker.LazyBitVectorPick(
                distance_fn, len(valid_features), n_to_pick, current_selection[-1],
                firstPicks=current_selection
            ))
            
            current_selection.extend(new_selections)
            remaining_to_select -= len(new_selections)
            
            # 更新进度
            progress = 0.1 + 0.8 * (len(current_selection) / num_to_select)
            cluster_progress.progress(progress)
            cluster_status.text(f"MaxMin选择中... {len(current_selection)}/{num_to_select}")
        
        local_indices = current_selection
        
        # 映射回全局索引
        representatives = [valid_idx[i] for i in local_indices]
    
    else:
        st.error(f"不支持的聚类方法: {method}")
        return []
    
    duration = time.time() - start_time
    cluster_progress.progress(1.0)
    cluster_status.text(f"选择完成: 共选出 {len(representatives)} 个代表分子, 耗时 {duration:.1f} 秒")
    
    return representatives


def validate_selection(processed_results, subset_indices):
    """验证子集选择的质量"""
    if not subset_indices:
        st.error("没有选出代表性分子，无法验证")
        return None
    
    # 创建进度条
    validate_progress = st.progress(0)
    validate_status = st.empty()
    validate_status.text("开始验证...")
    
    # 提取子集数据
    subset_fps = [processed_results['fps_binary'][i] for i in subset_indices]
    subset_basic_desc = [processed_results['basic_desc'][i] for i in subset_indices]
    
    # 过滤有效分子
    valid_idx = [i for i, m in enumerate(processed_results['mols']) if m is not None]
    valid_fps = [processed_results['fps_binary'][i] for i in valid_idx]
    valid_basic_desc = [processed_results['basic_desc'][i] for i in valid_idx]
    
    validate_progress.progress(0.1)
    
    # 1. 计算覆盖度指标
    validate_status.text("计算覆盖度指标...")
    metrics = calculate_coverage_metrics(valid_fps, subset_fps)
    
    validate_progress.progress(0.4)
    
    # 2. 比较属性分布
    validate_status.text("比较属性分布...")
    prop_names = ['mw', 'logp', 'tpsa', 'hba', 'hbd', 'rotatable_bonds']
    prop_fig = plot_property_distributions(valid_basic_desc, subset_basic_desc, prop_names)
    
    validate_progress.progress(0.7)
    
    # 3. 最近邻分析
    validate_status.text("执行最近邻分析...")
    from utils.validation_utils import calculate_nearest_neighbor_distance, plot_nearest_neighbor_histogram
    
    nn_distances, mean_dist, max_dist, median_dist = calculate_nearest_neighbor_distance(
        valid_fps, subset_fps
    )
    nn_fig = plot_nearest_neighbor_histogram(nn_distances)
    
    validate_progress.progress(1.0)
    validate_status.text("验证完成!")
    
    # 返回验证结果
    return {
        'coverage_metrics': metrics,
        'property_fig': prop_fig,
        'nn_fig': nn_fig,
        'nn_distances': nn_distances
    }


# 开始处理逻辑
if uploaded_file is not None:
    # 内存中读取CSV
    df = pd.read_csv(uploaded_file)
    
    # 检查SMILES列
    if smiles_col not in df.columns:
        st.error(f"找不到SMILES列: {smiles_col}")
    else:
        with tab1:
            st.subheader("数据概览")
            st.write(f"共加载 {len(df)} 条记录")
            st.dataframe(df.head())
            
            # 创建配置字典
            config = {
                'data': {
                    'filtering': {
                        'enabled': True,
                        'max_mw': 1000,
                        'min_mw': 100
                    },
                    'conformers': {
                        'enabled': include_3d
                    },
                    'charges': {
                        'enabled': include_charges if include_3d else False
                    },
                    'batching': {
                        'enabled': True,
                        'batch_size': batch_size,
                        'n_jobs': n_jobs
                    }
                },
                'features': {
                    'fingerprints': {
                        'types': [fps_type],
                        'morgan_radius': morgan_radius if fps_type == 'morgan' else 2,
                        'morgan_bits': morgan_bits if fps_type == 'morgan' else 1024
                    }
                },
                'clustering': {
                    'method': clustering_method,
                    'butina': {
                        'cutoff': cutoff if clustering_method == 'butina' else 0.4
                    },
                    'kmeans': {
                        'n_clusters': n_clusters if clustering_method == 'kmeans' and 'use_fixed_clusters' in locals() and use_fixed_clusters else 100,
                        'max_iter': kmeans_iterations if clustering_method == 'kmeans' else 100,
                        'use_ratio': not (clustering_method == 'kmeans' and 'use_fixed_clusters' in locals() and use_fixed_clusters)
                    },
                    'maxmin': {
                        'init_method': init_method if clustering_method == 'maxmin' else 'random'
                    }
                },
                'subset_ratio': subset_ratio
            }
            
            # 添加GPU配置
            if 'use_gpu' in locals() and use_gpu:
                config['gpu'] = {
                    'enabled': True,
                    'device_id': gpu_id if 'gpu_id' in locals() else 0,
                    'auto_detect': True,
                    'features': {
                        'kmeans': gpu_kmeans if 'gpu_kmeans' in locals() else True,
                        'distances': gpu_distances if 'gpu_distances' in locals() else True,
                        'pca': gpu_pca if 'gpu_pca' in locals() else True
                    }
                }
                # 如果选择了GPU，设置CUDA环境变量
                if 'gpu_id' in locals():
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            # 处理按钮
            if st.button("开始处理分子"):
                with st.spinner("处理分子中..."):
                    # 执行分子处理
                    processed_results = process_molecules(df, config)
                    
                    # 保存结果到Session State
                    st.session_state['processed_results'] = processed_results
                    st.session_state['df'] = df
                    st.session_state['config'] = config
                    
                    # 显示有效分子数量
                    valid_count = sum(1 for m in processed_results['mols'] if m is not None)
                    st.success(f"处理完成: {valid_count}/{len(df)} 个有效分子")
                    
                    # 显示一些分子结构
                    if valid_count > 0:
                        st.subheader("部分分子结构预览")
                        valid_mols = [m for m in processed_results['mols'] if m is not None]
                        sample_size = min(10, len(valid_mols))
                        sample_mols = np.random.choice(valid_mols, sample_size, replace=False).tolist()
                        
                        # 绘制分子图像
                        img = Draw.MolsToGridImage(sample_mols, molsPerRow=5, subImgSize=(200, 200))
                        st.image(img)
        
        # 聚类和选择标签页
        with tab2:
            st.subheader("聚类与子集选择")
            
            # 检查是否已处理分子
            if 'processed_results' not in st.session_state:
                st.info("请先在数据处理标签页中处理分子")
            else:
                st.write(f"聚类方法: **{clustering_method}**")
                
                # 显示聚类特定参数
                if clustering_method == "butina":
                    st.write(f"相似度阈值: **{cutoff}**")
                elif clustering_method == "kmeans":
                    if 'use_fixed_clusters' in locals() and use_fixed_clusters:
                        st.write(f"簇数量: **{n_clusters}**")
                    else:
                        valid_count = sum(1 for m in st.session_state['processed_results']['mols'] if m is not None)
                        estimated_clusters = max(1, int(valid_count * subset_ratio / 100.0))
                        st.write(f"预计簇数量: **{estimated_clusters}** (基于{subset_ratio}%的子集比例)")
                elif clustering_method == "maxmin":
                    st.write(f"初始化方法: **{init_method}**")
                
                st.write(f"目标子集比例: **{subset_ratio}%**")
                
                # 聚类按钮
                if st.button("开始聚类和选择"):
                    with st.spinner("聚类与选择中..."):
                        # 执行聚类和选择
                        subset_indices = cluster_and_select(
                            st.session_state['processed_results'],
                            st.session_state['config']
                        )
                        
                        # 保存结果到Session State
                        st.session_state['subset_indices'] = subset_indices
                        
                        # 显示子集大小
                        st.success(f"子集选择完成: 选出 {len(subset_indices)} 个代表性分子")
                        
                        # 显示代表分子预览
                        if subset_indices:
                            st.subheader("代表性分子预览")
                            subset_mols = [st.session_state['processed_results']['mols'][i] 
                                          for i in subset_indices if st.session_state['processed_results']['mols'][i] is not None]
                            
                            # 最多显示15个分子
                            preview_size = min(15, len(subset_mols))
                            preview_mols = subset_mols[:preview_size]
                            
                            # 绘制分子图像
                            img = Draw.MolsToGridImage(preview_mols, molsPerRow=5, subImgSize=(200, 200))
                            st.image(img)
        
        # 验证和下载标签页
        with tab3:
            st.subheader("子集验证与下载")
            
            # 检查是否已选择子集
            if 'subset_indices' not in st.session_state:
                st.info("请先在'聚类与子集选择'标签页中选择子集")
            else:
                subset_indices = st.session_state['subset_indices']
                
                # 显示子集大小和比例
                total_mols = len(st.session_state['df'])
                valid_mols = sum(1 for m in st.session_state['processed_results']['mols'] if m is not None)
                subset_size = len(subset_indices)
                
                st.write(f"子集大小: **{subset_size}** 分子")
                st.write(f"占全部分子比例: **{subset_size/total_mols:.2%}**")
                st.write(f"占有效分子比例: **{subset_size/valid_mols:.2%}**")
                
                # 验证按钮
                if st.button("验证子集质量"):
                    with st.spinner("验证中..."):
                        # 执行验证
                        validation_results = validate_selection(
                            st.session_state['processed_results'],
                            subset_indices
                        )
                        
                        # 保存结果到Session State
                        st.session_state['validation_results'] = validation_results
                        
                        if validation_results:
                            # 显示覆盖度指标
                            st.subheader("覆盖度指标")
                            metrics = validation_results['coverage_metrics']
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("覆盖率", f"{metrics['coverage_ratio']:.2%}")
                            with col2:
                                st.metric("平均距离", f"{metrics['mean_distance']:.3f}")
                            with col3:
                                st.metric("中位数距离", f"{metrics['median_distance']:.3f}")
                            
                            # 显示属性分布图
                            st.subheader("属性分布比较")
                            st.pyplot(validation_results['property_fig'])
                            
                            # 显示最近邻分析图
                            st.subheader("最近邻距离分析")
                            st.pyplot(validation_results['nn_fig'])
                
                # 下载子集按钮
                st.subheader("下载子集")
                
                if subset_indices:
                    # 提取子集数据
                    subset_df = st.session_state['df'].iloc[subset_indices].copy().reset_index(drop=True)
                    
                    # CSV下载
                    csv_buffer = io.StringIO()
                    subset_df.to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label="下载CSV格式子集",
                        data=csv_buffer.getvalue(),
                        file_name="representative_subset.csv",
                        mime="text/csv"
                    )
                    
                    # SDF下载
                    try:
                        subset_mols = [st.session_state['processed_results']['mols'][i] 
                                      for i in subset_indices if st.session_state['processed_results']['mols'][i] is not None]
                        
                        if subset_mols:
                            # 创建临时文件
                            with tempfile.NamedTemporaryFile(suffix='.sdf', delete=False) as tmp:
                                # 写入SDF
                                with Chem.SDWriter(tmp.name) as w:
                                    for mol in subset_mols:
                                        w.write(mol)
                                
                                # 读取文件内容
                                with open(tmp.name, 'rb') as f:
                                    sdf_data = f.read()
                                
                                # 提供下载
                                st.download_button(
                                    label="下载SDF格式子集",
                                    data=sdf_data,
                                    file_name="representative_subset.sdf",
                                    mime="chemical/x-mdl-sdfile"
                                )
                                
                                # 删除临时文件
                                os.unlink(tmp.name)
                    except Exception as e:
                        st.error(f"创建SDF文件时出错: {e}")
else:
    st.info("请上传含有SMILES数据的CSV文件开始分析。") 