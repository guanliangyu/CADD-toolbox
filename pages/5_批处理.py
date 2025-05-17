"""
分子库代表性子集选择系统 - 批处理页面
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
import concurrent.futures
import io
import json
import tempfile

# 添加项目根目录到路径，确保能导入utils模块
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

# 导入工具模块
from utils.molecular_utils import MoleculeProcessor
from utils.state_utils import initialize_session_state, display_state_sidebar
from utils.config_utils import create_config_from_parameters
from utils.clustering_utils import (
    butina_clustering, kmeans_clustering, maxmin_selection, 
    select_cluster_representatives, select_representatives_from_kmeans
)

# 设置页面标题和配置
st.set_page_config(
    page_title="分子库代表性子集选择系统 - 批处理",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
initialize_session_state()

# 在侧边栏显示当前状态
display_state_sidebar()

# 页面标题
st.title("批处理")
st.markdown("本页面提供批量处理多个数据集的功能，可以一次性处理多个文件并导出结果。")

# 配置区域
st.subheader("批处理配置")

col1, col2 = st.columns(2)
with col1:
    # 上传多个文件
    uploaded_files = st.file_uploader("上传多个SMILES文件", type=["csv"], accept_multiple_files=True)
    
    # SMILES列名
    smiles_col = st.text_input("SMILES列名", "SMILES")
    
    # 子集比例
    subset_ratio = st.slider("子集比例 (%)", 0.1, 10.0, 1.0, 0.1)

with col2:
    # 聚类方法
    clustering_method = st.selectbox(
        "聚类/选择算法",
        ["butina", "kmeans", "maxmin"]
    )
    
    # 根据聚类方法显示不同参数
    if clustering_method == "butina":
        cutoff = st.slider("相似度阈值", 0.0, 1.0, 0.6, 0.05)
        clustering_params = {"cutoff": cutoff}
    elif clustering_method == "kmeans":
        use_fixed_clusters = st.checkbox("使用固定簇数量", value=False)
        if use_fixed_clusters:
            n_clusters = st.number_input("簇数量", 10, 10000, 100)
            clustering_params = {"n_clusters": n_clusters, "use_fixed_clusters": True}
        else:
            st.info(f"将使用子集比例 ({subset_ratio}%) 计算簇数量")
            clustering_params = {"use_fixed_clusters": False}
        
        kmeans_iterations = st.slider("最大迭代次数", 10, 1000, 100)
        clustering_params["kmeans_iterations"] = kmeans_iterations
    elif clustering_method == "maxmin":
        init_method = st.selectbox("初始点选择", ["random", "first"])
        clustering_params = {"init_method": init_method}

# 指纹配置
st.subheader("指纹与特征配置")
fps_type = st.selectbox("指纹类型", ["morgan", "rdkit", "maccs"])
if fps_type == "morgan":
    morgan_radius = st.slider("Morgan半径", 1, 4, 2)
    morgan_bits = st.selectbox("Morgan位数", [512, 1024, 2048])
    fps_params = {"morgan_radius": morgan_radius, "morgan_bits": morgan_bits}
else:
    fps_params = {}

include_properties = st.checkbox("包含理化性质", value=True)
validation_stats = st.checkbox("计算验证统计", value=True)

# 并行配置
n_jobs = st.slider("并行处理任务数", 1, 8, 4)

# 生成配置
def create_batch_config():
    config = {
        'smiles_col': smiles_col,
        'subset_ratio': subset_ratio,
        'clustering_method': clustering_method,
        'fps_type': fps_type,
        'include_3d': False,
        'include_charges': False,
        'batch_size': 1000,
        'n_jobs': n_jobs,
        **clustering_params,
        **fps_params
    }
    return config

# 批处理函数
def process_file(file, config):
    """处理单个文件
    
    参数:
        file: 上传的文件对象
        config: 配置字典
        
    返回:
        处理结果字典
    """
    start_time = time.time()
    
    # 读取文件
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return {
            'file_name': file.name,
            'status': 'error',
            'message': f"读取文件失败: {str(e)}",
            'duration': 0,
            'subset_size': 0,
            'total_size': 0,
            'subset_data': None
        }
    
    # 检查SMILES列
    smiles_col = config.get('smiles_col', 'SMILES')
    if smiles_col not in df.columns:
        return {
            'file_name': file.name,
            'status': 'error',
            'message': f"找不到SMILES列: {smiles_col}",
            'duration': 0,
            'subset_size': 0,
            'total_size': 0,
            'subset_data': None
        }
    
    # 创建完整配置
    full_config = create_config_from_parameters(**config)
    
    # 初始化分子处理器
    processor = MoleculeProcessor(full_config)
    
    # 处理分子
    results = {
        'mols': [],
        'fps': [],
        'fps_binary': [],
        'basic_desc': [],
        'shape_desc': [],
        'charges': []
    }
    
    # 简化版本的处理逻辑
    smiles_list = df[smiles_col].tolist()
    
    # 简化批处理
    for smiles in smiles_list:
        mol = processor.prepare_molecule(smiles)
        
        if mol is None:
            results['mols'].append(None)
            results['fps'].append(None)
            results['fps_binary'].append(None)
            results['basic_desc'].append({})
            results['shape_desc'].append({})
            results['charges'].append(None)
            continue
        
        # 计算指纹
        fp_config = full_config.get('features', {}).get('fingerprints', {})
        fp_type = fp_config.get('types', ['morgan'])[0]
        radius = fp_config.get('morgan_radius', 2)
        n_bits = fp_config.get('morgan_bits', 1024)
        
        fp = processor.compute_fingerprint(mol, radius=radius, nBits=n_bits, fp_type=fp_type)
        fp_array = processor.fp_to_numpy(fp)
        
        # 计算基本描述符
        basic_desc = processor.compute_basic_descriptors(mol) if include_properties else {}
        
        # 添加结果
        results['mols'].append(mol)
        results['fps'].append(fp_array)
        results['fps_binary'].append(fp)
        results['basic_desc'].append(basic_desc)
        results['shape_desc'].append({})
        results['charges'].append(None)
    
    # 聚类和选择
    method = full_config.get('clustering', {}).get('method', 'butina')
    subset_ratio = full_config.get('subset_ratio', 1.0)
    
    # 过滤有效分子
    valid_idx = [i for i, m in enumerate(results['mols']) if m is not None]
    
    if not valid_idx:
        return {
            'file_name': file.name,
            'status': 'error',
            'message': "没有有效分子可供聚类",
            'duration': 0,
            'subset_size': 0,
            'total_size': len(df),
            'subset_data': None
        }
    
    # 计算目标选择数量
    total_valid = len(valid_idx)
    target_count = max(1, int(total_valid * subset_ratio / 100.0))
    
    # 执行聚类和选择
    representatives = []
    
    try:
        if method == 'butina':
            # Butina聚类
            valid_fps = [results['fps_binary'][i] for i in valid_idx]
            cutoff = full_config.get('clustering', {}).get('butina', {}).get('cutoff', 0.4)
            
            clusters = butina_clustering(valid_fps, cutoff)
            
            # 选择代表分子
            local_indices = []
            for cluster in clusters:
                if cluster:  # 确保簇非空
                    local_indices.append(cluster[0])
                    
            representatives = [valid_idx[i] for i in local_indices]
            
        elif method == 'kmeans':
            # K-means聚类
            valid_features = [results['fps'][i] for i in valid_idx]
            valid_features = np.array([f for f in valid_features if f is not None])
            
            use_ratio = full_config.get('clustering', {}).get('kmeans', {}).get('use_ratio', True)
            if use_ratio:
                n_clusters = target_count
            else:
                n_clusters = full_config.get('clustering', {}).get('kmeans', {}).get('n_clusters', target_count)
                
            max_iter = full_config.get('clustering', {}).get('kmeans', {}).get('max_iter', 100)
            
            # 使用scikit-learn
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=max_iter)
            labels = kmeans.fit_predict(valid_features)
            centers = kmeans.cluster_centers_
            
            # 选择最接近簇中心的点
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
            # MaxMin选择
            valid_features = [results['fps'][i] for i in valid_idx]
            valid_features = np.array([f for f in valid_features if f is not None])
            
            num_to_select = min(target_count, len(valid_features))
            
            # 基本实现
            from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
            
            # 定义距离函数
            def distance_fn(i, j):
                return np.linalg.norm(valid_features[i] - valid_features[j])
            
            # 初始种子点
            init_method = full_config.get('clustering', {}).get('maxmin', {}).get('init_method', 'random')
            if init_method == 'random':
                seed_idx = np.random.randint(0, len(valid_features))
            else:
                seed_idx = 0
                
            # MaxMin选择
            picker = MaxMinPicker()
            indices = list(picker.LazyBitVectorPick(
                distance_fn, len(valid_features), num_to_select, seed_idx
            ))
            
            # 映射回原始索引
            representatives = [valid_idx[i] for i in indices]
            
    except Exception as e:
        return {
            'file_name': file.name,
            'status': 'error',
            'message': f"聚类失败: {str(e)}",
            'duration': time.time() - start_time,
            'subset_size': 0,
            'total_size': len(df),
            'subset_data': None
        }
    
    # 提取子集数据
    subset_df = df.iloc[representatives].copy().reset_index(drop=True)
    
    # 添加理化性质
    if include_properties:
        for prop_name in ['mw', 'logp', 'tpsa', 'hba', 'hbd', 'rotatable_bonds']:
            subset_df[prop_name] = [
                results['basic_desc'][i].get(prop_name, 0) 
                for i in representatives
            ]
    
    # 计算验证统计
    validation_data = None
    if validation_stats:
        # 基本验证统计
        valid_fps = [results['fps_binary'][i] for i in valid_idx]
        subset_fps = [results['fps_binary'][i] for i in representatives]
        
        from utils.validation_utils import calculate_coverage_metrics, calculate_nearest_neighbor_distance
        
        coverage_metrics = calculate_coverage_metrics(valid_fps, subset_fps)
        nn_distances, mean_dist, max_dist, median_dist = calculate_nearest_neighbor_distance(
            valid_fps, subset_fps
        )
        
        validation_data = {
            'coverage_metrics': coverage_metrics,
            'nn_stats': {
                'mean_dist': mean_dist,
                'max_dist': max_dist,
                'median_dist': median_dist
            }
        }
    
    # 准备返回结果
    duration = time.time() - start_time
    return {
        'file_name': file.name,
        'status': 'success',
        'message': "处理成功",
        'duration': duration,
        'subset_size': len(representatives),
        'total_size': len(df),
        'subset_data': subset_df,
        'validation_data': validation_data
    }

# 启动批处理
if uploaded_files:
    if st.button("开始批处理"):
        # 生成配置
        config = create_batch_config()
        
        # 显示批处理进度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # 处理结果容器
        all_results = []
        
        with st.spinner("处理文件中..."):
            total_files = len(uploaded_files)
            
            # 并行处理文件
            if n_jobs > 1 and total_files > 1:
                with concurrent.futures.ProcessPoolExecutor(max_workers=min(n_jobs, total_files)) as executor:
                    futures = [
                        executor.submit(process_file, file, config)
                        for file in uploaded_files
                    ]
                    
                    for i, future in enumerate(concurrent.futures.as_completed(futures)):
                        try:
                            result = future.result()
                            all_results.append(result)
                        except Exception as e:
                            all_results.append({
                                'file_name': uploaded_files[i].name,
                                'status': 'error',
                                'message': f"处理失败: {str(e)}",
                                'duration': 0,
                                'subset_size': 0,
                                'total_size': 0,
                                'subset_data': None
                            })
                        
                        # 更新进度
                        progress = (i + 1) / total_files
                        progress_bar.progress(progress)
                        status_text.text(f"已处理 {i+1}/{total_files} 个文件")
            else:
                # 串行处理
                for i, file in enumerate(uploaded_files):
                    try:
                        result = process_file(file, config)
                        all_results.append(result)
                    except Exception as e:
                        all_results.append({
                            'file_name': file.name,
                            'status': 'error',
                            'message': f"处理失败: {str(e)}",
                            'duration': 0,
                            'subset_size': 0,
                            'total_size': 0,
                            'subset_data': None
                        })
                    
                    # 更新进度
                    progress = (i + 1) / total_files
                    progress_bar.progress(progress)
                    status_text.text(f"已处理 {i+1}/{total_files} 个文件")
        
        # 显示结果
        st.success("批处理完成!")
        
        # 结果摘要表格
        summary_data = []
        for result in all_results:
            summary_data.append({
                "文件名": result['file_name'],
                "状态": result['status'],
                "消息": result['message'],
                "总分子数": result['total_size'],
                "子集大小": result['subset_size'],
                "子集比例": f"{result['subset_size']/result['total_size']:.2%}" if result['total_size'] > 0 else "N/A",
                "处理时间": f"{result['duration']:.2f}秒"
            })
        
        st.subheader("处理结果摘要")
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df)
        
        # 下载选项
        st.subheader("下载结果")
        
        # 合并所有成功的子集为一个ZIP文件
        if any(r['status'] == 'success' for r in all_results):
            # 准备ZIP文件
            import zipfile
            
            with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                with zipfile.ZipFile(tmp.name, 'w') as zipf:
                    # 添加CSV文件
                    for result in all_results:
                        if result['status'] == 'success' and result['subset_data'] is not None:
                            # 创建CSV
                            csv_buffer = io.StringIO()
                            result['subset_data'].to_csv(csv_buffer, index=False)
                            
                            # 生成文件名
                            file_base = os.path.splitext(result['file_name'])[0]
                            csv_name = f"{file_base}_subset.csv"
                            
                            # 添加到ZIP
                            zipf.writestr(csv_name, csv_buffer.getvalue())
                            
                            # 如果有验证统计，添加JSON
                            if result['validation_data'] is not None:
                                json_name = f"{file_base}_validation.json"
                                zipf.writestr(json_name, json.dumps(result['validation_data']))
                    
                    # 添加批处理配置
                    zipf.writestr('batch_config.json', json.dumps(config))
                
                # 读取ZIP文件
                with open(tmp.name, 'rb') as f:
                    zip_data = f.read()
                
                # 提供下载
                st.download_button(
                    label="下载所有子集和验证数据 (ZIP)",
                    data=zip_data,
                    file_name="molecular_subsets.zip",
                    mime="application/zip"
                )
                
                # 删除临时文件
                os.unlink(tmp.name)
        
        # 提供单个子集的下载
        for i, result in enumerate(all_results):
            if result['status'] == 'success' and result['subset_data'] is not None:
                with st.expander(f"下载 {result['file_name']} 的子集"):
                    # CSV下载
                    csv_buffer = io.StringIO()
                    result['subset_data'].to_csv(csv_buffer, index=False)
                    
                    st.download_button(
                        label=f"下载CSV格式子集",
                        data=csv_buffer.getvalue(),
                        file_name=f"{os.path.splitext(result['file_name'])[0]}_subset.csv",
                        mime="text/csv",
                        key=f"csv_download_{i}"
                    )
                    
                    # 显示验证统计
                    if result['validation_data'] is not None:
                        metrics = result['validation_data']['coverage_metrics']
                        nn_stats = result['validation_data']['nn_stats']
                        
                        st.write("验证统计:")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("覆盖率", f"{metrics['coverage_ratio']:.2%}")
                        with col2:
                            st.metric("平均距离", f"{nn_stats['mean_dist']:.3f}")
                        with col3:
                            st.metric("中位数距离", f"{nn_stats['median_dist']:.3f}")
else:
    st.info("请上传包含SMILES数据的CSV文件以开始批处理")

# 页面底部信息
st.markdown("---")
st.info("""
**批处理功能说明**

1. 可以同时上传多个CSV文件，每个文件包含SMILES列
2. 所有文件将使用相同的参数配置进行处理
3. 结果将包含所有子集和验证统计
4. 支持并行处理多个文件，提高效率
""") 