"""
CADD-Toolbox - 多样性筛选页面
基于分子描述符进行多样性筛选，支持GPU加速和多种算法
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import traceback

# 尝试导入PyTorch以支持GPU加速
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

st.set_page_config(page_title="多样性筛选", layout="wide")
st.title("🎯 多样性筛选")

st.markdown("""
基于分子描述符进行多样性筛选，从大规模化合物库中选择代表性子集。

🔧 **支持算法**: 贪心算法、球体排除法  
📏 **距离度量**: 欧氏距离、曼哈顿距离、余弦距离  
⚡ **加速选项**: GPU加速（PyTorch）、CPU并行处理  
📊 **输出格式**: 保持原始CSV格式和列结构  
""")

# 检查GPU可用性
if TORCH_AVAILABLE:
    if CUDA_AVAILABLE:
        st.success(f"✅ GPU加速可用 (PyTorch {torch.__version__})")
        gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "未知"
        st.info(f"🎮 GPU设备: {gpu_info}")
    else:
        st.warning("⚠️ PyTorch已安装但CUDA不可用，将使用CPU")
else:
    st.warning("⚠️ 未安装PyTorch，仅支持CPU计算")

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

def calculate_euclidean_distance_cpu(data_np, selected_indices, candidate_indices=None):
    """CPU版本欧氏距离计算"""
    if candidate_indices is None:
        candidate_indices = np.arange(len(data_np))
    
    # 计算候选点到已选集合的最小距离
    min_distances = np.full(len(data_np), np.inf)
    
    for selected_idx in selected_indices:
        distances = np.sum((data_np - data_np[selected_idx])**2, axis=1)
        min_distances = np.minimum(min_distances, distances)
    
    # 已选点距离设为0
    min_distances[selected_indices] = 0.0
    
    return min_distances

def calculate_manhattan_distance_cpu(data_np, selected_indices):
    """CPU版本曼哈顿距离计算"""
    min_distances = np.full(len(data_np), np.inf)
    
    for selected_idx in selected_indices:
        distances = np.sum(np.abs(data_np - data_np[selected_idx]), axis=1)
        min_distances = np.minimum(min_distances, distances)
    
    min_distances[selected_indices] = 0.0
    return min_distances

def calculate_cosine_distance_cpu(data_np, selected_indices):
    """CPU版本余弦距离计算"""
    # L2归一化
    data_norm = data_np / (np.linalg.norm(data_np, axis=1, keepdims=True) + 1e-8)
    
    min_distances = np.full(len(data_np), np.inf)
    
    for selected_idx in selected_indices:
        # 余弦相似度 = 点积 (归一化后)
        similarities = np.dot(data_norm, data_norm[selected_idx])
        distances = 1 - similarities  # 余弦距离 = 1 - 余弦相似度
        min_distances = np.minimum(min_distances, distances)
    
    min_distances[selected_indices] = 0.0
    return min_distances

def calculate_euclidean_distance_gpu(data_tensor, selected_indices):
    """GPU版本欧氏距离计算"""
    device = data_tensor.device
    min_distances = torch.full((len(data_tensor),), float('inf'), device=device)
    
    for selected_idx in selected_indices:
        diff = data_tensor - data_tensor[selected_idx]
        distances = torch.sum(diff * diff, dim=1)
        min_distances = torch.minimum(min_distances, distances)
    
    # 已选点距离设为0
    min_distances[selected_indices] = 0.0
    return min_distances

def calculate_manhattan_distance_gpu(data_tensor, selected_indices):
    """GPU版本曼哈顿距离计算"""
    device = data_tensor.device
    min_distances = torch.full((len(data_tensor),), float('inf'), device=device)
    
    for selected_idx in selected_indices:
        distances = torch.sum(torch.abs(data_tensor - data_tensor[selected_idx]), dim=1)
        min_distances = torch.minimum(min_distances, distances)
    
    min_distances[selected_indices] = 0.0
    return min_distances

def calculate_cosine_distance_gpu(data_tensor, selected_indices):
    """GPU版本余弦距离计算"""
    device = data_tensor.device
    # L2归一化
    data_norm = torch.nn.functional.normalize(data_tensor, p=2.0, dim=1)
    
    min_distances = torch.full((len(data_tensor),), float('inf'), device=device)
    
    for selected_idx in selected_indices:
        similarities = torch.matmul(data_norm, data_norm[selected_idx].unsqueeze(1)).squeeze()
        distances = 1 - similarities
        min_distances = torch.minimum(min_distances, distances)
    
    min_distances[selected_indices] = 0.0
    return min_distances

def greedy_selection_cpu(data_np, subset_size, distance_method='euclidean', initial_method='random', seed=42):
    """CPU版本贪心算法多样性筛选"""
    np.random.seed(seed)
    N = len(data_np)
    
    # 选择初始点
    if initial_method == 'random':
        initial_idx = np.random.randint(0, N)
    elif initial_method == 'centroid':
        # 选择距离质心最远的点
        centroid = np.mean(data_np, axis=0)
        distances = np.sum((data_np - centroid)**2, axis=1)
        initial_idx = np.argmax(distances)
    else:  # 'first'
        initial_idx = 0
    
    selected_indices = [initial_idx]
    
    # 选择距离计算函数
    if distance_method == 'euclidean':
        distance_fn = calculate_euclidean_distance_cpu
    elif distance_method == 'manhattan':
        distance_fn = calculate_manhattan_distance_cpu
    elif distance_method == 'cosine':
        distance_fn = calculate_cosine_distance_cpu
    else:
        raise ValueError(f"不支持的距离方法: {distance_method}")
    
    # 贪心迭代选择
    for i in range(1, subset_size):
        # 计算所有点到已选集合的最小距离
        min_distances = distance_fn(data_np, selected_indices)
        
        # 选择距离最远的点
        farthest_idx = np.argmax(min_distances)
        selected_indices.append(farthest_idx)
        
        if (i + 1) % 100 == 0:
            st.write(f"已选择 {i + 1}/{subset_size} 个分子")
    
    return selected_indices

def greedy_selection_gpu(data_tensor, subset_size, distance_method='euclidean', initial_method='random', seed=42):
    """GPU版本贪心算法多样性筛选"""
    torch.manual_seed(seed)
    device = data_tensor.device
    N = len(data_tensor)
    
    # 选择初始点
    if initial_method == 'random':
        initial_idx = torch.randint(0, N, (1,), device=device).item()
    elif initial_method == 'centroid':
        # 选择距离质心最远的点
        centroid = torch.mean(data_tensor, dim=0, keepdim=True)
        if distance_method == 'cosine':
            data_norm = torch.nn.functional.normalize(data_tensor, p=2.0, dim=1)
            centroid_norm = torch.nn.functional.normalize(centroid, p=2.0, dim=1)
            distances = 1 - torch.matmul(data_norm, centroid_norm.T).squeeze()
        else:
            distances = torch.sum((data_tensor - centroid)**2, dim=1)
        initial_idx = torch.argmax(distances).item()
    else:  # 'first'
        initial_idx = 0
    
    selected_indices = [initial_idx]
    
    # 选择距离计算函数
    if distance_method == 'euclidean':
        distance_fn = calculate_euclidean_distance_gpu
    elif distance_method == 'manhattan':
        distance_fn = calculate_manhattan_distance_gpu
    elif distance_method == 'cosine':
        distance_fn = calculate_cosine_distance_gpu
    else:
        raise ValueError(f"不支持的距离方法: {distance_method}")
    
    # 贪心迭代选择
    for i in range(1, subset_size):
        # 计算所有点到已选集合的最小距离
        min_distances = distance_fn(data_tensor, selected_indices)
        
        # 选择距离最远的点
        farthest_idx = torch.argmax(min_distances).item()
        selected_indices.append(farthest_idx)
        
        if (i + 1) % 100 == 0:
            st.write(f"已选择 {i + 1}/{subset_size} 个分子")
    
    return selected_indices

def sphere_exclusion_cpu(data_np, subset_size, distance_method='euclidean', radius=None, seed=42):
    """CPU版本球体排除法多样性筛选"""
    np.random.seed(seed)
    N = len(data_np)
    
    if radius is None:
        # 自动估计合适的半径
        sample_size = min(1000, N)
        sample_indices = np.random.choice(N, sample_size, replace=False)
        sample_data = data_np[sample_indices]
        
        if distance_method == 'euclidean':
            pairwise_dists = np.sqrt(np.sum((sample_data[:, None] - sample_data[None, :])**2, axis=2))
        elif distance_method == 'manhattan':
            pairwise_dists = np.sum(np.abs(sample_data[:, None] - sample_data[None, :]), axis=2)
        elif distance_method == 'cosine':
            sample_norm = sample_data / (np.linalg.norm(sample_data, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(sample_norm, sample_norm.T)
            pairwise_dists = 1 - similarities
        
        # 使用分位数作为半径估计
        radius = np.percentile(pairwise_dists[pairwise_dists > 0], 20)
        st.info(f"自动估计排除半径: {radius:.4f}")
    
    selected_indices = []
    excluded = np.zeros(N, dtype=bool)
    
    while len(selected_indices) < subset_size and not np.all(excluded):
        # 随机选择一个未被排除的点
        available = np.where(~excluded)[0]
        if len(available) == 0:
            break
        
        current_idx = np.random.choice(available)
        selected_indices.append(current_idx)
        
        # 计算当前点到所有点的距离并排除在半径内的点
        if distance_method == 'euclidean':
            distances = np.sqrt(np.sum((data_np - data_np[current_idx])**2, axis=1))
        elif distance_method == 'manhattan':
            distances = np.sum(np.abs(data_np - data_np[current_idx]), axis=1)
        elif distance_method == 'cosine':
            data_norm = data_np / (np.linalg.norm(data_np, axis=1, keepdims=True) + 1e-8)
            current_norm = data_norm[current_idx]
            similarities = np.dot(data_norm, current_norm)
            distances = 1 - similarities
        
        # 排除在半径内的点
        excluded |= (distances <= radius)
        
        if len(selected_indices) % 100 == 0:
            st.write(f"已选择 {len(selected_indices)}/{subset_size} 个分子")
    
    return selected_indices

def sphere_exclusion_gpu(data_tensor, subset_size, distance_method='euclidean', radius=None, seed=42):
    """GPU版本球体排除法多样性筛选"""
    torch.manual_seed(seed)
    device = data_tensor.device
    N = len(data_tensor)
    
    if radius is None:
        # 自动估计合适的半径
        sample_size = min(1000, N)
        sample_indices = torch.randperm(N, device=device)[:sample_size]
        sample_data = data_tensor[sample_indices]
        
        if distance_method == 'euclidean':
            diff = sample_data.unsqueeze(1) - sample_data.unsqueeze(0)
            pairwise_dists = torch.sqrt(torch.sum(diff * diff, dim=2))
        elif distance_method == 'manhattan':
            diff = sample_data.unsqueeze(1) - sample_data.unsqueeze(0)
            pairwise_dists = torch.sum(torch.abs(diff), dim=2)
        elif distance_method == 'cosine':
            sample_norm = torch.nn.functional.normalize(sample_data, p=2.0, dim=1)
            similarities = torch.matmul(sample_norm, sample_norm.T)
            pairwise_dists = 1 - similarities
        
        # 使用分位数作为半径估计
        mask = pairwise_dists > 0
        valid_dists = pairwise_dists[mask]
        radius = torch.quantile(valid_dists, 0.2).item()
        st.info(f"自动估计排除半径: {radius:.4f}")
    
    selected_indices = []
    excluded = torch.zeros(N, dtype=torch.bool, device=device)
    
    while len(selected_indices) < subset_size and not torch.all(excluded):
        # 随机选择一个未被排除的点
        available = torch.where(~excluded)[0]
        if len(available) == 0:
            break
        
        rand_idx = torch.randint(0, len(available), (1,), device=device).item()
        current_idx = available[rand_idx].item()
        selected_indices.append(current_idx)
        
        # 计算当前点到所有点的距离并排除在半径内的点
        if distance_method == 'euclidean':
            distances = torch.sqrt(torch.sum((data_tensor - data_tensor[current_idx])**2, dim=1))
        elif distance_method == 'manhattan':
            distances = torch.sum(torch.abs(data_tensor - data_tensor[current_idx]), dim=1)
        elif distance_method == 'cosine':
            data_norm = torch.nn.functional.normalize(data_tensor, p=2.0, dim=1)
            current_norm = data_norm[current_idx]
            similarities = torch.matmul(data_norm, current_norm.unsqueeze(1)).squeeze()
            distances = 1 - similarities
        
        # 排除在半径内的点
        excluded |= (distances <= radius)
        
        if len(selected_indices) % 100 == 0:
            st.write(f"已选择 {len(selected_indices)}/{subset_size} 个分子")
    
    return selected_indices

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
    
    selected_file = st.selectbox("选择描述符CSV文件:", csv_files)
    
    if selected_file:
        file_path = os.path.join(DATA_DIR, selected_folder, selected_file)
        
        # 显示文件信息
        file_info = get_file_info(file_path)
        if file_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("文件大小", f"{file_info['size_mb']:.1f} MB")
            with col2:
                st.metric("修改时间", file_info['modified'])
            with col3:
                # 快速统计行数
                with st.spinner("统计分子数..."):
                    try:
                        df_shape = pd.read_csv(file_path, nrows=0).shape
                        # 快速计算行数
                        with open(file_path, 'r') as f:
                            row_count = sum(1 for line in f) - 1  # 减去表头
                        st.metric("分子数量", row_count)
                    except Exception as e:
                        st.metric("分子数量", "读取失败")

        # 筛选设置
        st.subheader("2. 筛选设置")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**计算设置**")
            
            # GPU/CPU选择
            if TORCH_AVAILABLE and CUDA_AVAILABLE:
                use_gpu = st.checkbox("使用GPU加速", value=True, 
                                    help="启用GPU可显著加速大规模数据的筛选")
            else:
                use_gpu = False
                st.info("GPU不可用，将使用CPU计算")
            
            # 距离算法选择
            distance_method = st.selectbox(
                "距离算法:",
                ["euclidean", "manhattan", "cosine"],
                format_func=lambda x: {
                    "euclidean": "欧氏距离 (L2)",
                    "manhattan": "曼哈顿距离 (L1)", 
                    "cosine": "余弦距离"
                }[x],
                help="选择用于计算分子间距离的度量方法"
            )
            
            # 筛选算法选择
            selection_algorithm = st.selectbox(
                "筛选算法:",
                ["greedy", "sphere_exclusion"],
                format_func=lambda x: {
                    "greedy": "贪心算法 (MaxMin)",
                    "sphere_exclusion": "球体排除法"
                }[x],
                help="贪心算法：逐步选择距离已选集合最远的分子\n球体排除法：排除选定分子周围一定半径内的分子"
            )
        
        with col2:
            st.markdown("**筛选参数**")
            
            # 子集大小设置
            subset_method = st.radio(
                "子集大小设置:",
                ["按比例", "按数量"],
                help="选择如何确定筛选后的子集大小"
            )
            
            if subset_method == "按比例":
                subset_ratio = st.slider(
                    "筛选比例 (%):",
                    0.1, 50.0, 10.0, 0.5,
                    help="选择保留原数据集的百分比"
                )
                subset_size = None
            else:
                subset_size = st.number_input(
                    "筛选数量:",
                    min_value=1,
                    max_value=100000,
                    value=1000,
                    help="直接指定要选择的分子数量"
                )
                subset_ratio = None
            
            # 初始点选择方法（仅贪心算法）
            if selection_algorithm == "greedy":
                initial_method = st.selectbox(
                    "初始点选择:",
                    ["random", "centroid", "first"],
                    format_func=lambda x: {
                        "random": "随机选择",
                        "centroid": "距质心最远",
                        "first": "第一个分子"
                    }[x],
                    help="选择第一个分子的策略"
                )
            else:
                initial_method = "random"
            
            # 球体排除法的半径设置
            if selection_algorithm == "sphere_exclusion":
                auto_radius = st.checkbox("自动估计排除半径", value=True,
                                        help="根据数据分布自动确定排除半径")
                if not auto_radius:
                    exclusion_radius = st.number_input(
                        "排除半径:",
                        min_value=0.001,
                        max_value=10.0,
                        value=0.1,
                        format="%.4f",
                        help="在此半径内的分子将被排除"
                    )
                else:
                    exclusion_radius = None
            else:
                exclusion_radius = None
            
            # 随机种子
            random_seed = st.number_input(
                "随机种子:",
                min_value=0,
                max_value=9999,
                value=42,
                help="设置随机种子以确保结果可重现"
            )

        # 开始筛选按钮
        if st.button("🚀 开始多样性筛选", type="primary"):
            try:
                # 读取数据
                with st.spinner("读取描述符数据..."):
                    df = pd.read_csv(file_path)
                    st.success(f"✅ 成功读取 {len(df)} 个分子的描述符数据")
                
                # 检查数据
                if len(df) == 0:
                    st.error("❌ 数据文件为空")
                    st.stop()
                
                # 分离数值列和非数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                
                if len(numeric_cols) == 0:
                    st.error("❌ 未找到数值类型的描述符列")
                    st.stop()
                
                st.info(f"🔢 找到 {len(numeric_cols)} 个数值描述符列，{len(non_numeric_cols)} 个非数值列")
                
                # 处理缺失值
                descriptor_data = df[numeric_cols].copy()
                
                # 检查和处理NaN值
                nan_count = descriptor_data.isna().sum().sum()
                if nan_count > 0:
                    st.warning(f"⚠️ 发现 {nan_count} 个NaN值，将用0填充")
                    descriptor_data = descriptor_data.fillna(0)
                
                # 检查和处理无穷大值
                inf_count = np.isinf(descriptor_data.values).sum()
                if inf_count > 0:
                    st.warning(f"⚠️ 发现 {inf_count} 个无穷大值，将用0替换")
                    descriptor_data = descriptor_data.replace([np.inf, -np.inf], 0)
                
                # 确定最终的子集大小
                total_molecules = len(df)
                if subset_method == "按比例":
                    final_subset_size = max(1, int(total_molecules * subset_ratio / 100))
                else:
                    final_subset_size = min(subset_size, total_molecules)
                
                st.info(f"🎯 将从 {total_molecules} 个分子中选择 {final_subset_size} 个代表性分子")
                
                # 执行筛选
                start_time = time.time()
                
                with st.spinner("执行多样性筛选..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    if use_gpu and TORCH_AVAILABLE and CUDA_AVAILABLE:
                        # GPU计算
                        status_text.text("准备GPU数据...")
                        device = torch.device('cuda')
                        data_tensor = torch.tensor(descriptor_data.values, dtype=torch.float32, device=device)
                        
                        progress_bar.progress(0.1)
                        status_text.text("GPU筛选进行中...")
                        
                        if selection_algorithm == "greedy":
                            selected_indices = greedy_selection_gpu(
                                data_tensor, final_subset_size, distance_method, initial_method, random_seed
                            )
                        else:  # sphere_exclusion
                            selected_indices = sphere_exclusion_gpu(
                                data_tensor, final_subset_size, distance_method, exclusion_radius, random_seed
                            )
                        
                        # 清理GPU内存
                        del data_tensor
                        torch.cuda.empty_cache()
                    
                    else:
                        # CPU计算
                        status_text.text("准备CPU数据...")
                        data_np = descriptor_data.values.astype(np.float32)
                        
                        progress_bar.progress(0.1)
                        status_text.text("CPU筛选进行中...")
                        
                        if selection_algorithm == "greedy":
                            selected_indices = greedy_selection_cpu(
                                data_np, final_subset_size, distance_method, initial_method, random_seed
                            )
                        else:  # sphere_exclusion
                            selected_indices = sphere_exclusion_cpu(
                                data_np, final_subset_size, distance_method, exclusion_radius, random_seed
                            )
                    
                    progress_bar.progress(1.0)
                    status_text.text("筛选完成！")
                
                # 提取筛选结果
                diverse_subset_df = df.iloc[selected_indices].copy()
                
                elapsed_time = time.time() - start_time
                
                # 显示结果
                st.subheader("3. 筛选结果")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("原始分子数", total_molecules)
                with col2:
                    st.metric("筛选后分子数", len(diverse_subset_df))
                with col3:
                    st.metric("筛选比例", f"{len(diverse_subset_df)/total_molecules*100:.1f}%")
                
                st.success(f"✅ 多样性筛选完成！用时 {elapsed_time:.1f} 秒")
                
                # 保存结果
                output_filename = f"diverse_subset_{selected_file.replace('.csv', '')}_{len(diverse_subset_df)}compounds_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                output_path = os.path.join(DATA_DIR, selected_folder, output_filename)
                
                with st.spinner("保存筛选结果..."):
                    diverse_subset_df.to_csv(output_path, index=False)
                
                st.success(f"📁 结果已保存到: {output_filename}")
                
                # 提供下载按钮
                csv_data = diverse_subset_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载筛选结果CSV文件",
                    data=csv_data,
                    file_name=output_filename,
                    mime="text/csv"
                )
                
                # 数据预览
                st.subheader("4. 结果预览")
                
                # 显示筛选参数
                st.markdown("**筛选参数:**")
                param_col1, param_col2 = st.columns(2)
                with param_col1:
                    st.text(f"计算设备: {'GPU' if use_gpu and TORCH_AVAILABLE and CUDA_AVAILABLE else 'CPU'}")
                    st.text(f"距离算法: {distance_method}")
                    st.text(f"筛选算法: {selection_algorithm}")
                with param_col2:
                    st.text(f"初始点选择: {initial_method}")
                    st.text(f"随机种子: {random_seed}")
                    if exclusion_radius is not None:
                        st.text(f"排除半径: {exclusion_radius}")
                
                # 数据表预览
                st.markdown("**筛选结果预览（前10行）:**")
                st.dataframe(diverse_subset_df.head(10))
                
                # 显示文件大小
                output_size = len(csv_data.encode('utf-8')) / (1024 * 1024)
                st.info(f"输出文件大小: {output_size:.2f} MB")
                
            except Exception as e:
                st.error(f"❌ 筛选过程中发生错误: {str(e)}")
                st.error("详细错误信息:")
                st.code(traceback.format_exc()) 