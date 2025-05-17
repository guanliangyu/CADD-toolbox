"""
应用程序配置管理工具 - 管理和生成配置字典
"""
import streamlit as st
import multiprocessing as mp
import os

def create_config_from_parameters(
    smiles_col="SMILES",
    subset_ratio=1.0,
    clustering_method="kmeans",
    fps_type="morgan",
    include_3d=False,
    include_charges=False,
    batch_size=1000,
    n_jobs=4,
    use_gpu=False,
    cutoff=0.6,
    n_clusters=100,
    use_fixed_clusters=False,
    kmeans_iterations=100,
    init_method="random",
    morgan_radius=2,
    morgan_bits=1024,
    gpu_config=None
):
    """
    从参数创建配置字典
    
    参数:
        smiles_col: SMILES列名
        subset_ratio: 子集比例
        clustering_method: 聚类方法
        fps_type: 指纹类型
        include_3d: 是否包含3D特征
        include_charges: 是否包含电荷
        batch_size: 批处理大小
        n_jobs: 并行作业数
        use_gpu: 是否使用GPU
        cutoff: Butina聚类阈值
        n_clusters: K-means聚类数量
        use_fixed_clusters: 是否使用固定簇数量
        kmeans_iterations: K-means最大迭代次数
        init_method: MaxMin初始点选择方法
        morgan_radius: Morgan指纹半径
        morgan_bits: Morgan指纹位数
        gpu_config: GPU配置字典
        
    返回:
        配置字典
    """
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
                'n_clusters': n_clusters if clustering_method == 'kmeans' and use_fixed_clusters else 100,
                'max_iter': kmeans_iterations if clustering_method == 'kmeans' else 100,
                'use_ratio': not use_fixed_clusters
            },
            'maxmin': {
                'init_method': init_method if clustering_method == 'maxmin' else 'random'
            }
        },
        'subset_ratio': subset_ratio
    }
    
    # 添加GPU配置
    if use_gpu:
        if gpu_config is None:
            gpu_config = {
                'enabled': True,
                'device_id': 0,
                'auto_detect': True,
                'features': {
                    'kmeans': True,
                    'distances': True,
                    'pca': True
                }
            }
        
        config['gpu'] = gpu_config
        
        # 如果设置了GPU设备ID，设置CUDA环境变量
        if 'device_id' in gpu_config:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_config['device_id'])
    
    return config

def render_clustering_parameters():
    """
    渲染聚类参数选择界面
    
    返回:
        聚类参数字典
    """
    st.sidebar.subheader("聚类方法")
    clustering_method = st.sidebar.selectbox(
        "选择聚类/选择算法",
        ["butina", "kmeans", "maxmin"],
        key="clustering_method"
    )
    
    params = {
        'clustering_method': clustering_method
    }
    
    # 根据聚类方法显示不同参数
    if clustering_method == "butina":
        cutoff = st.sidebar.slider("相似度阈值", 0.0, 1.0, 0.6, 0.05, key="cutoff")
        params['cutoff'] = cutoff
        
    elif clustering_method == "kmeans":
        use_fixed_clusters = st.sidebar.checkbox("使用固定簇数量", value=False, key="use_fixed_clusters")
        params['use_fixed_clusters'] = use_fixed_clusters
        
        if use_fixed_clusters:
            n_clusters = st.sidebar.number_input("簇数量", 10, 10000, 100, key="n_clusters")
            params['n_clusters'] = n_clusters
        else:
            subset_ratio = st.session_state.get('subset_ratio', 1.0)
            st.sidebar.info(f"将使用子集比例 ({subset_ratio}%) 计算簇数量")
            
        kmeans_iterations = st.sidebar.slider("最大迭代次数", 10, 1000, 100, key="kmeans_iterations")
        params['kmeans_iterations'] = kmeans_iterations
        
    elif clustering_method == "maxmin":
        init_method = st.sidebar.selectbox("初始点选择", ["random", "first"], key="init_method")
        params['init_method'] = init_method
    
    return params

def render_feature_parameters():
    """
    渲染特征计算参数选择界面
    
    返回:
        特征参数字典
    """
    st.sidebar.subheader("特征计算")
    fps_type = st.sidebar.selectbox("指纹类型", ["morgan", "rdkit", "maccs"], key="fps_type")
    
    params = {
        'fps_type': fps_type
    }
    
    if fps_type == "morgan":
        morgan_radius = st.sidebar.slider("Morgan半径", 1, 4, 2, key="morgan_radius")
        morgan_bits = st.sidebar.selectbox("Morgan位数", [512, 1024, 2048], key="morgan_bits")
        params['morgan_radius'] = morgan_radius
        params['morgan_bits'] = morgan_bits

    include_3d = st.sidebar.checkbox("生成3D构象", value=False, key="include_3d")
    params['include_3d'] = include_3d
    
    if include_3d:
        include_charges = st.sidebar.checkbox("计算Gasteiger电荷", value=False, key="include_charges")
        params['include_charges'] = include_charges
    
    return params

def render_performance_parameters():
    """
    渲染性能参数选择界面
    
    返回:
        性能参数字典
    """
    st.sidebar.subheader("计算参数")
    batch_size = st.sidebar.slider("批处理大小", 100, 5000, 1000, 100, key="batch_size")
    n_jobs = st.sidebar.slider("并行作业数", 1, mp.cpu_count(), min(4, mp.cpu_count()), key="n_jobs")
    
    params = {
        'batch_size': batch_size,
        'n_jobs': n_jobs
    }
    
    # 导入GPU工具
    try:
        from utils.gpu_utils import check_gpu_availability
        GPU_TOOLS_AVAILABLE = True
    except ImportError:
        GPU_TOOLS_AVAILABLE = False
        
    # GPU加速设置
    if GPU_TOOLS_AVAILABLE:
        st.sidebar.subheader("GPU加速")
        use_gpu = st.sidebar.checkbox("启用GPU加速", value=True, key="use_gpu")
        params['use_gpu'] = use_gpu
        
        if use_gpu:
            gpu_status = check_gpu_availability()
            if gpu_status['any_gpu']:
                st.sidebar.success("检测到可用GPU")
                
                # 显示可用的GPU库
                gpu_libs = [k for k, v in gpu_status.items() if v and k != 'any_gpu']
                if gpu_libs:
                    st.sidebar.info(f"可用GPU库: {', '.join(gpu_libs)}")
                
                # GPU设置选项
                gpu_id = st.sidebar.number_input("GPU设备ID", 0, 7, 0, 1, key="gpu_id")
                params['gpu_id'] = gpu_id
                
                # 启用的功能
                st.sidebar.subheader("GPU加速功能")
                gpu_kmeans = st.sidebar.checkbox("K-means聚类", value=True, key="gpu_kmeans")
                gpu_distances = st.sidebar.checkbox("距离计算", value=True, key="gpu_distances")
                gpu_pca = st.sidebar.checkbox("PCA降维", value=True, key="gpu_pca")
                
                params['gpu_config'] = {
                    'enabled': True,
                    'device_id': gpu_id,
                    'auto_detect': True,
                    'features': {
                        'kmeans': gpu_kmeans,
                        'distances': gpu_distances,
                        'pca': gpu_pca
                    }
                }
            else:
                st.sidebar.warning("未检测到可用GPU，将使用CPU计算")
                params['use_gpu'] = False
    else:
        params['use_gpu'] = False
    
    return params

def render_sidebar_parameters():
    """
    在侧边栏渲染所有参数选择界面
    
    返回:
        包含所有参数的字典
    """
    st.sidebar.header("配置参数")

    # 文件上传区域
    uploaded_file = st.sidebar.file_uploader("上传SMILES文件 (CSV)", type=["csv"])
    
    # 基本参数
    st.sidebar.subheader("基本参数")
    smiles_col = st.sidebar.text_input("SMILES列名", "SMILES", key="smiles_col")
    subset_ratio = st.sidebar.slider("子集比例 (%)", 0.1, 10.0, 1.0, 0.1, key="subset_ratio")
    
    # 聚类参数
    clustering_params = render_clustering_parameters()
    
    # 特征计算参数
    feature_params = render_feature_parameters()
    
    # 性能参数
    performance_params = render_performance_parameters()
    
    # 合并所有参数
    all_params = {
        'uploaded_file': uploaded_file,
        'smiles_col': smiles_col,
        'subset_ratio': subset_ratio,
        **clustering_params,
        **feature_params,
        **performance_params
    }
    
    return all_params 