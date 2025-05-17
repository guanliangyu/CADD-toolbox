"""
分子库代表性子集选择系统 - 可视化分析页面
"""
import os
import sys
import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import matplotlib.pyplot as plt

# 添加项目根目录到路径，确保能导入utils模块
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

# 导入工具模块
from utils.visualization_utils import (
    plot_property_distributions, plot_fps_pca,
    plot_fps_tsne, plot_fps_umap
)
from utils.file_utils import (
    load_dataframe, load_pickle, load_json,
    file_exists, MOLECULES_FILE, CONFIG_FILE, PROCESSED_RESULTS_FILE,
    CLUSTERING_RESULTS_FILE, SELECTED_SUBSET_FILE
)

# 设置页面标题和配置
st.set_page_config(
    page_title="分子库代表性子集选择系统 - 可视化分析",
    page_icon="🧪",
    layout="wide"
)

# 页面标题
st.title("可视化分析")

# 主内容区域
if not file_exists(PROCESSED_RESULTS_FILE):
    st.info("请先在数据处理页面处理分子数据")
    if st.button("前往数据处理页面"):
        st.switch_page("pages/1_数据处理.py")
else:
    # 加载数据
    df = load_dataframe(MOLECULES_FILE)
    processed_results = load_pickle(PROCESSED_RESULTS_FILE)
    config = load_json(CONFIG_FILE)
    subset_indices = load_pickle(SELECTED_SUBSET_FILE) if file_exists(SELECTED_SUBSET_FILE) else None
    
    # 创建两列布局
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        # 创建选项卡
        tab1, tab2, tab3 = st.tabs(["分子空间可视化", "分子结构浏览", "聚类结果分析"])
        
        with tab1:
            st.subheader("分子空间可视化")
            
            # 降维方法选择
            dim_method = st.selectbox(
                "降维方法",
                ["PCA", "t-SNE", "UMAP"]
            )
            
            # 根据方法提供相应参数
            params_col1, params_col2 = st.columns(2)
            
            with params_col1:
                n_components = st.slider("降维维度", 2, 3, 2)
                
                if dim_method == "t-SNE":
                    perplexity = st.slider("困惑度", 5, 100, 30)
                    early_exaggeration = st.slider("早期夸大", 1, 50, 12)
                elif dim_method == "UMAP":
                    n_neighbors = st.slider("邻居数", 2, 100, 15)
                    min_dist = st.slider("最小距离", 0.0, 1.0, 0.1)
            
            # 绘图参数
            with params_col2:
                show_subset = st.checkbox("突出显示子集", value=True)
                color_by = st.selectbox(
                    "着色依据",
                    ["子集/原始", "分子量", "LogP", "TPSA", "HBA", "HBD"]
                )
                plot_3d = st.checkbox("3D绘图", value=n_components == 3)
            
            # 降维和绘图按钮
            if st.button("生成可视化"):
                with st.spinner(f"使用{dim_method}降维中..."):
                    # 提取有效分子的特征
                    valid_idx = [i for i, m in enumerate(processed_results['mols']) if m is not None]
                    features = np.array([processed_results['fps'][i] for i in valid_idx if processed_results['fps'][i] is not None])
                    
                    if len(features) == 0:
                        st.error("没有有效特征可供降维")
                    else:
                        # 执行降维
                        if dim_method == "PCA":
                            fig = plot_fps_pca(features, valid_idx, subset_indices, n_components, color_by, plot_3d)
                        elif dim_method == "t-SNE":
                            fig = plot_fps_tsne(features, valid_idx, subset_indices, n_components, perplexity, early_exaggeration, color_by, plot_3d)
                        else:  # UMAP
                            fig = plot_fps_umap(features, valid_idx, subset_indices, n_components, n_neighbors, min_dist, color_by, plot_3d)
                        
                        st.pyplot(fig)
        
        with tab2:
            st.subheader("分子结构浏览器")
            
            # 过滤选项
            browse_option = st.radio(
                "浏览选项",
                ["所有分子", "仅代表分子", "分子属性筛选"]
            )
            
            if browse_option == "所有分子":
                # 显示所有有效分子
                valid_idx = [i for i, m in enumerate(processed_results['mols']) if m is not None]
                target_indices = valid_idx
            elif browse_option == "仅代表分子":
                if subset_indices is None:
                    st.warning("尚未选择代表性分子，请先进行聚类和选择")
                    target_indices = []
                else:
                    target_indices = subset_indices
            else:  # 分子属性筛选
                # 添加属性范围滑块
                st.write("设置属性范围")
                mw_range = st.slider("分子量范围", 0, 1000, (0, 1000))
                logp_range = st.slider("LogP范围", -10, 10, (-10, 10))
                
                # 根据属性范围筛选分子
                target_indices = []
                for i, desc in enumerate(processed_results['basic_desc']):
                    if desc is not None:
                        mw = desc.get('mw', 0)
                        logp = desc.get('logp', 0)
                        if mw_range[0] <= mw <= mw_range[1] and logp_range[0] <= logp <= logp_range[1]:
                            target_indices.append(i)
            
            # 分页显示分子
            if target_indices:
                page_size = st.slider("每页显示分子数", 5, 50, 20)
                n_pages = (len(target_indices) + page_size - 1) // page_size
                page = st.selectbox("选择页面", range(1, n_pages + 1)) - 1
                
                start_idx = page * page_size
                end_idx = min(start_idx + page_size, len(target_indices))
                display_indices = target_indices[start_idx:end_idx]
                
                # 显示分子结构
                mols = [processed_results['mols'][i] for i in display_indices]
                legends = [
                    f"{i}: MW={processed_results['basic_desc'][i].get('mw', 0):.1f}, LogP={processed_results['basic_desc'][i].get('logp', 0):.1f}"
                    for i in display_indices
                ]
                
                img = Draw.MolsToGridImage(
                    mols, 
                    molsPerRow=5, 
                    subImgSize=(200, 200),
                    legends=legends
                )
                st.image(img)
            else:
                st.info("没有找到符合条件的分子")
        
        with tab3:
            st.subheader("聚类结果分析")
            
            if subset_indices is None:
                st.info("请先在聚类与选择页面生成子集")
                if st.button("前往聚类与选择页面", key="goto_cluster"):
                    st.switch_page("pages/2_聚类与选择.py")
            else:
                # 加载聚类结果
                clustering_results = load_pickle(CLUSTERING_RESULTS_FILE)
                if clustering_results is None:
                    st.error("无法加载聚类结果")
                else:
                    # 获取聚类方法
                    method = config.get('clustering', {}).get('method', '')
                    
                    # 显示聚类参数
                    st.write(f"聚类方法: **{method}**")
                    st.write(f"选择的代表分子数量: **{len(subset_indices)}**")
                    
                    # 根据不同方法显示特定信息
                    if method == "butina":
                        cutoff = config.get('clustering', {}).get('butina', {}).get('cutoff', 0)
                        st.write(f"相似度阈值: **{cutoff}**")
                        
                        if st.button("分析聚类结果"):
                            with st.spinner("分析聚类结果..."):
                                # 计算聚类统计
                                cluster_sizes = np.bincount(clustering_results['labels'])
                                n_clusters = len(cluster_sizes)
                                avg_size = np.mean(cluster_sizes)
                                max_size = np.max(cluster_sizes)
                                
                                # 显示统计信息
                                st.write(f"聚类数量: **{n_clusters}**")
                                st.write(f"平均簇大小: **{avg_size:.2f}**")
                                st.write(f"最大簇大小: **{max_size}**")
                                
                                # 绘制簇大小分布
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.hist(cluster_sizes, bins=50, alpha=0.7, color='#4DAF4A')
                                ax.set_xlabel('簇大小')
                                ax.set_ylabel('簇数量')
                                ax.set_title('簇大小分布')
                                ax.grid(alpha=0.3)
                                st.pyplot(fig)
                    
                    elif method == "kmeans":
                        n_clusters = config.get('clustering', {}).get('kmeans', {}).get('n_clusters', 0)
                        max_iter = config.get('clustering', {}).get('kmeans', {}).get('max_iter', 0)
                        use_ratio = config.get('clustering', {}).get('kmeans', {}).get('use_ratio', True)
                        
                        if use_ratio:
                            st.write(f"基于比例计算的簇数量: **{len(subset_indices)}**")
                        else:
                            st.write(f"固定簇数量: **{n_clusters}**")
                        
                        if st.button("分析聚类质量"):
                            with st.spinner("计算聚类质量指标..."):
                                from sklearn.metrics import silhouette_score, davies_bouldin_score
                                
                                # 提取特征和标签
                                valid_idx = [i for i, m in enumerate(processed_results['mols']) if m is not None]
                                features = np.array([processed_results['fps'][i] for i in valid_idx if processed_results['fps'][i] is not None])
                                labels = clustering_results['labels']
                                
                                # 计算评估指标
                                sil_score = silhouette_score(features, labels)
                                db_score = davies_bouldin_score(features, labels)
                                
                                # 显示评估结果
                                st.write(f"轮廓系数: **{sil_score:.3f}**")
                                st.write(f"Davies-Bouldin指数: **{db_score:.3f}**")
                                
                                # 绘制聚类结果的PCA可视化
                                fig = plot_fps_pca(features, valid_idx, subset_indices, color_by="cluster")
                                st.pyplot(fig)
                    
                    elif method == "maxmin":
                        init_method = config.get('clustering', {}).get('maxmin', {}).get('init_method', 'random')
                        st.write(f"初始点选择方法: **{init_method}**")
                        
                        if st.button("分析选择覆盖性"):
                            with st.spinner("计算覆盖分析..."):
                                # 计算每个原始分子到最近代表分子的距离
                                valid_idx = [i for i, m in enumerate(processed_results['mols']) if m is not None]
                                valid_fps = [processed_results['fps_binary'][i] for i in valid_idx]
                                subset_fps = [processed_results['fps_binary'][i] for i in subset_indices]
                                
                                from utils.validation_utils import calculate_nearest_neighbor_distance
                                nn_distances, mean_dist, max_dist, median_dist = calculate_nearest_neighbor_distance(
                                    valid_fps, subset_fps
                                )
                                
                                # 显示统计信息
                                st.write(f"平均距离: **{mean_dist:.3f}**")
                                st.write(f"最大距离: **{max_dist:.3f}**")
                                st.write(f"中位数距离: **{median_dist:.3f}**")
                                
                                # 绘制距离分布直方图
                                fig, ax = plt.subplots(figsize=(10, 6))
                                ax.hist(nn_distances, bins=50, alpha=0.7, color='#4DAF4A')
                                ax.set_xlabel('到最近代表分子的距离')
                                ax.set_ylabel('分子数量')
                                ax.set_title('MaxMin覆盖分析')
                                ax.grid(alpha=0.3)
                                ax.axvline(x=mean_dist, color='r', linestyle='--', label=f'平均距离: {mean_dist:.3f}')
                                ax.legend()
                                st.pyplot(fig)
                    
                    # 通用分子属性分布比较
                    if st.button("比较属性分布"):
                        with st.spinner("生成属性分布对比..."):
                            valid_idx = [i for i, m in enumerate(processed_results['mols']) if m is not None]
                            valid_basic_desc = [processed_results['basic_desc'][i] for i in valid_idx]
                            subset_basic_desc = [processed_results['basic_desc'][i] for i in subset_indices]
                            
                            # 获取属性分布图
                            prop_names = ['mw', 'logp', 'tpsa', 'hba', 'hbd', 'rotatable_bonds']
                            prop_fig = plot_property_distributions(valid_basic_desc, subset_basic_desc, prop_names)
                            st.pyplot(prop_fig)
    
    with right_col:
        st.subheader("分析控制面板")
        
        # 数据集信息
        st.write("📊 数据集信息")
        total_mols = len(df)
        valid_mols = sum(1 for m in processed_results['mols'] if m is not None)
        st.write(f"- 总分子数：{total_mols}")
        st.write(f"- 有效分子数：{valid_mols}")
        if subset_indices:
            st.write(f"- 选择子集大小：{len(subset_indices)}")
        
        # 可视化选项
        st.write("🎨 可视化选项")
        st.write("- 分子空间可视化")
        st.write("  - PCA")
        st.write("  - t-SNE")
        st.write("  - UMAP")
        
        st.write("- 分子结构浏览")
        st.write("  - 全部分子")
        st.write("  - 代表分子")
        st.write("  - 属性筛选")
        
        st.write("- 聚类分析")
        st.write("  - 聚类统计")
        st.write("  - 质量评估")
        st.write("  - 属性分布")
        
        # 导航按钮
        st.write("⚡ 快速导航")
        if st.button("前往验证与下载"):
            st.switch_page("pages/3_验证与下载.py")
        if st.button("返回聚类与选择"):
            st.switch_page("pages/2_聚类与选择.py") 