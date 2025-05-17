"""
分子库代表性子集选择系统 - 聚类与选择页面
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

# 导入工具模块
from utils.molecular_utils import MoleculeProcessor
from utils.clustering_utils import perform_clustering, evaluate_clustering, cluster_quality_metrics
from utils.visualization_utils import plot_fps_pca, plot_fps_tsne
from utils.file_utils import (
    load_dataframe, load_pickle, load_json, save_pickle, save_json, save_dataframe,
    file_exists, MOLECULES_FILE, CONFIG_FILE, PROCESSED_RESULTS_FILE,
    CLUSTERING_RESULTS_FILE, SELECTED_SUBSET_FILE
)

# 设置页面标题和配置
st.set_page_config(
    page_title="分子库代表性子集选择系统 - 聚类与选择",
    page_icon="🧪",
    layout="wide"
)

# 显示当前状态
def display_status_sidebar():
    """在侧边栏显示当前处理状态"""
    st.sidebar.subheader("当前状态")
    
    # 检查是否有保存的文件
    if file_exists(MOLECULES_FILE):
        df = load_dataframe(MOLECULES_FILE)
        if df is not None:
            st.sidebar.text(f"数据集: {len(df)}条记录")
    
    if file_exists(PROCESSED_RESULTS_FILE):
        processed_results = load_pickle(PROCESSED_RESULTS_FILE)
        if processed_results is not None:
            valid_count = sum(1 for m in processed_results['mols'] if m is not None)
            st.sidebar.text(f"有效分子: {valid_count}个")
    
    if file_exists(CLUSTERING_RESULTS_FILE):
        clustering_results = load_pickle(CLUSTERING_RESULTS_FILE)
        if clustering_results is not None:
            cluster_count = len(set(clustering_results['cluster_labels']))
            st.sidebar.text(f"聚类: {cluster_count}个簇")
    
    if file_exists(SELECTED_SUBSET_FILE):
        subset_df = load_dataframe(SELECTED_SUBSET_FILE)
        if subset_df is not None:
            st.sidebar.text(f"子集: {len(subset_df)}个分子")

# 显示页面标题
st.title("聚类与选择")

# 显示侧边栏状态
display_status_sidebar()

# 检查是否有处理结果
if not file_exists(PROCESSED_RESULTS_FILE) or not file_exists(CONFIG_FILE):
    st.error("未找到处理结果，请先在'数据处理'页面上传并处理分子数据")
    if st.button("返回数据处理页面"):
        st.switch_page("pages/1_数据处理.py")
else:
    # 加载数据和配置
    df = load_dataframe(MOLECULES_FILE)
    config = load_json(CONFIG_FILE)
    processed_results = load_pickle(PROCESSED_RESULTS_FILE)
    
    # 获取有效分子和指纹
    valid_indices = [i for i, m in enumerate(processed_results['mols']) if m is not None]
    valid_count = len(valid_indices)
    
    # 创建左右列布局
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.header("聚类参数")
        
        # 聚类方法
        clustering_method = st.selectbox(
            "选择聚类算法",
            ["butina", "kmeans", "maxmin"],
            index=["butina", "kmeans", "maxmin"].index(config.get('clustering_method', 'butina'))
        )
        
        # 根据聚类方法显示不同参数
        if clustering_method == "butina":
            cutoff = st.slider(
                "相似度阈值", 
                0.0, 1.0, 
                config.get('cutoff', 0.6), 
                0.05
            )
        elif clustering_method == "kmeans":
            use_fixed_clusters = st.checkbox(
                "使用固定簇数量", 
                value=config.get('use_fixed_clusters', False)
            )
            
            if use_fixed_clusters:
                n_clusters = st.number_input(
                    "簇数量", 
                    10, 10000, 
                    config.get('n_clusters', 100)
                )
            else:
                subset_ratio = st.slider(
                    "子集比例 (%)", 
                    0.1, 50.0, 
                    config.get('subset_ratio', 1.0), 
                    0.1
                )
                estimated_clusters = max(10, int(valid_count * subset_ratio / 100))
                st.info(f"估计簇数量: {estimated_clusters}")
                n_clusters = estimated_clusters
                
            kmeans_iterations = st.slider(
                "最大迭代次数", 
                10, 1000, 
                config.get('kmeans_iterations', 100)
            )
        elif clustering_method == "maxmin":
            init_method = st.selectbox(
                "初始点选择", 
                ["random", "first"],
                index=["random", "first"].index(config.get('init_method', 'random'))
            )
            subset_ratio = st.slider(
                "子集比例 (%)", 
                0.1, 50.0, 
                config.get('subset_ratio', 1.0), 
                0.1
            )
        
        # 选择子集方法
        selection_method = st.selectbox(
            "选择子集方法",
            ["centroid", "maxmin", "random"],
            index=0
        )
        
        # 执行聚类按钮
        if st.button("执行聚类"):
            with st.spinner("执行聚类中..."):
                # 准备参数
                cluster_params = {
                    'method': clustering_method,
                    'cutoff': cutoff if clustering_method == "butina" else 0.6,
                    'n_clusters': n_clusters if clustering_method == "kmeans" else None,
                    'max_iterations': kmeans_iterations if clustering_method == "kmeans" else 100,
                    'init_method': init_method if clustering_method == "maxmin" else "random",
                    'subset_ratio': subset_ratio if clustering_method in ["maxmin", "kmeans"] and not (clustering_method == "kmeans" and use_fixed_clusters) else 1.0,
                }
                
                # 保存聚类参数到配置文件
                config.update(cluster_params)
                save_json(config, CONFIG_FILE)
                
                # 准备数据
                valid_fps = [processed_results['fps_binary'][i] for i in valid_indices]
                valid_features = [processed_results['features'][i] for i in valid_indices]
                
                # 执行聚类
                start_time = time.time()
                cluster_labels, selected_indices = perform_clustering(
                    valid_fps, 
                    cluster_params, 
                    selection_method=selection_method
                )
                duration = time.time() - start_time
                
                # 评估聚类结果
                evaluation_results = evaluate_clustering(
                    valid_features,
                    cluster_labels,
                    clustering_method
                )
                
                # 准备结果
                clustering_results = {
                    'cluster_labels': cluster_labels,
                    'valid_indices': valid_indices,
                    'selected_indices': selected_indices,
                    'params': cluster_params,
                    'selection_method': selection_method,
                    'evaluation': evaluation_results,
                    'duration': duration
                }
                
                # 保存结果
                save_pickle(clustering_results, CLUSTERING_RESULTS_FILE)
                
                # 创建选择的子集
                selected_smiles_idx = [valid_indices[i] for i in selected_indices]
                smiles_col = config.get('smiles_col', 'SMILES')
                subset_df = df.iloc[selected_smiles_idx].copy()
                
                # 添加簇标签
                cluster_map = {idx: label for idx, label in zip(range(len(valid_indices)), cluster_labels)}
                subset_df['Cluster'] = [cluster_map[valid_indices.index(idx)] for idx in selected_smiles_idx]
                
                # 保存子集
                save_dataframe(subset_df, SELECTED_SUBSET_FILE)
                
                st.success(f"聚类完成！用时: {duration:.2f}秒")
    
    with col_right:
        st.header("聚类结果")
        
        # 如果有聚类结果，显示评估指标和可视化
        if file_exists(CLUSTERING_RESULTS_FILE):
            clustering_results = load_pickle(CLUSTERING_RESULTS_FILE)
            
            # 显示评估指标
            st.subheader("评估指标")
            metrics = clustering_results.get('evaluation', {})
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("轮廓系数", f"{metrics.get('silhouette_score', 0.0):.3f}")
                st.metric("Davies-Bouldin指数", f"{metrics.get('davies_bouldin_score', 0.0):.3f}")
            
            with col2:
                st.metric("Calinski-Harabasz指数", f"{metrics.get('calinski_harabasz_score', 0.0):.3f}")
                st.metric("簇数量", len(set(clustering_results['cluster_labels'])))
            
            # 显示聚类可视化
            st.subheader("聚类可视化")
            tab1, tab2 = st.tabs(["PCA投影", "t-SNE投影"])
            
            with tab1:
                valid_features = [processed_results['features'][i] for i in valid_indices]
                fig_pca = plot_fps_pca(
                    valid_features,
                    clustering_results['cluster_labels'],
                    clustering_results['selected_indices']
                )
                st.pyplot(fig_pca)
            
            with tab2:
                fig_tsne = plot_fps_tsne(
                    valid_features,
                    clustering_results['cluster_labels'],
                    clustering_results['selected_indices']
                )
                st.pyplot(fig_tsne)
            
            # 显示选择的子集
            if file_exists(SELECTED_SUBSET_FILE):
                subset_df = load_dataframe(SELECTED_SUBSET_FILE)
                st.subheader("选择的子集")
                st.write(f"选择的分子数量: {len(subset_df)}")
                
                # 显示子集预览
                st.dataframe(subset_df.head())
                
                # 下载按钮
                csv = subset_df.to_csv(index=False)
                st.download_button(
                    "下载选择的子集",
                    csv,
                    "selected_subset.csv",
                    "text/csv",
                    key='download-csv'
                ) 