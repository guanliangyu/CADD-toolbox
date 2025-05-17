"""
分子库代表性子集选择系统 - 验证与下载页面
"""
import os
import sys
import io
import tempfile
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools

# 添加项目根目录到路径，确保能导入utils模块
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

# 导入工具模块
from utils.validation_utils import (
    plot_property_distributions, plot_nearest_neighbor_histogram,
    plot_pca_visualization, calculate_coverage_metrics,
    calculate_nearest_neighbor_distance
)
from utils.state_utils import initialize_session_state, display_state_sidebar
from utils.config_utils import render_sidebar_parameters
from utils.file_utils import (
    load_dataframe, load_pickle, load_json,
    file_exists, MOLECULES_FILE, CONFIG_FILE, PROCESSED_RESULTS_FILE,
    CLUSTERING_RESULTS_FILE, SELECTED_SUBSET_FILE
)

# 设置页面标题和配置
st.set_page_config(
    page_title="分子库代表性子集选择系统 - 验证与下载",
    page_icon="🧪",
    layout="wide"
)

# 页面标题
st.title("验证与下载")

# 主内容区域
if not file_exists(SELECTED_SUBSET_FILE):
    st.info("请先在聚类与选择页面选择代表性分子")
    if st.button("前往聚类与选择页面"):
        st.switch_page("pages/2_聚类与选择.py")
else:
    # 加载数据
    df = load_dataframe(MOLECULES_FILE)
    processed_results = load_pickle(PROCESSED_RESULTS_FILE)
    subset_indices = load_pickle(SELECTED_SUBSET_FILE)
    
    # 创建两列布局
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        # 显示子集大小和比例
        total_mols = len(df)
        valid_mols = sum(1 for m in processed_results['mols'] if m is not None)
        subset_size = len(subset_indices)
        
        st.subheader("子集概览")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("子集大小", f"{subset_size}个分子")
        with col2:
            st.metric("占全部分子比例", f"{subset_size/total_mols:.2%}")
        with col3:
            st.metric("占有效分子比例", f"{subset_size/valid_mols:.2%}")
        
        # 验证选项卡
        tab1, tab2 = st.tabs(["子集验证", "下载数据"])
        
        with tab1:
            if st.button("验证子集质量", key="validate_btn"):
                with st.spinner("验证中..."):
                    # 执行验证
                    validation_results = validate_selection(
                        processed_results,
                        subset_indices
                    )
                    
                    # 显示验证结果
                    if validation_results:
                        _show_validation_results(validation_results)
        
        with tab2:
            st.subheader("下载选项")
            if subset_indices:
                # 提取子集数据
                subset_df = df.iloc[subset_indices].copy().reset_index(drop=True)
                
                # CSV下载
                csv_buffer = io.StringIO()
                subset_df.to_csv(csv_buffer, index=False)
                
                st.download_button(
                    label="下载CSV格式子集",
                    data=csv_buffer.getvalue(),
                    file_name="representative_subset.csv",
                    mime="text/csv"
                )
                
                # 添加分子属性
                if st.checkbox("添加理化性质到下载文件", value=True):
                    properties_df = subset_df.copy()
                    
                    # 添加理化性质列
                    for i, mol_idx in enumerate(subset_indices):
                        if processed_results['mols'][mol_idx] is not None:
                            props = processed_results['basic_desc'][mol_idx]
                            for prop_name, value in props.items():
                                if prop_name not in properties_df.columns:
                                    properties_df[prop_name] = None
                                properties_df.at[i, prop_name] = value
                    
                    # 导出带属性的CSV
                    props_csv_buffer = io.StringIO()
                    properties_df.to_csv(props_csv_buffer, index=False)
                    
                    st.download_button(
                        label="下载带理化性质的CSV子集",
                        data=props_csv_buffer.getvalue(),
                        file_name="representative_subset_with_properties.csv",
                        mime="text/csv"
                    )
    
    with right_col:
        st.subheader("数据状态")
        
        # 显示数据集信息
        st.write("📊 数据集信息")
        st.write(f"- 总分子数：{total_mols}")
        st.write(f"- 有效分子数：{valid_mols}")
        st.write(f"- 选择子集大小：{subset_size}")
        
        # 显示处理状态
        st.write("🔄 处理状态")
        st.write("- ✅ 数据集已加载")
        st.write("- ✅ 分子已处理")
        st.write("- ✅ 子集已选择")
        
        # 显示验证选项
        st.write("🔍 验证选项")
        st.write("- 覆盖度分析")
        st.write("- 属性分布比较")
        st.write("- 最近邻分析")
        
        # 显示下载选项
        st.write("💾 下载选项")
        st.write("- CSV格式")
        st.write("- 带理化性质")
        
        # 导航按钮
        st.write("⚡ 快速导航")
        if st.button("前往可视化分析"):
            st.switch_page("pages/4_可视化分析.py")
        if st.button("返回聚类与选择"):
            st.switch_page("pages/2_聚类与选择.py")

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

def _show_validation_results(validation_results):
    """显示验证结果"""
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