"""
应用程序状态管理工具 - 用于跨页面共享数据和状态
"""
import streamlit as st

def initialize_session_state():
    """初始化会话状态，确保关键变量存在"""
    defaults = {
        # 旧版键（兼容）
        'df': None,
        'processed_results': None,
        'config': None,
        'subset_indices': None,
        'validation_results': None,
        'smiles_col': "SMILES",
        'subset_ratio': 1.0,
        'clustering_method': "kmeans",
        # 现有页面常用键
        'preview_df': None,
        'preview_columns': None,
        'full_df': None,
        'full_columns': None,
        'validation_summary': None,
        'druglike_df': None,
        'filtered_df': None,
        'original_count': None,
        'valid_count': None,
        'invalid_count': None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def clear_state(keys=None):
    """清除指定的会话状态变量
    
    参数:
        keys: 要清除的键列表，如果为None则清除所有状态
    """
    if keys is None:
        # 清除所有相关状态（含新旧键）
        keys = [
            'df', 'processed_results', 'config', 'subset_indices', 'validation_results',
            'preview_df', 'preview_columns', 'full_df', 'full_columns',
            'validation_summary', 'validation_cache',
            'druglike_df', 'filtered_df', 'original_count', 'valid_count', 'invalid_count',
            'fps_cache', 'metric_cache'
        ]
    
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

def get_state_summary():
    """获取当前状态摘要，用于显示在侧边栏"""
    summary = {}

    full_df = st.session_state.get('full_df')
    preview_df = st.session_state.get('preview_df')
    legacy_df = st.session_state.get('df')

    if full_df is not None:
        summary['数据加载'] = f"全量 {len(full_df):,} 条"
    elif preview_df is not None:
        summary['数据加载'] = f"预览 {len(preview_df):,} 条"
    elif legacy_df is not None:
        summary['数据加载'] = f"{len(legacy_df):,} 条"

    validation_summary = st.session_state.get('validation_summary')
    if isinstance(validation_summary, dict) and validation_summary.get('evaluated_count'):
        valid_count = int(validation_summary.get('valid_count', 0))
        evaluated_count = int(validation_summary.get('evaluated_count', 0))
        ratio = (valid_count / evaluated_count * 100.0) if evaluated_count else 0.0
        summary['SMILES校验'] = f"{valid_count}/{evaluated_count} ({ratio:.1f}%)"

    filtered_df = st.session_state.get('filtered_df')
    druglike_df = st.session_state.get('druglike_df')
    if filtered_df is not None:
        summary['成药性筛选'] = f"筛后 {len(filtered_df):,} 条"
    elif druglike_df is not None:
        summary['基础筛选'] = f"{len(druglike_df):,} 条"

    subset_indices = st.session_state.get('subset_indices')
    if subset_indices is not None:
        summary['代表性分子'] = f"{len(subset_indices):,} 个"
        base_count = None
        if full_df is not None:
            base_count = len(full_df)
        elif preview_df is not None:
            base_count = len(preview_df)
        elif legacy_df is not None:
            base_count = len(legacy_df)
        if base_count:
            summary['子集比例'] = f"{len(subset_indices)/base_count:.2%}"

    if st.session_state.get('metric_cache') or st.session_state.get('fps_cache'):
        summary['多样性评估'] = "已运行"

    return summary

def display_state_sidebar():
    """在侧边栏显示当前状态摘要"""
    summary = get_state_summary()
    
    if summary:
        st.sidebar.subheader("当前状态")
        for key, value in summary.items():
            st.sidebar.text(f"{key}: {value}")
    
    # 添加清除状态按钮
    if summary and st.sidebar.button("清除所有数据"):
        clear_state()
        st.rerun()

def save_results_to_session(results_dict, keys=None):
    """将结果保存到会话状态
    
    参数:
        results_dict: 包含结果的字典
        keys: 要保存的键列表，如果为None则保存所有键
    """
    if keys is None:
        for key, value in results_dict.items():
            st.session_state[key] = value
    else:
        for key in keys:
            if key in results_dict:
                st.session_state[key] = results_dict[key] 
