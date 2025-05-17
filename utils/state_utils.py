"""
应用程序状态管理工具 - 用于跨页面共享数据和状态
"""
import streamlit as st
import pandas as pd
from rdkit import Chem

def initialize_session_state():
    """初始化会话状态，确保关键变量存在"""
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    
    if 'processed_results' not in st.session_state:
        st.session_state['processed_results'] = None
    
    if 'config' not in st.session_state:
        st.session_state['config'] = None
    
    if 'subset_indices' not in st.session_state:
        st.session_state['subset_indices'] = None
    
    if 'validation_results' not in st.session_state:
        st.session_state['validation_results'] = None
    
    if 'smiles_col' not in st.session_state:
        st.session_state['smiles_col'] = "SMILES"
    
    if 'subset_ratio' not in st.session_state:
        st.session_state['subset_ratio'] = 1.0
    
    if 'clustering_method' not in st.session_state:
        st.session_state['clustering_method'] = "kmeans"

def clear_state(keys=None):
    """清除指定的会话状态变量
    
    参数:
        keys: 要清除的键列表，如果为None则清除所有状态
    """
    if keys is None:
        # 清除所有相关状态
        keys = ['df', 'processed_results', 'config', 'subset_indices', 'validation_results']
    
    for key in keys:
        if key in st.session_state:
            del st.session_state[key]

def get_state_summary():
    """获取当前状态摘要，用于显示在侧边栏"""
    summary = {}
    
    # 数据状态
    if 'df' in st.session_state and st.session_state['df'] is not None:
        df = st.session_state['df']
        summary['数据集'] = f"{len(df)}条记录"
        
        if 'processed_results' in st.session_state and st.session_state['processed_results'] is not None:
            mols = st.session_state['processed_results']['mols']
            valid_count = sum(1 for m in mols if m is not None)
            summary['有效分子'] = f"{valid_count}/{len(df)}个"
    
    # 聚类状态
    if 'subset_indices' in st.session_state and st.session_state['subset_indices'] is not None:
        subset_indices = st.session_state['subset_indices']
        summary['代表性分子'] = f"{len(subset_indices)}个"
        
        if 'df' in st.session_state and st.session_state['df'] is not None:
            df = st.session_state['df']
            summary['子集比例'] = f"{len(subset_indices)/len(df):.2%}"
    
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
        st.experimental_rerun()

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