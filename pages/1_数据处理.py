"""
分子库代表性子集选择系统 - 数据处理页面
"""
import os
import sys
import time
import pandas as pd
import numpy as np
import streamlit as st
from rdkit import Chem
from multiprocessing import Pool, cpu_count
from functools import partial
from typing import List, Tuple, Dict, Any

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

# 导入工具模块
from utils.molecular_utils import MoleculeProcessor
from utils.file_utils import (
    load_dataframe, load_json, save_dataframe, save_json, save_pickle,
    MOLECULES_FILE, CONFIG_FILE, PROCESSED_RESULTS_FILE
)

# 设置页面
st.set_page_config(
    page_title="分子库代表性子集选择系统 - 数据处理",
    page_icon="🧪",
    layout="wide"
)

st.title("数据处理")

def process_single_mol(smiles: str, processor: MoleculeProcessor) -> Tuple[Any, Any, Any]:
    """处理单个分子
    
    参数:
        smiles: SMILES字符串
        processor: 分子处理器实例
        
    返回:
        (mol, fp, features)元组
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None
            
        # 计算指纹
        fp = processor.compute_fingerprint(mol)
        if fp is None:
            return None, None, None
            
        # 计算特征
        features = processor.compute_features(mol)
        if features is None:
            return None, None, None
            
        return mol, fp, features
    except Exception as e:
        print(f"处理分子时出错: {str(e)}")
        return None, None, None

def process_batch(batch_data: List[str], config: Dict) -> List[Tuple[Any, Any, Any]]:
    """处理一批分子
    
    参数:
        batch_data: SMILES字符串列表
        config: 配置字典
        
    返回:
        处理结果列表
    """
    processor = MoleculeProcessor(config)
    results = []
    for smiles in batch_data:
        result = process_single_mol(smiles, processor)
        results.append(result)
    return results

def display_status_sidebar():
    """在侧边栏显示当前处理状态"""
    st.sidebar.subheader("当前状态")
    
    # 检查是否有已上传的数据
    df = load_dataframe(MOLECULES_FILE)
    if df is not None:
        st.sidebar.text(f"已上传: {len(df)}条记录")
    
    # 检查是否有处理结果
    config = load_json(CONFIG_FILE)
    if config is not None:
        st.sidebar.text("配置已保存")

# 显示侧边栏状态
display_status_sidebar()

# 创建两列布局
col1, col2 = st.columns([1, 1])

with col1:
    st.header("配置参数")
    
    # 文件上传
    uploaded_file = st.file_uploader("上传CSV文件", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"成功读取CSV文件: {len(df)}条记录")
        
        # SMILES列选择
        smiles_col = st.selectbox(
            "选择SMILES列",
            df.columns.tolist(),
            index=df.columns.tolist().index("SMILES") if "SMILES" in df.columns else 0
        )
        
        # 基本参数
        st.subheader("基本参数")
        batch_size = st.number_input("批处理大小", 100, 10000, 1000, 100)
        n_jobs = st.number_input("并行进程数", 1, cpu_count(), min(4, cpu_count()))
        
        # 指纹计算参数
        st.subheader("指纹计算")
        fp_type = st.selectbox("指纹类型", ["ECFP", "FCFP"])
        radius = st.slider("半径", 1, 4, 2)
        nBits = st.slider("比特数", 512, 2048, 1024, 512)
        
        # 特征计算参数
        st.subheader("特征计算")
        use_3d = st.checkbox("计算3D特征", value=False)
        
        # 保存配置
        config = {
            "smiles_col": smiles_col,
            "batch_size": batch_size,
            "n_jobs": n_jobs,
            "fp_type": fp_type,
            "radius": radius,
            "nBits": nBits,
            "use_3d": use_3d
        }
        
        # 保存数据和配置
        save_dataframe(df, MOLECULES_FILE)
        save_json(config, CONFIG_FILE)
        
        if st.button("开始处理分子"):
            with st.spinner("正在处理分子..."):
                try:
                    # 准备数据
                    smiles_list = df[smiles_col].tolist()
                    total_mols = len(smiles_list)
                    
                    # 创建进度条
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # 分批处理
                    batches = [
                        smiles_list[i:i + batch_size]
                        for i in range(0, total_mols, batch_size)
                    ]
                    
                    # 创建进程池
                    with Pool(processes=n_jobs) as pool:
                        # 使用partial固定config参数
                        process_func = partial(process_batch, config=config)
                        
                        # 收集结果
                        mols = []
                        fps = []
                        features = []
                        
                        # 处理每个批次
                        for i, batch_results in enumerate(pool.imap(process_func, batches)):
                            # 更新进度
                            progress = (i + 1) / len(batches)
                            progress_bar.progress(progress)
                            status_text.text(f"已处理: {(i + 1) * batch_size}/{total_mols}")
                            
                            # 收集批次结果
                            for mol, fp, feat in batch_results:
                                mols.append(mol)
                                fps.append(fp)
                                features.append(feat)
                    
                    # 保存处理结果
                    results = {
                        "mols": mols,
                        "fps": fps,
                        "features": features
                    }
                    save_pickle(results, PROCESSED_RESULTS_FILE)
                    
                    st.success("分子处理完成!")
                    
                except Exception as e:
                    st.error(f"处理分子时出错: {str(e)}")

with col2:
    st.header("预览")
    
    # 显示数据预览
    if uploaded_file is not None:
        st.subheader("数据预览")
        st.dataframe(df.head())
        
        # 显示SMILES预览
        if "SMILES" in df.columns:
            st.subheader("分子结构预览")
            for i, row in df.head().iterrows():
                smi = row["SMILES"]
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    st.text(f"分子 {i+1}")
                    st.text(smi) 