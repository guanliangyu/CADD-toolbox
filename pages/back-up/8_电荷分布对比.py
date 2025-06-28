"""
分子库代表性子集选择系统 - 电荷分布对比页面
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, rdPartialCharges
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import umap
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
import time
from tqdm.auto import tqdm

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

# 设置页面
st.set_page_config(
    page_title="分子库代表性子集选择系统 - 电荷分布对比",
    page_icon="⚡",
    layout="wide"
)

# 检查CUDA是否可用
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    DEVICE = torch.device("cuda")
    st.sidebar.success("✅ CUDA可用，将使用GPU加速")
else:
    DEVICE = torch.device("cpu")
    st.sidebar.info("ℹ️ CUDA不可用，将使用CPU计算")

# 显示CPU核心数
CPU_COUNT = cpu_count()
st.sidebar.info(f"可用CPU核心数：{CPU_COUNT}")

# 设置并行计算的进程数
N_JOBS = min(CPU_COUNT - 1, 8)  # 保留一个核心给系统
st.sidebar.info(f"将使用 {N_JOBS} 个CPU核心进行并行计算")

st.title("电荷分布对比")

def load_smiles(file, smiles_col='SMILES'):
    """从CSV读取SMILES并转换为RDKit Mol对象"""
    try:
        df = pd.read_csv(file)
        if smiles_col not in df.columns:
            st.error(f"未找到SMILES列: {smiles_col}")
            return None, None
        
        mols = []
        valid_indices = []
        for i, row in df.iterrows():
            smi = str(row[smiles_col]).strip()
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mols.append(mol)
                valid_indices.append(i)
        
        return mols, df.iloc[valid_indices]
    except Exception as e:
        st.error(f"读取CSV文件时出错: {str(e)}")
        return None, None

def compute_gasteiger_charges(mol):
    """计算Gasteiger部分电荷"""
    try:
        Chem.AllChem.ComputeGasteigerCharges(mol)
        charges = []
        for atom in mol.GetAtoms():
            if atom.HasProp("_GasteigerCharge"):
                val = atom.GetProp("_GasteigerCharge")
                if val == 'nan':
                    charges.append(0.0)
                else:
                    charges.append(float(val))
            else:
                charges.append(0.0)
        return charges
    except:
        return None

def calc_dipole_moment(mol, charges):
    """计算偶极矩（Debye）"""
    DIP_CONST = 4.80298
    
    conf = mol.GetConformer()
    if conf is None:
        return 0.0
    
    dip_x, dip_y, dip_z = 0.0, 0.0, 0.0
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        q = charges[i]
        dip_x += q * pos.x
        dip_y += q * pos.y
        dip_z += q * pos.z
    
    dip = np.sqrt(dip_x**2 + dip_y**2 + dip_z**2)
    dip_debye = dip * DIP_CONST
    return dip_debye

def calc_charge_features(mol):
    """计算分子的电荷相关特征"""
    if mol.GetNumConformers() == 0:
        mol3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol3d, AllChem.ETKDGv3())
        try:
            AllChem.MMFFOptimizeMolecule(mol3d)
        except:
            pass
        mol = mol3d
    
    charges = compute_gasteiger_charges(mol)
    if not charges:
        return None
    
    max_charge = max(charges)
    min_charge = min(charges)
    total_charge = sum(charges)
    dipole = calc_dipole_moment(mol, charges)
    
    return {
        "max_charge": max_charge,
        "min_charge": min_charge,
        "total_charge": total_charge,
        "dipole": dipole
    }

def calc_peoe_vsa_descriptors(mol):
    """计算PEOE_VSA描述符"""
    try:
        vsa_vals = rdMolDescriptors.CalcPEOE_VSA(mol)
        return list(vsa_vals)
    except:
        return None

def assemble_electrostatic_df(mols, max_samples=1000):
    """组装电荷特征数据框"""
    if len(mols) > max_samples:
        mols = random.sample(mols, max_samples)
    
    records = []
    for mol in mols:
        feats_charge = calc_charge_features(mol)
        if feats_charge is None:
            continue
        peoe_vals = calc_peoe_vsa_descriptors(mol)
        
        row = {
            "max_charge": feats_charge["max_charge"],
            "min_charge": feats_charge["min_charge"],
            "total_charge": feats_charge["total_charge"],
            "dipole": feats_charge["dipole"],
        }
        if peoe_vals:
            for i, val in enumerate(peoe_vals, start=1):
                row[f"PEOE_VSA{i}"] = val
        records.append(row)
    
    df = pd.DataFrame(records)
    return df

def plot_dimensionality_reduction(dfA, dfB, cols, title, method="PCA"):
    """降维可视化"""
    dfA["_label"] = "A"
    dfB["_label"] = "B"
    df_all = pd.concat([dfA, dfB], ignore_index=True)
    df_all = df_all.dropna(subset=cols)
    
    X = df_all[cols].values
    labels = df_all["_label"].values
    
    if method == "PCA":
        reducer = PCA(n_components=2)
    else:  # UMAP
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    
    coords = reducer.fit_transform(X)
    xvals = coords[:,0]
    yvals = coords[:,1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cA = (labels == "A")
    cB = (labels == "B")
    ax.scatter(xvals[cA], yvals[cA], c='blue', alpha=0.5, label="数据集A", s=10)
    ax.scatter(xvals[cB], yvals[cB], c='red', alpha=0.5, label="数据集B", s=10)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

# 主界面
col1, col2 = st.columns(2)

with col1:
    st.subheader("数据集A")
    fileA = st.file_uploader("上传第一个CSV文件", type="csv")
    
with col2:
    st.subheader("数据集B")
    fileB = st.file_uploader("上传第二个CSV文件", type="csv")

# 参数设置
st.subheader("参数设置")
col1, col2 = st.columns(2)

with col1:
    smiles_col = st.text_input("SMILES列名", value="SMILES")
    max_samples = st.number_input("每个数据集的最大样本数", 100, 5000, 1000)

with col2:
    n_peoe_vsa = st.slider("使用PEOE_VSA描述符数量", 1, 14, 4)
    viz_method = st.selectbox("降维方法", ["PCA", "UMAP"] if HAS_UMAP else ["PCA"])

if st.button("开始分析") and fileA is not None and fileB is not None:
    with st.spinner("正在进行电荷分布分析..."):
        try:
            # 加载分子
            molsA, dfA = load_smiles(fileA, smiles_col)
            molsB, dfB = load_smiles(fileB, smiles_col)
            
            if molsA and molsB:
                st.success(f"成功加载: 数据集A {len(molsA)}个分子, 数据集B {len(molsB)}个分子")
                
                # 计算电荷特征
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 处理数据集A
                status_text.text("正在处理数据集A...")
                dfA_elec = assemble_electrostatic_df(molsA, max_samples)
                progress_bar.progress(0.5)
                
                # 处理数据集B
                status_text.text("正在处理数据集B...")
                dfB_elec = assemble_electrostatic_df(molsB, max_samples)
                progress_bar.progress(1.0)
                
                status_text.text("分析完成！")
                
                # 显示结果
                st.subheader("基本统计")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("数据集A统计:")
                    st.write(dfA_elec[["max_charge", "min_charge", "total_charge", "dipole"]].describe())
                
                with col2:
                    st.write("数据集B统计:")
                    st.write(dfB_elec[["max_charge", "min_charge", "total_charge", "dipole"]].describe())
                
                # 电荷分布可视化
                st.subheader("电荷分布可视化")
                
                # 准备特征列
                base_cols = ["dipole", "max_charge", "min_charge", "total_charge"]
                vsa_cols = [f"PEOE_VSA{i}" for i in range(1, n_peoe_vsa + 1)]
                all_cols = base_cols + vsa_cols
                
                # 降维可视化
                fig = plot_dimensionality_reduction(
                    dfA_elec.copy(), 
                    dfB_elec.copy(), 
                    all_cols,
                    f"{viz_method} (电荷 + PEOE_VSA特征)",
                    method=viz_method
                )
                st.pyplot(fig)
                
                # 偶极矩分布
                st.subheader("偶极矩分布")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.kdeplot(data=dfA_elec["dipole"], label="数据集A", ax=ax)
                sns.kdeplot(data=dfB_elec["dipole"], label="数据集B", ax=ax)
                ax.set_xlabel("偶极矩 (Debye)")
                ax.set_ylabel("密度")
                ax.legend()
                st.pyplot(fig)
                
                # 电荷范围分布
                st.subheader("电荷范围分布")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                sns.boxplot(data=[dfA_elec["max_charge"], dfB_elec["max_charge"]], 
                          labels=["数据集A", "数据集B"], ax=ax1)
                ax1.set_title("最大正电荷分布")
                
                sns.boxplot(data=[dfA_elec["min_charge"], dfB_elec["min_charge"]], 
                          labels=["数据集A", "数据集B"], ax=ax2)
                ax2.set_title("最小负电荷分布")
                
                plt.tight_layout()
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"分析过程中出错: {str(e)}") 