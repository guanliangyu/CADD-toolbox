"""
分子库代表性子集选择系统 - 化学空间对比页面
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit import DataStructs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import umap

# Configure matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Add project root directory to path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

# Set page configuration
st.set_page_config(
    page_title="Chemical Space Comparison",
    page_icon="🧪",
    layout="wide"
)

st.title("Chemical Space Comparison")

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

def compute_morgan_fingerprints(mols, radius=2, nBits=1024, use_features=False):
    """计算Morgan指纹"""
    fps = []
    fp_array_list = []
    
    # 创建Morgan指纹生成器
    morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits, useFeatures=use_features)
    
    for mol in mols:
        # 使用新的API生成指纹
        fp = morgan_gen.GetFingerprint(mol)
        fps.append(fp)
        # 转换为数组
        arr = np.zeros((nBits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_array_list.append(arr)
    
    fp_array = np.array(fp_array_list)
    return fps, fp_array

def generate_3d_conformer(mol):
    """生成3D构象"""
    try:
        mol_3d = Chem.AddHs(mol)
        ps = AllChem.ETKDGv3()
        ps.maxAttempts = 50
        cid = AllChem.EmbedMolecule(mol_3d, ps)
        if cid < 0:
            return None
        AllChem.MMFFOptimizeMolecule(mol_3d)
        mol_3d = Chem.RemoveHs(mol_3d)
        return mol_3d
    except:
        return None

def compute_pmi_ratios(mol):
    """计算PMI比率"""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    try:
        inertia = rdMolDescriptors.CalcPMIValues(mol)
        I1, I2, I3 = inertia
        if abs(I3) < 1e-9:
            return None
        return (I1 / I3, I2 / I3)
    except:
        return None

def plot_2d_coords(coordsA, coordsB, title, labelA="Dataset A", labelB="Dataset B"):
    """Plot 2D coordinate scatter plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coordsA[:,0], coordsA[:,1], c='blue', alpha=0.5, label=labelA, s=10)
    ax.scatter(coordsB[:,0], coordsB[:,1], c='red', alpha=0.5, label=labelB, s=10)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_pmi_triangle(pmi_list_A, pmi_list_B, labelA="Dataset A", labelB="Dataset B"):
    """Plot PMI triangle diagram"""
    def to_triangle_coords(a, b):
        x = b + (a/2.0)
        y = (np.sqrt(3)/2.0) * a
        return (x, y)
    
    xyA = [to_triangle_coords(a, b) for (a,b) in pmi_list_A if a is not None and b is not None]
    xyB = [to_triangle_coords(a, b) for (a,b) in pmi_list_B if a is not None and b is not None]
    
    if not xyA or not xyB:
        return None
        
    xyA = np.array(xyA)
    xyB = np.array(xyB)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xyA[:,0], xyA[:,1], c='blue', alpha=0.5, label=labelA, s=10)
    ax.scatter(xyB[:,0], xyB[:,1], c='red', alpha=0.5, label=labelB, s=10)
    ax.set_title("PMI Triangle Plot (Shape Space)")
    ax.legend()
    plt.tight_layout()
    return fig

# Main interface
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset A")
    fileA = st.file_uploader("Upload first CSV file", type="csv")
    
with col2:
    st.subheader("Dataset B")
    fileB = st.file_uploader("Upload second CSV file", type="csv")

# Parameter settings
st.subheader("Parameter Settings")
col1, col2, col3 = st.columns(3)

with col1:
    smiles_col = st.text_input("SMILES Column Name", value="SMILES")
    max_samples = st.number_input("Max Samples per Dataset", 100, 10000, 1000)

with col2:
    radius = st.slider("Morgan Fingerprint Radius", 1, 4, 2)
    nBits = st.slider("Number of Bits", 512, 2048, 1024, 512)

with col3:
    perplexity = st.slider("t-SNE Perplexity", 5, 50, 30)
    n_neighbors = st.slider("UMAP Neighbors", 5, 50, 15)

if st.button("Start Analysis") and fileA is not None and fileB is not None:
    with st.spinner("Performing chemical space analysis..."):
        try:
            # Load molecules
            molsA, dfA = load_smiles(fileA, smiles_col)
            molsB, dfB = load_smiles(fileB, smiles_col)
            
            if molsA and molsB:
                st.success(f"Successfully loaded: Dataset A ({len(molsA)} molecules), Dataset B ({len(molsB)} molecules)")
                
                # Random sampling
                if len(molsA) > max_samples:
                    molsA = random.sample(molsA, max_samples)
                if len(molsB) > max_samples:
                    molsB = random.sample(molsB, max_samples)
                
                # Calculate fingerprints
                fpsA, arrA = compute_morgan_fingerprints(molsA, radius, nBits)
                fpsB, arrB = compute_morgan_fingerprints(molsB, radius, nBits)
                
                # 1. PCA Analysis
                st.subheader("1. PCA Analysis")
                pca = PCA(n_components=2)
                coordsA_pca = pca.fit_transform(arrA)
                coordsB_pca = pca.fit_transform(arrB)
                fig_pca = plot_2d_coords(coordsA_pca, coordsB_pca, "PCA Projection of Morgan Fingerprints")
                st.pyplot(fig_pca)
                
                # 2. t-SNE Analysis
                st.subheader("2. t-SNE Analysis")
                tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
                coordsA_tsne = tsne.fit_transform(arrA)
                coordsB_tsne = tsne.fit_transform(arrB)
                fig_tsne = plot_2d_coords(coordsA_tsne, coordsB_tsne, "t-SNE Projection of Morgan Fingerprints")
                st.pyplot(fig_tsne)
                
                # 3. UMAP Analysis
                st.subheader("3. UMAP Analysis")
                reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
                coordsA_umap = reducer.fit_transform(arrA)
                coordsB_umap = reducer.fit_transform(arrB)
                fig_umap = plot_2d_coords(coordsA_umap, coordsB_umap, "UMAP Projection of Morgan Fingerprints")
                st.pyplot(fig_umap)
                
                # 4. PMI Analysis
                st.subheader("4. PMI Shape Space Analysis")
                with st.spinner("Generating 3D conformers and calculating PMI..."):
                    pmi_list_A = []
                    for m in molsA:
                        m3d = generate_3d_conformer(m)
                        if m3d:
                            ratios = compute_pmi_ratios(m3d)
                            if ratios:
                                pmi_list_A.append(ratios)
                    
                    pmi_list_B = []
                    for m in molsB:
                        m3d = generate_3d_conformer(m)
                        if m3d:
                            ratios = compute_pmi_ratios(m3d)
                            if ratios:
                                pmi_list_B.append(ratios)
                    
                    if pmi_list_A and pmi_list_B:
                        fig_pmi = plot_pmi_triangle(pmi_list_A, pmi_list_B)
                        if fig_pmi:
                            st.pyplot(fig_pmi)
                    else:
                        st.warning("Unable to generate 3D conformers or calculate PMI ratios for some molecules")
                
        except Exception as e:
            st.error(f"Error during analysis: {str(e)}") 