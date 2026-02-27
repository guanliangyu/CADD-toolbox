"""
分子库代表性子集选择系统 - 数据预处理页面
"""

import os
import sys
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Draw
import tempfile
from rdkit import RDLogger

# 设置RDKit日志级别
RDLogger.DisableLog("rdApp.*")

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

# 设置页面
st.set_page_config(
    page_title="分子库代表性子集选择系统 - 数据预处理", page_icon="🔄", layout="wide"
)

st.title("数据预处理")

# 初始化session state
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "available_fields" not in st.session_state:
    st.session_state.available_fields = []
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False
if "current_file_name" not in st.session_state:
    st.session_state.current_file_name = None


def process_sdf(file):
    """处理SDF文件并返回DataFrame和可用字段"""
    try:
        # 创建临时文件保存上传的内容
        with tempfile.NamedTemporaryFile(delete=False, suffix=".sdf") as tmp_file:
            tmp_file.write(file.getvalue())
            tmp_path = tmp_file.name

        # 创建进度条和状态文本
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 首先计算总分子数
        suppl = Chem.SDMolSupplier(tmp_path)
        total_mols = len(suppl)
        status_text.text(f"发现 {total_mols} 个分子，开始处理...")

        # 读取SDF文件
        df = PandasTools.LoadSDF(tmp_path, smilesName="SMILES", molColName=None)

        # 删除临时文件
        os.unlink(tmp_path)

        # 检查是否存在SMILES列
        if "SMILES" not in df.columns:
            # 显示所有可用列
            st.warning("未找到SMILES列，请从以下列中选择：")
            smiles_col = st.selectbox("选择包含SMILES的列", options=df.columns.tolist())
            # 重命名选择的列为SMILES
            df = df.rename(columns={smiles_col: "SMILES"})

        # 获取所有字段
        available_fields = [col for col in df.columns if col != "SMILES"]

        # 标准化SMILES
        valid_indices = []
        valid_smiles = []
        invalid_count = 0

        status_text.text("正在标准化SMILES结构...")
        for idx, row in df.iterrows():
            try:
                smi = str(row["SMILES"]).strip()
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    # 生成标准SMILES
                    canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                    valid_indices.append(idx)
                    valid_smiles.append(canonical_smi)
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1

            # 更新进度
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            if (idx + 1) % 100 == 0:
                status_text.text(f"已处理 {idx + 1}/{len(df)} 个分子...")

        # 创建新的DataFrame
        valid_df = df.iloc[valid_indices].copy()
        valid_df["SMILES"] = valid_smiles

        # 显示处理结果
        if invalid_count > 0:
            st.warning(f"发现 {invalid_count} 个无效的分子结构")

        # 清除进度显示
        progress_bar.empty()
        status_text.empty()

        return valid_df, available_fields
    except Exception as e:
        st.error(f"处理SDF文件时出错: {str(e)}")
        return None, []


def process_csv(file):
    """处理CSV文件并返回DataFrame和可用字段"""
    try:
        df = pd.read_csv(file)

        # 检查是否存在SMILES列
        smiles_candidates = [
            "SMILES",
            "Smiles",
            "smiles",
            "SMILE",
            "Smile",
            "smile",
            "Canonical_SMILES",
            "CanonicalSmiles",
            "canonicalsmiles",
            "SMILES_str",
            "smiles_str",
            "Structure",
            "structure",
        ]
        smiles_col = None
        for col in smiles_candidates:
            if col in df.columns:
                smiles_col = col
                break

        if smiles_col is None:
            # 显示所有可用列
            st.warning("未自动找到SMILES列，请从以下列中选择：")
            st.write("可用列：")
            for col in df.columns:
                # 显示列名和前几个值的示例
                st.text(f"{col}：\n{df[col].head(3).values}")
            smiles_col = st.selectbox("选择包含SMILES的列", options=df.columns.tolist())

        # 创建进度条和状态文本
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text(f"开始处理 {len(df)} 条记录...")

        # 验证SMILES并标准化
        valid_indices = []
        valid_smiles = []
        invalid_count = 0
        invalid_smiles = []

        for idx, row in df.iterrows():
            try:
                # 处理SMILES字符串，移除可能的转义字符
                smi = str(row[smiles_col]).strip()
                smi = smi.replace("\\\\", "\\")  # 处理双反斜杠
                smi = smi.replace("\\/", "/")  # 处理顺式构型
                smi = smi.replace("\\\\", "\\")  # 处理反式构型

                mol = Chem.MolFromSmiles(smi)
                if mol:
                    # 生成标准SMILES
                    canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=True)
                    valid_indices.append(idx)
                    valid_smiles.append(canonical_smi)
                else:
                    invalid_count += 1
                    if len(invalid_smiles) < 5:  # 只记录前5个无效的SMILES
                        invalid_smiles.append(smi)
            except Exception:
                invalid_count += 1
                if len(invalid_smiles) < 5:
                    invalid_smiles.append(smi)

            # 更新进度
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            if (idx + 1) % 100 == 0:
                status_text.text(f"已处理 {idx + 1}/{len(df)} 条记录...")

        if invalid_count > 0:
            st.warning(f"发现 {invalid_count} 个无效的SMILES结构")
            if invalid_smiles:
                st.write("前几个无效的SMILES示例：")
                for i, smi in enumerate(invalid_smiles, 1):
                    st.text(f"{i}. {smi}")

        # 创建新的DataFrame
        valid_df = df.iloc[valid_indices].copy()
        valid_df["SMILES"] = valid_smiles

        # 获取所有可用字段
        available_fields = [col for col in valid_df.columns if col != "SMILES"]

        # 清除进度显示
        progress_bar.empty()
        status_text.empty()

        return valid_df, available_fields
    except Exception as e:
        st.error(f"处理CSV文件时出错: {str(e)}")
        return None, []


def standardize_output(df, name_field=None):
    """标准化输出数据"""
    try:
        # 确保SMILES列存在
        if "SMILES" not in df.columns:
            st.error("数据中缺少SMILES列")
            return None

        # 创建输出DataFrame
        output_df = pd.DataFrame()
        output_df["SMILES"] = df["SMILES"]

        # 添加名称列
        if name_field and name_field in df.columns:
            output_df["Name"] = df[name_field]
        else:
            # 如果没有指定名称字段，使用索引作为名称
            output_df["Name"] = [f"Compound_{i+1}" for i in range(len(df))]

        return output_df
    except Exception as e:
        st.error(f"标准化数据时出错: {str(e)}")
        return None


def display_molecule_grid(df, n_cols=5, n_rows=2):
    """显示分子结构网格"""
    if len(df) == 0:
        return

    n_mols = min(len(df), n_cols * n_rows)
    mols = []
    legends = []
    for i, smi in enumerate(df["SMILES"].head(n_mols)):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mols.append(mol)
                legends.append(f"#{i+1}")
        except Exception:
            continue

    if mols:
        try:
            # 设置绘图参数
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=n_cols,
                subImgSize=(300, 300),
                legends=legends,
                returnPNG=False,
            )
            st.image(img)
        except Exception as e:
            st.error(f"绘制分子结构时出错: {str(e)}")


# 主界面
st.write("请上传SDF或CSV文件进行预处理。文件应包含分子结构信息（SMILES或分子对象）。")

# 文件上传
uploaded_file = st.file_uploader("上传文件", type=["sdf", "csv"])

if uploaded_file is not None:
    # 检查是否需要重新处理文件
    if (st.session_state.current_file_name != uploaded_file.name) or (
        not st.session_state.file_processed
    ):
        # 显示处理进度
        with st.spinner("正在处理文件..."):
            # 根据文件类型处理
            file_type = uploaded_file.name.split(".")[-1].lower()

            if file_type == "sdf":
                df, available_fields = process_sdf(uploaded_file)
            else:  # csv
                df, available_fields = process_csv(uploaded_file)

            # 更新session state
            st.session_state.processed_df = df
            st.session_state.available_fields = available_fields
            st.session_state.file_processed = True
            st.session_state.current_file_name = uploaded_file.name

    # 使用已处理的数据
    df = st.session_state.processed_df
    available_fields = st.session_state.available_fields

    if df is not None and len(available_fields) > 0:
        st.success(f"成功读取 {len(df)} 条记录")

        # 显示可用字段
        st.subheader("可用字段")
        st.write("以下是文件中可用的字段：")
        st.write(available_fields)

        # 选择名称字段
        name_field = st.selectbox(
            "选择作为化合物名称的字段（可选）", options=["无"] + available_fields
        )

        if name_field == "无":
            name_field = None

        # 预览原始数据
        st.subheader("原始数据预览")
        preview_df = df.copy()
        if "Molecule" in preview_df.columns:
            preview_df = preview_df.drop("Molecule", axis=1)
        st.write(preview_df.head())

        # 显示部分分子结构
        st.subheader("分子结构预览")
        display_molecule_grid(df)

        # 处理并预览标准化数据
        if st.button("生成标准CSV"):
            output_df = standardize_output(df, name_field)

            if output_df is not None:
                st.subheader("标准化数据预览")
                st.write(output_df.head())

                # 提供下载
                csv = output_df.to_csv(index=False)
                st.download_button(
                    label="下载标准CSV文件",
                    data=csv,
                    file_name="standardized_compounds.csv",
                    mime="text/csv",
                )

                # 显示统计信息
                st.subheader("数据统计")
                st.write(f"- 总记录数：{len(output_df)}")
                st.write("- 包含字段：SMILES, Name")

                # 验证SMILES
                valid_count = sum(
                    1
                    for smi in output_df["SMILES"]
                    if Chem.MolFromSmiles(smi) is not None
                )
                st.write(f"- 有效SMILES数：{valid_count}")
    else:
        st.error("文件处理失败或没有可用字段")
