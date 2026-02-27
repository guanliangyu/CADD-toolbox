"""
CADD-Toolbox - 2D分子描述符生成页面
基于SMILES生成多种2D分子描述符和指纹
"""

import os
import time
import multiprocessing as mp

from typing import Any, Dict

import pandas as pd
import streamlit as st
import numpy as np

from utils.descriptor_generation import (
    AVALON_AVAILABLE,
    build_molecule_cache,
    generate_fingerprints,
    generate_molecular_descriptors,
    merge_feature_frames,
    process_chunk_worker,
)

try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except (ImportError, ModuleNotFoundError):
    get_script_run_ctx = None  # type: ignore[assignment]

_FALLBACK_STATE: Dict[str, Any] = {}


def _get_state() -> Dict[str, Any]:
    """获取当前Streamlit会话状态；在非Streamlit环境下使用后备字典"""
    if get_script_run_ctx is None:
        return _FALLBACK_STATE
    try:
        if get_script_run_ctx() is None:
            return _FALLBACK_STATE
    except RuntimeError:
        return _FALLBACK_STATE
    return st.session_state


# 设置页面配置
st.set_page_config(page_title="生成2D分子描述符", page_icon="🧬", layout="wide")

st.title("🧬 生成2D分子描述符")

# 数据目录设置
DATA_DIR = os.path.abspath("data")


def list_data_folders():
    """列出data目录下的所有文件夹"""
    if not os.path.exists(DATA_DIR):
        return []
    folders = []
    for item in os.listdir(DATA_DIR):
        item_path = os.path.join(DATA_DIR, item)
        if os.path.isdir(item_path):
            folders.append(item)
    return sorted(folders)


def list_csv_files_in_folder(folder_name):
    """列出指定文件夹中的CSV文件"""
    if not folder_name:
        return []
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.isdir(folder_path):
        return []
    return [
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith(".csv") and os.path.isfile(os.path.join(folder_path, f))
    ]


st.markdown(
    """
基于SMILES字符串生成多种2D分子描述符和指纹，用于机器学习和QSAR建模。

🧬 **分子描述符**: 分子量、LogP、TPSA、拓扑指数等50+个描述符  
🔗 **分子指纹**: Morgan、RDKit、MACCS、Avalon等多种指纹  
⚡ **并行处理**: 支持多线程加速处理大数据集  
📊 **批量生成**: 支持多种描述符类型同时生成  
📈 **质量控制**: 自动验证SMILES有效性和处理异常
"""
)

# 文件选择
st.header("1. 📂 选择输入文件")

available_folders = list_data_folders()
if not available_folders:
    st.error(f"在 '{DATA_DIR}' 目录中未找到数据文件夹。请先运行基础成药性筛选页面。")
    st.stop()

# 文件夹选择
selected_folder = st.selectbox(
    "选择数据文件夹:",
    options=available_folders,
    index=0,
    help="选择包含CSV文件的文件夹",
)

if selected_folder:
    csv_files = list_csv_files_in_folder(selected_folder)

    if not csv_files:
        st.warning(f"选中的文件夹中没有找到CSV文件: {selected_folder}")
        st.stop()

    # 文件选择
    selected_file = st.selectbox(
        "选择CSV文件:", options=csv_files, index=0, help="选择包含SMILES列的CSV文件"
    )

    if selected_file:
        file_path = os.path.join(DATA_DIR, selected_folder, selected_file)
        st.success(f"已选择文件: {file_path}")

        # 显示文件信息
        try:
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            st.info(f"📁 文件大小: {file_size:.2f} MB")
        except Exception:
            pass

# 读取和预览数据
if "selected_file" in locals() and selected_file:
    st.header("2. 📊 数据预览")

    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)

        st.success(f"成功读取数据: {len(df)} 行 × {len(df.columns)} 列")

        # 显示基本信息
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("总分子数", len(df))
        with col2:
            st.metric("列数", len(df.columns))
        with col3:
            memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
            st.metric("内存使用", f"{memory_usage:.1f} MB")

        # 显示前几行
        st.subheader("数据预览")
        st.dataframe(df.head(10))

        # 检查SMILES列
        smiles_candidates = [col for col in df.columns if "smiles" in col.lower()]

        if not smiles_candidates:
            st.error("未找到SMILES列！请确保CSV文件包含名为'SMILES'的列。")
            st.stop()

        # SMILES列选择
        smiles_column = st.selectbox(
            "选择SMILES列:",
            options=smiles_candidates,
            index=0,
            help="选择包含SMILES字符串的列",
        )

        # 构建分子缓存并验证有效性
        if smiles_column:
            smiles_series = df[smiles_column]
            cache_key = (selected_folder, selected_file, smiles_column)
            state = _get_state()

            if state.get("mol_cache_key") != cache_key:
                with st.spinner("正在解析SMILES并构建分子缓存..."):
                    mol_cache, valid_mask = build_molecule_cache(smiles_series)
                state["mol_cache_key"] = cache_key
                state["mol_cache"] = mol_cache
                state["mol_valid_mask"] = valid_mask
            else:
                mol_cache = state.get("mol_cache")
                valid_mask = state.get("mol_valid_mask")
                if mol_cache is None or valid_mask is None:
                    with st.spinner("正在解析SMILES并构建分子缓存..."):
                        mol_cache, valid_mask = build_molecule_cache(smiles_series)
                    state["mol_cache"] = mol_cache
                    state["mol_valid_mask"] = valid_mask

            valid_count = int(valid_mask.sum())
            total_smiles = len(valid_mask)
            invalid_count = total_smiles - valid_count

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "有效SMILES",
                    valid_count,
                    delta=f"{valid_count/total_smiles*100:.1f}%",
                )
            with col2:
                st.metric(
                    "无效SMILES",
                    invalid_count,
                    delta=f"{invalid_count/total_smiles*100:.1f}%",
                )
            with col3:
                st.metric("有效率", f"{valid_count/total_smiles*100:.1f}%")

            if invalid_count > 0:
                st.warning(
                    f"发现 {invalid_count} 个无效SMILES，将在生成描述符时用NaN填充"
                )

                invalid_examples = smiles_series[~valid_mask].dropna().head(5)
                if not invalid_examples.empty:
                    with st.expander("查看无效SMILES示例"):
                        for idx, smiles in invalid_examples.items():
                            st.code(f"行 {idx}: {smiles}")

    except Exception as e:
        st.error(f"读取CSV文件失败: {e}")
        st.stop()

# 描述符类型选择
if "df" in locals():
    st.header("3. 🧬 选择描述符类型")

    # 分子描述符选择
    st.subheader("分子描述符")
    include_molecular_descriptors = st.checkbox(
        "包含分子描述符",
        value=True,
        help="包含50+个常用分子描述符：分子量、LogP、TPSA、拓扑指数等",
    )

    if include_molecular_descriptors:
        st.info(
            "将生成50+个分子描述符，包括：分子量、LogP、氢键供体/受体、TPSA、旋转键数、芳香环数、拓扑指数、BCUT描述符等"
        )

    # 分子指纹选择
    st.subheader("分子指纹")

    fingerprint_types = []
    fp_config = {}

    # Morgan指纹
    include_morgan = st.checkbox(
        "Morgan指纹 (ECFP)", value=True, help="基于原子环境的圆形指纹，最常用的分子指纹"
    )

    if include_morgan:
        fingerprint_types.append("morgan")
        col1, col2 = st.columns(2)
        with col1:
            morgan_radius = st.selectbox(
                "Morgan半径:",
                options=[1, 2, 3, 4],
                index=1,
                help="指纹半径，影响原子环境的大小",
            )
        with col2:
            morgan_bits = st.selectbox(
                "Morgan位数:",
                options=[512, 1024, 2048, 4096],
                index=1,
                help="指纹向量的长度",
            )
        fp_config["morgan"] = {"radius": morgan_radius, "nbits": morgan_bits}

    # RDKit指纹
    include_rdkit = st.checkbox("RDKit指纹", value=False, help="基于路径的分子指纹")

    if include_rdkit:
        fingerprint_types.append("rdkit")
        rdkit_bits = st.selectbox(
            "RDKit位数:", options=[512, 1024, 2048, 4096], index=1, key="rdkit_bits"
        )
        fp_config["rdkit"] = {"nbits": rdkit_bits}

    # MACCS指纹
    include_maccs = st.checkbox(
        "MACCS指纹", value=False, help="基于药物化学子结构的166位指纹"
    )

    if include_maccs:
        fingerprint_types.append("maccs")
        st.info("MACCS指纹固定为167位，基于药物化学相关的子结构")

    # Avalon指纹
    include_avalon = st.checkbox(
        "Avalon指纹",
        value=False,
        help="基于特征路径的分子指纹",
        disabled=not AVALON_AVAILABLE,
    )

    if include_avalon and not AVALON_AVAILABLE:
        st.warning(
            "当前RDKit环境未提供Avalon模块，请安装带Avalon支持的RDKit或关闭该选项。"
        )
    elif include_avalon:
        fingerprint_types.append("avalon")
        avalon_bits = st.selectbox(
            "Avalon位数:", options=[512, 1024, 2048, 4096], index=1, key="avalon_bits"
        )
        fp_config["avalon"] = {"nbits": avalon_bits}

    # 原子对指纹
    include_atompairs = st.checkbox(
        "原子对指纹", value=False, help="基于原子对的分子指纹"
    )

    if include_atompairs:
        fingerprint_types.append("atompairs")
        ap_bits = st.selectbox(
            "原子对位数:", options=[512, 1024, 2048, 4096], index=1, key="ap_bits"
        )
        fp_config["atompairs"] = {"nbits": ap_bits}

    # 扭转指纹
    include_torsions = st.checkbox(
        "拓扑扭转指纹", value=False, help="基于拓扑扭转的分子指纹"
    )

    if include_torsions:
        fingerprint_types.append("torsions")
        torsion_bits = st.selectbox(
            "扭转指纹位数:",
            options=[512, 1024, 2048, 4096],
            index=1,
            key="torsion_bits",
        )
        fp_config["torsions"] = {"nbits": torsion_bits}

    # 汇总选择的描述符类型
    descriptor_types = []
    if include_molecular_descriptors:
        descriptor_types.append("molecular_descriptors")
    descriptor_types.extend(fingerprint_types)

    if not descriptor_types:
        st.warning("请至少选择一种描述符类型！")
        st.stop()

    # 显示选择摘要
    st.subheader("选择摘要")
    col1, col2 = st.columns(2)

    with col1:
        st.info("**已选择的描述符类型:**")
        if include_molecular_descriptors:
            st.write("- 🧬 分子描述符 (50+个)")
        for fp_type in fingerprint_types:
            if fp_type == "morgan":
                st.write(f"- 🔗 Morgan指纹 (半径{morgan_radius}, {morgan_bits}位)")
            elif fp_type == "rdkit":
                st.write(f"- 🔗 RDKit指纹 ({rdkit_bits}位)")
            elif fp_type == "maccs":
                st.write("- 🔗 MACCS指纹 (167位)")
            elif fp_type == "avalon":
                st.write(f"- 🔗 Avalon指纹 ({avalon_bits}位)")
            elif fp_type == "atompairs":
                st.write(f"- 🔗 原子对指纹 ({ap_bits}位)")
            elif fp_type == "torsions":
                st.write(f"- 🔗 扭转指纹 ({torsion_bits}位)")

    with col2:
        # 估算总列数
        total_columns = len(df.columns)  # 原有列
        if include_molecular_descriptors:
            total_columns += 50  # 分子描述符
        for fp_type in fingerprint_types:
            if fp_type == "maccs":
                total_columns += 167
            elif fp_type in fp_config:
                total_columns += fp_config[fp_type]["nbits"]

        st.metric("预计总列数", total_columns)

        # 估算内存使用
        estimated_memory = (
            len(df) * total_columns * 8 / (1024 * 1024)
        )  # 假设每个值8字节
        st.metric("预计内存", f"{estimated_memory:.1f} MB")

        if estimated_memory > 1000:
            st.warning("⚠️ 预计内存使用较大，建议减少指纹位数或分批处理")

# 处理选项
st.header("4. ⚙️ 处理选项")

processing_options = [
    "处理所有分子",
    "仅处理前100个分子",
    "仅处理前500个分子",
    "仅处理前1000个分子",
]

selected_scope = st.selectbox(
    "处理范围:", options=processing_options, index=0, help="选择要处理的分子数量"
)

st.caption("当前版本采用单线程顺序处理，如需更快速度可减少指纹位数或分批导出。")

# 确定处理数量
if "df" in locals():
    total_molecules = len(df)
    if selected_scope == "处理所有分子":
        molecules_to_process = total_molecules
    elif selected_scope == "仅处理前100个分子":
        molecules_to_process = min(100, total_molecules)
    elif selected_scope == "仅处理前500个分子":
        molecules_to_process = min(500, total_molecules)
    elif selected_scope == "仅处理前1000个分子":
        molecules_to_process = min(1000, total_molecules)
    else:
        molecules_to_process = total_molecules

    max_workers = max(1, mp.cpu_count() or 1)
    worker_count = st.slider(
        "并行进程数:",
        min_value=1,
        max_value=max_workers,
        value=min(4, max_workers),
        help="自动检测CPU核心数，可根据数据量调整并行进程数",
    )

    st.caption("当数据量较大时建议使用多进程，数据量很小时单进程更高效。")
    st.info(f"将处理 {molecules_to_process} 个分子")

# 生成描述符
if "df" in locals() and "smiles_column" in locals() and descriptor_types:
    st.header("5. 🚀 生成2D描述符")

    if st.button("开始生成2D描述符", type="primary"):
        start_time = time.time()

        # 准备数据
        df_subset = df.head(molecules_to_process).copy()
        if smiles_column not in df_subset.columns:
            st.error(
                f"选择的SMILES列 `{smiles_column}` 不在当前数据中，请重新选择后再试。"
            )
            st.stop()
        smiles_series = df_subset[smiles_column]

        state = _get_state()
        mol_cache = state.get("mol_cache")
        valid_mask_series = state.get("mol_valid_mask")
        if mol_cache is None or valid_mask_series is None:
            mol_cache, valid_mask_series = build_molecule_cache(df[smiles_column])
            state["mol_cache"] = mol_cache
            state["mol_valid_mask"] = valid_mask_series
            state["mol_cache_key"] = (selected_folder, selected_file, smiles_column)

        mol_subset = mol_cache[:molecules_to_process]

        st.info(f"开始生成2D描述符，共 {len(df_subset)} 个分子...")

        progress_bar = st.progress(0.0)
        status_text = st.empty()

        use_parallel = worker_count > 1 and len(df_subset) >= worker_count * 2

        try:
            if len(df_subset) == 0:
                final_df = df_subset.copy()
            elif use_parallel:
                status_text.text("正在并行生成描述符...")
                num_chunks = max(1, min(len(df_subset), worker_count * 2))
                chunk_indices = [
                    idxs
                    for idxs in np.array_split(np.arange(len(df_subset)), num_chunks)
                    if len(idxs) > 0
                ]

                if not chunk_indices:
                    final_df = df_subset.copy()
                else:
                    tasks = []
                    for idxs in chunk_indices:
                        chunk_df = df_subset.iloc[idxs].copy()
                        chunk_df["__row_id"] = chunk_df.index
                        tasks.append(
                            (
                                chunk_df.to_dict(orient="records"),
                                descriptor_types,
                                fingerprint_types,
                                fp_config,
                                smiles_column,
                            )
                        )

                    results = []
                    ctx = mp.get_context("spawn")
                    with ctx.Pool(processes=worker_count) as pool:
                        for completed, chunk_result in enumerate(
                            pool.imap_unordered(process_chunk_worker, tasks), start=1
                        ):
                            results.append(chunk_result)
                            progress_bar.progress(
                                min(0.95, completed / len(chunk_indices))
                            )
                            status_text.text(
                                f"并行处理中... {completed}/{len(chunk_indices)} 批完成"
                            )

                    final_df = pd.concat(results, ignore_index=True)
                    final_df = (
                        final_df.sort_values("__row_id")
                        .drop(columns="__row_id")
                        .reset_index(drop=True)
                    )

            else:
                all_descriptors = []

                if "molecular_descriptors" in descriptor_types:
                    status_text.text("正在生成分子描述符...")

                    def update_progress(current, total):
                        progress = current / total * 0.4
                        progress_bar.progress(min(0.95, progress))
                        status_text.text(f"正在生成分子描述符... {current}/{total}")

                    mol_desc_df = generate_molecular_descriptors(
                        mol_subset, update_progress
                    )
                    all_descriptors.append(mol_desc_df)
                    st.success(
                        f"✅ 分子描述符生成完成: {len(mol_desc_df.columns)} 个描述符"
                    )

                fingerprint_progress_start = (
                    0.4 if "molecular_descriptors" in descriptor_types else 0.0
                )
                fingerprint_progress_per_type = (
                    (1.0 - fingerprint_progress_start) / len(fingerprint_types)
                    if fingerprint_types
                    else 0
                )

                for i, fp_type in enumerate(fingerprint_types):
                    status_text.text(f"正在生成{fp_type.upper()}指纹...")

                    def update_fp_progress(
                        current,
                        total,
                        base=fingerprint_progress_start,
                        idx=i,
                        label=fp_type.upper(),
                    ):
                        base_progress = base + idx * fingerprint_progress_per_type
                        type_progress = (
                            (current / total) * fingerprint_progress_per_type
                            if total
                            else 0
                        )
                        progress_bar.progress(min(0.95, base_progress + type_progress))
                        status_text.text(f"正在生成{label}指纹... {current}/{total}")

                    if fp_type == "morgan":
                        fp_df = generate_fingerprints(
                            mol_subset,
                            fp_type,
                            radius=fp_config[fp_type]["radius"],
                            nbits=fp_config[fp_type]["nbits"],
                            progress_callback=update_fp_progress,
                        )
                    elif fp_type == "maccs":
                        fp_df = generate_fingerprints(
                            mol_subset, fp_type, progress_callback=update_fp_progress
                        )
                    else:
                        fp_df = generate_fingerprints(
                            mol_subset,
                            fp_type,
                            nbits=fp_config[fp_type]["nbits"],
                            progress_callback=update_fp_progress,
                        )

                    all_descriptors.append(fp_df)
                    st.success(
                        f"✅ {fp_type.upper()}指纹生成完成: {len(fp_df.columns)} 个特征"
                    )

                final_df = merge_feature_frames(df_subset.copy(), all_descriptors)

            progress_bar.progress(1.0)
            status_text.text("处理完成")
            total_time = time.time() - start_time

            st.success("🎉 2D描述符生成完成！")
            st.info(
                f"总耗时: {total_time:.1f}秒 | 最终数据: {len(final_df)} 行 × {len(final_df.columns)} 列"
            )

            # 显示结果统计
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("原始列数", len(df_subset.columns))
            with col2:
                st.metric("新增列数", len(final_df.columns) - len(df_subset.columns))
            with col3:
                st.metric("最终列数", len(final_df.columns))

            # 预览结果
            st.subheader("结果预览")

            # 显示新增的描述符列
            original_columns = set(df_subset.columns)
            new_columns = [
                col for col in final_df.columns if col not in original_columns
            ]

            if new_columns:
                st.info(f"新增了 {len(new_columns)} 个描述符特征")
                with st.expander("查看新增的描述符列"):
                    # 按类型分组显示
                    if "molecular_descriptors" in descriptor_types:
                        mol_desc_cols = [
                            col
                            for col in new_columns
                            if not any(
                                fp in col
                                for fp in [
                                    "MORGAN",
                                    "RDKIT",
                                    "MACCS",
                                    "AVALON",
                                    "ATOMPAIRS",
                                    "TORSIONS",
                                ]
                            )
                        ]
                        if mol_desc_cols:
                            st.write("**分子描述符:**")
                            st.write(", ".join(mol_desc_cols))

                    for fp_type in fingerprint_types:
                        fp_cols = [col for col in new_columns if fp_type.upper() in col]
                        if fp_cols:
                            st.write(f"**{fp_type.upper()}指纹特征:**")
                            st.write(
                                f"{len(fp_cols)} 个二进制特征 ({fp_cols[0]} ... {fp_cols[-1]})"
                            )

            # 显示前几行
            st.dataframe(final_df.head(5))

            # 保存结果
            output_filename = f"2d_fingerprint_{selected_file}"
            output_path = os.path.join(DATA_DIR, selected_folder, output_filename)

            try:
                final_df.to_csv(output_path, index=False)
                st.success(f"✅ 结果已保存到: {output_path}")

                # 显示文件大小
                output_size = os.path.getsize(output_path) / (1024 * 1024)
                st.info(f"📁 输出文件大小: {output_size:.2f} MB")

                with open(output_path, "rb") as fh:
                    st.download_button(
                        label="📥 下载2D描述符CSV文件",
                        data=fh,
                        file_name=output_filename,
                        mime="text/csv",
                        help="下载包含2D描述符的完整CSV文件",
                    )

            except Exception as e:
                st.error(f"保存文件失败: {e}")

            # 显示描述符统计
            st.subheader("描述符统计")

            # 只对数值列进行统计
            numeric_columns = final_df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                stats_df = final_df[numeric_columns].describe()
                st.dataframe(stats_df)

                # 检查缺失值
                missing_stats = final_df[numeric_columns].isnull().sum()
                missing_stats = missing_stats[missing_stats > 0]

                if len(missing_stats) > 0:
                    st.warning("⚠️ 缺失值统计:")
                    for col, missing_count in missing_stats.items():
                        st.write(
                            f"- {col}: {missing_count} 个缺失值 ({missing_count/len(final_df)*100:.1f}%)"
                        )
                else:
                    st.success("✅ 所有描述符都已成功计算，无缺失值")

        except Exception as e:
            st.error(f"生成2D描述符时出错: {e}")
            import traceback

            st.code(traceback.format_exc(), language="text")

# 帮助信息
with st.sidebar:
    st.header("📚 帮助信息")

    with st.expander("🧬 分子描述符说明"):
        st.markdown(
            """
        **常用分子描述符：**
        - **MolWt**: 分子量
        - **LogP**: 脂水分配系数
        - **TPSA**: 拓扑极性表面积
        - **NumHDonors/NumHAcceptors**: 氢键供体/受体数
        - **NumRotatableBonds**: 可旋转键数
        - **FractionCsp3**: sp3碳原子分数
        - **Chi0v-Chi4v**: 价连接性指数
        - **Kappa1-Kappa3**: 分子形状指数
        - **BertzCT**: 分子复杂度
        - **QED**: 类药性评分
        """
        )

    with st.expander("🔗 分子指纹说明"):
        st.markdown(
            """
        **指纹类型对比：**
        
        **Morgan (ECFP)**:
        - 最常用的圆形指纹
        - 基于原子环境
        - 适合相似性搜索
        
        **RDKit**:
        - 基于路径的指纹
        - 考虑分子路径信息
        
        **MACCS**:
        - 167位固定指纹
        - 基于药物化学子结构
        - 适合药物分子
        
        **Avalon**:
        - 基于特征路径
        - 良好的多样性表现
        
        **原子对**:
        - 基于原子对距离
        - 适合骨架跃迁
        
        **扭转**:
        - 基于拓扑扭转
        - 捕获3D特征
        """
        )

    with st.expander("⚙️ 参数建议"):
        st.markdown(
            """
        **指纹参数建议：**
        
        **Morgan半径：**
        - 半径2: 标准设置，适合大多数应用
        - 半径3: 更大的原子环境，适合复杂分子
        
        **指纹位数：**
        - 1024位: 标准设置，平衡性能和精度
        - 2048位: 更高精度，适合大数据集
        
        **并行处理：**
        - 线程数 = CPU核心数的50-75%
        - 批处理大小: 100-500个分子
        """
        )

    with st.expander("🎯 使用建议"):
        st.markdown(
            """
        **应用场景：**
        
        **机器学习：**
        - 分子描述符 + Morgan指纹
        - 考虑使用特征选择
        
        **相似性搜索：**
        - Morgan指纹 (半径2, 1024位)
        - 或MACCS指纹
        
        **QSAR建模：**
        - 分子描述符为主
        - 结合Morgan指纹
        
        **虚拟筛选：**
        - 多种指纹组合
        - 考虑计算成本
        """
        )

st.divider()
st.markdown("🧬 **CADD-Toolbox** - 2D分子描述符生成工具")
