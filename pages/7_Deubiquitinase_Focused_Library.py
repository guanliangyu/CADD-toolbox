"""
CADD-Toolbox - Deubiquitinase Focused Library 页面
当前版本先提供 CSV 读取与预览能力
"""

from __future__ import annotations

import importlib
from dataclasses import asdict
from pathlib import Path
import re

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import deubiquitinase_library_data as dub_data


dub_data = importlib.reload(dub_data)
CsvLoadMeta = dub_data.CsvLoadMeta


SESSION_KEY_DUB_DF = "dub_library_df"
SESSION_KEY_DUB_META = "dub_library_meta"
SESSION_KEY_DUB_SOURCE = "dub_library_source"
SESSION_KEY_DUB_FILE_PATH = "dub_library_file_path"
SESSION_KEY_DUB_PARTIAL_LOAD = "dub_library_partial_load"
DEFAULT_DATA_DIR = Path("data")


st.set_page_config(
    page_title="Deubiquitinase Focused Library",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 Deubiquitinase Focused Library")
st.markdown("当前页面已支持 `.csv` 读取、基础信息展示和数据预览。")


def _store_dataset(
    df: pd.DataFrame,
    meta: CsvLoadMeta,
    file_path: str | Path | None = None,
    partial_load: bool = False,
) -> None:
    """将读取到的数据写入会话状态，供后续流程复用"""
    st.session_state[SESSION_KEY_DUB_DF] = df
    st.session_state[SESSION_KEY_DUB_META] = asdict(meta)
    st.session_state[SESSION_KEY_DUB_SOURCE] = meta.source_name
    st.session_state[SESSION_KEY_DUB_FILE_PATH] = str(file_path) if file_path is not None else None
    st.session_state[SESSION_KEY_DUB_PARTIAL_LOAD] = bool(partial_load)


def _uploaded_file_signature(uploaded_file) -> str:
    """构建上传文件签名，用于生成稳定组件 key"""
    file_name = getattr(uploaded_file, "name", "uploaded.csv")
    file_size = getattr(uploaded_file, "size", 0)
    return f"{file_name}_{file_size}"


def _load_dataset_from_session() -> tuple[pd.DataFrame | None, CsvLoadMeta | None]:
    """从会话状态恢复当前数据集"""
    df = st.session_state.get(SESSION_KEY_DUB_DF)
    meta_dict = st.session_state.get(SESSION_KEY_DUB_META)
    if df is None or meta_dict is None:
        return None, None

    try:
        meta = CsvLoadMeta(**meta_dict)
    except TypeError:
        return None, None
    return df, meta


def _render_dataset_summary(df: pd.DataFrame, meta: CsvLoadMeta) -> None:
    """渲染当前数据集摘要和预览"""
    st.subheader("2. 当前数据集概览")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("记录数", f"{meta.n_rows:,}")
    with col2:
        st.metric("字段数", f"{meta.n_cols}")
    with col3:
        st.metric("编码", meta.encoding)
    with col4:
        separator_label = r"\t (Tab)" if meta.separator == "\t" else meta.separator
        st.metric("分隔符", separator_label)

    st.caption(f"数据来源: `{meta.source_name}`")

    preview_mode = st.radio(
        "预览范围",
        ["前N行（推荐）", "全部行"],
        horizontal=True,
        key="dub_preview_mode",
    )

    if preview_mode == "全部行":
        st.caption(f"当前显示全部 `{len(df):,}` 行。")
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        max_preview_rows = int(max(10, min(5000, len(df))))
        default_preview_rows = int(min(100, max_preview_rows))
        preview_rows = int(
            st.slider(
                "预览行数",
                min_value=10,
                max_value=max_preview_rows,
                value=default_preview_rows,
                step=10,
                key="dub_preview_rows",
            )
        )
        st.dataframe(df.head(preview_rows), use_container_width=True, hide_index=True)

    st.markdown("**字段信息**")
    schema_df = pd.DataFrame(
        {
            "列名": df.columns,
            "数据类型": df.dtypes.astype(str),
            "缺失值数量": df.isna().sum().to_numpy(),
        }
    )
    st.dataframe(schema_df, use_container_width=True, hide_index=True)

    name_lower = [col.lower() for col in df.columns]
    candidate_smiles = [col for col, lower in zip(df.columns, name_lower) if "smiles" in lower]
    candidate_id = [col for col, lower in zip(df.columns, name_lower) if lower in {"id", "compound_id", "molecule_id"}]
    st.info(
        "字段建议: "
        f"SMILES候选列={candidate_smiles if candidate_smiles else '未检测到'}；"
        f"ID候选列={candidate_id if candidate_id else '未检测到'}。"
    )


def _render_target_method_bubble(df: pd.DataFrame) -> None:
    """渲染 Target_normalized 和 Method_normalized 的计数气泡图"""
    target_col = "Target_normalized"
    method_col = "Method_normalized"
    missing_cols = [col for col in (target_col, method_col) if col not in df.columns]
    if missing_cols:
        st.info(
            "无法绘制气泡图，缺少字段: "
            + "、".join(f"`{col}`" for col in missing_cols)
        )
        return

    grouped_df = (
        df[[target_col, method_col]]
        .dropna(subset=[target_col, method_col])
        .groupby([target_col, method_col], dropna=False)
        .size()
        .reset_index(name="compound_count")
        .sort_values("compound_count", ascending=False)
    )
    if grouped_df.empty:
        st.info("未统计到有效 Target/Method 组合，请检查数据内容。")
        return

    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    with stats_col1:
        st.metric("Target 数量", int(grouped_df[target_col].nunique()))
    with stats_col2:
        st.metric("Method 数量", int(grouped_df[method_col].nunique()))
    with stats_col3:
        st.metric("Target-Method 组合", int(len(grouped_df)))
    with stats_col4:
        st.metric("组合计数总和", int(grouped_df["compound_count"].sum()))

    sorted_targets = sorted(
        grouped_df[target_col].dropna().unique().tolist(),
        key=lambda x: str(x).lower(),
    )
    sorted_methods = sorted(
        grouped_df[method_col].dropna().unique().tolist(),
        key=lambda x: str(x).lower(),
    )
    method_palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    method_color_map = {
        method: method_palette[idx % len(method_palette)]
        for idx, method in enumerate(sorted_methods)
    }

    fig_target_method = px.scatter(
        grouped_df,
        x=target_col,
        y=method_col,
        size="compound_count",
        color=method_col,
        category_orders={
            target_col: sorted_targets,
            method_col: sorted_methods,
        },
        color_discrete_map=method_color_map,
        hover_data={
            target_col: True,
            method_col: True,
            "compound_count": True,
        },
        size_max=42,
        title="气泡图 1：X=Target，Y=Method（点大小按数量比例）",
    )
    fig_target_method.update_layout(
        xaxis_title="Target_normalized",
        yaxis_title="Method_normalized",
        legend_title_text="Method_normalized",
    )
    fig_target_method.update_xaxes(
        showgrid=True,
        gridcolor="rgba(160, 160, 160, 0.45)",
        gridwidth=1,
    )
    st.plotly_chart(fig_target_method, use_container_width=True)

    fig_method_target = px.scatter(
        grouped_df,
        x=method_col,
        y=target_col,
        size="compound_count",
        color=method_col,
        category_orders={
            method_col: sorted_methods,
            target_col: sorted_targets,
        },
        color_discrete_map=method_color_map,
        hover_data={
            target_col: True,
            method_col: True,
            "compound_count": True,
        },
        size_max=42,
        title="气泡图 2：X=Method，Y=Target（纵向）",
    )
    vertical_height = int(min(3200, max(1200, len(sorted_targets) * 45)))
    vertical_width = 650
    fig_method_target.update_layout(
        xaxis_title="Method_normalized",
        yaxis_title="Target_normalized",
        legend_title_text="Method_normalized",
        height=vertical_height,
        width=vertical_width,
        margin=dict(l=120, r=30, t=80, b=90),
    )
    fig_method_target.update_xaxes(automargin=True)
    fig_method_target.update_yaxes(automargin=True)
    st.plotly_chart(fig_method_target, use_container_width=False)
    st.dataframe(grouped_df, use_container_width=True, hide_index=True)


def _sync_meta_shape(df: pd.DataFrame) -> None:
    """同步会话中元信息的行列统计"""
    meta_dict = st.session_state.get(SESSION_KEY_DUB_META)
    if not isinstance(meta_dict, dict):
        return
    updated_meta = dict(meta_dict)
    updated_meta["n_rows"] = int(len(df))
    updated_meta["n_cols"] = int(df.shape[1])
    st.session_state[SESSION_KEY_DUB_META] = updated_meta


def _render_column_unification(df: pd.DataFrame) -> pd.DataFrame | None:
    """渲染字段统计与标准化统一面板"""
    st.subheader("3. 字段统计与统一格式")
    if df.empty:
        st.info("当前数据为空，无法进行字段统计与统一。")
        return None

    selected_column = st.selectbox(
        "选择要处理的列（已包含全部字段）",
        options=df.columns.tolist(),
        key="dub_unify_selected_column",
    )
    delimiter = st.text_input(
        "该列多值分隔符",
        value=";",
        key="dub_unify_delimiter",
        help="若单元格中包含多个字段值（如 A;B;C），请填写分隔符。",
    )

    if not delimiter:
        st.warning("分隔符不能为空。")
        return None

    option_df, stats = dub_data.analyze_column_field_options(df[selected_column], delimiter=delimiter)

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("总行数", f"{stats['total_rows']:,}")
    with stat_col2:
        st.metric("有可用值行数", f"{stats['non_empty_rows']:,}")
    with stat_col3:
        st.metric("多值行数", f"{stats['multi_value_rows']:,}")
    with stat_col4:
        st.metric("可用字段种类", f"{stats['unique_options']:,}")

    if option_df.empty:
        st.info("该列未解析到可用字段信息。")
        return None

    max_display = int(min(200, len(option_df)))
    default_display = int(min(30, len(option_df)))
    display_rows = int(
        st.slider(
            "显示可用字段条数",
            min_value=1,
            max_value=max_display,
            value=default_display,
            step=1,
            key="dub_unify_display_rows",
        )
    )
    st.dataframe(option_df.head(display_rows), use_container_width=True, hide_index=True)

    normalized_col_name = re.sub(r"[\s_]+", "", selected_column.strip().lower())
    is_submitted_by_column = normalized_col_name == "submittedby"
    if is_submitted_by_column:
        st.caption("检测到 `SubmittedBy` 列：将按“每行第一个字段的第一个单词”进行推荐标准化。")

    unify_mode_label = st.radio(
        "统一方式",
        ["每行取该列第一个字段（推荐）", "统一为指定字段（不匹配则置空）"],
        horizontal=True,
        key="dub_unify_mode",
    )

    selected_field_value: str | None = None
    if unify_mode_label == "统一为指定字段（不匹配则置空）":
        selected_field_value = st.selectbox(
            "选择用于统一的字段值",
            options=option_df["字段值"].tolist(),
            key="dub_unify_selected_field_value",
        )

    write_mode = st.radio(
        "写入方式",
        ["新增标准化列", "覆盖原列"],
        horizontal=True,
        key="dub_unify_write_mode",
    )

    output_column = selected_column
    if write_mode == "新增标准化列":
        output_column = st.text_input(
            "标准化列名",
            value=f"{selected_column}_normalized",
            key="dub_unify_output_column",
        ).strip()
        if not output_column:
            st.warning("标准化列名不能为空。")
            return None

    if st.button("应用统一格式", key="dub_apply_unify", type="primary"):
        try:
            if unify_mode_label == "每行取该列第一个字段（推荐）":
                mode = "first_word" if is_submitted_by_column else "first"
            else:
                mode = "selected"
            standardized_series = dub_data.standardize_multi_value_series(
                df[selected_column],
                mode=mode,
                selected_value=selected_field_value,
                delimiter=delimiter,
            )

            result_df = df.copy()
            result_df[output_column] = standardized_series
            filled_rows = int(result_df[output_column].notna().sum())

            st.session_state[SESSION_KEY_DUB_DF] = result_df
            _sync_meta_shape(result_df)

            st.success(f"已完成统一：输出列 `{output_column}`，填充 {filled_rows:,}/{len(result_df):,} 行。")

            source_path_value = st.session_state.get(SESSION_KEY_DUB_FILE_PATH)
            if source_path_value:
                source_path = Path(source_path_value)
                safe_col_name = re.sub(r"[^A-Za-z0-9._-]+", "_", output_column).strip("_") or "normalized"
                output_path = source_path.with_name(f"{source_path.stem}_{safe_col_name}.csv")
                result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
                st.success(f"已导出 CSV: `{output_path.as_posix()}`")
                if st.session_state.get(SESSION_KEY_DUB_PARTIAL_LOAD):
                    st.warning("当前数据来自预览读取，导出文件仅包含已加载行。若需全量导出，请先使用全量读取。")
            else:
                st.warning("当前数据缺少来源文件路径，未自动导出 CSV。")

            preview_cols = [selected_column] if output_column == selected_column else [selected_column, output_column]
            st.dataframe(
                result_df[preview_cols].head(100),
                use_container_width=True,
                hide_index=True,
            )
            return result_df
        except Exception as exc:
            st.error(f"应用统一格式失败: {exc}")

    return None


st.subheader("1. 读取 Deubiquitinase Focused Library CSV")
upload_tab, local_tab = st.tabs(["上传本地 CSV", "从 data 目录读取"])

with upload_tab:
    dub_data.ensure_data_dir(DEFAULT_DATA_DIR)

    st.markdown("**上传并保存到 `data/` 子文件夹**")
    folder_col1, folder_col2 = st.columns([3, 1])
    with folder_col1:
        new_folder_name = st.text_input(
            "新建文件夹名称",
            key="dub_new_folder_name",
            placeholder="例如: DUB_20260224",
        )
    with folder_col2:
        st.write("")
        if st.button("创建文件夹", key="dub_create_folder_btn", use_container_width=True):
            try:
                created_folder = dub_data.create_data_subfolder(new_folder_name, DEFAULT_DATA_DIR)
                st.session_state["dub_target_folder"] = created_folder.name
                st.success(f"已创建文件夹: data/{created_folder.name}")
                st.rerun()
            except Exception as exc:
                st.error(f"创建文件夹失败: {exc}")

    folders = dub_data.list_data_folders(DEFAULT_DATA_DIR)
    if not folders:
        st.info("当前 `data/` 下还没有文件夹，请先创建文件夹。")
    else:
        default_folder_idx = 0
        default_folder_name = st.session_state.get("dub_target_folder")
        if default_folder_name in folders:
            default_folder_idx = folders.index(default_folder_name)

        target_folder = st.selectbox(
            "选择保存目标文件夹",
            folders,
            index=default_folder_idx,
            key="dub_target_folder_select",
        )
        st.session_state["dub_target_folder"] = target_folder

    uploaded_file = st.file_uploader(
        "上传 Deubiquitinase Focused Library 的 CSV 文件",
        type=["csv"],
        accept_multiple_files=False,
        key="dub_csv_uploader",
    )
    if uploaded_file is not None and folders:
        file_size_mb = getattr(uploaded_file, "size", 0) / (1024 * 1024)
        st.caption(f"文件: `{uploaded_file.name}` | 大小: `{file_size_mb:.2f} MB`")
        upload_sig = _uploaded_file_signature(uploaded_file)

        save_file_name = st.text_input(
            "保存文件名（可修改）",
            value=dub_data.sanitize_uploaded_filename(uploaded_file.name),
            key=f"dub_save_file_name_{upload_sig}",
        )
        overwrite_existing = st.checkbox(
            "覆盖同名文件",
            value=False,
            key=f"dub_overwrite_existing_{upload_sig}",
        )

        read_scope = st.radio(
            "读取范围",
            ["预览模式（推荐）", "全量读取（大文件可能较慢）"],
            index=0,
            horizontal=True,
            key="dub_upload_read_scope",
        )
        preview_nrows = int(
            st.number_input(
                "预览模式读取行数",
                min_value=100,
                max_value=200_000,
                value=20_000,
                step=100,
                key="dub_upload_preview_rows",
                disabled=read_scope != "预览模式（推荐）",
            )
        )
        requested_nrows = preview_nrows if read_scope == "预览模式（推荐）" else None

        if st.button("保存并读取 CSV", key="save_and_load_uploaded_dub_csv", type="primary"):
            try:
                with st.spinner("正在保存并读取文件..."):
                    saved_path = dub_data.save_uploaded_csv_to_data(
                        uploaded_file=uploaded_file,
                        folder_name=target_folder,
                        data_dir=DEFAULT_DATA_DIR,
                        filename=save_file_name,
                        overwrite=overwrite_existing,
                    )
                    loaded_df, loaded_meta = dub_data.load_csv_from_path(saved_path, nrows=requested_nrows)

                _store_dataset(
                    loaded_df,
                    loaded_meta,
                    file_path=saved_path,
                    partial_load=(requested_nrows is not None),
                )
                saved_hint = saved_path.as_posix()
                if requested_nrows is None:
                    st.success(f"已保存并全量读取: `{saved_hint}`")
                else:
                    st.success(
                        f"已保存并读取（预览模式）: `{saved_hint}`，显示前 {len(loaded_df):,} 行"
                    )
            except Exception as exc:
                st.error(f"保存并读取 CSV 失败: {exc}")

with local_tab:
    dub_data.ensure_data_dir(DEFAULT_DATA_DIR)
    folders = dub_data.list_data_folders(DEFAULT_DATA_DIR)
    if not folders:
        st.info("当前 `data/` 目录下没有可选子文件夹。")
    else:
        selected_folder = st.selectbox("选择文件夹", folders, key="dub_folder_select")
        csv_files = dub_data.list_csv_files_in_folder(selected_folder, DEFAULT_DATA_DIR)
        if not csv_files:
            st.info(f"`{selected_folder}` 下没有 CSV 文件。")
        else:
            selected_csv = st.selectbox("选择 CSV 文件", csv_files, key="dub_csv_select")
            if st.button("读取所选 CSV", key="load_selected_dub_csv", type="primary"):
                try:
                    file_path = DEFAULT_DATA_DIR / selected_folder / selected_csv
                    loaded_df, loaded_meta = dub_data.load_csv_from_path(file_path)
                    _store_dataset(loaded_df, loaded_meta, file_path=file_path, partial_load=False)
                    st.success(f"已成功读取文件: {selected_folder}/{selected_csv}")
                except Exception as exc:
                    st.error(f"读取本地 CSV 失败: {exc}")

active_df, active_meta = _load_dataset_from_session()

if active_df is not None and active_meta is not None:
    _render_dataset_summary(active_df, active_meta)
    updated_df = _render_column_unification(active_df)
    if updated_df is not None:
        active_df = updated_df
    if st.button("绘制 Target/Method 计数气泡图", key="show_target_method_bubble"):
        _render_target_method_bubble(active_df)
    if st.button("清空当前数据", key="clear_dub_library_data"):
        st.session_state.pop(SESSION_KEY_DUB_DF, None)
        st.session_state.pop(SESSION_KEY_DUB_META, None)
        st.session_state.pop(SESSION_KEY_DUB_SOURCE, None)
        st.session_state.pop(SESSION_KEY_DUB_FILE_PATH, None)
        st.session_state.pop(SESSION_KEY_DUB_PARTIAL_LOAD, None)
        st.rerun()
else:
    st.caption("尚未加载 CSV 数据，请在上方选择上传或本地文件。")

st.markdown("---")
st.caption("后续将基于当前数据继续接入 Deubiquitinase Focused Library 专项分析流程。")
