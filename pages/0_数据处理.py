"""
CADD-Toolbox - 数据预处理页面
提供文件管理和数据标准化功能
"""

from pathlib import Path
import json
import hashlib
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt  # type: ignore[attr-defined]

CHUNK_SIZE = 10 * 1024 * 1024  # 10MB 分块写入
DOWNLOAD_INLINE_THRESHOLD = 50 * 1024 * 1024  # 50MB 内联下载阈值
PREVIEW_DEFAULT_ROWS = 1000
FOLDER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._\-\u4e00-\u9fff]+$")

CSV_PARSER_DISPLAY = {
    "pandas": "Pandas (兼容性最佳)",
    "pyarrow": "PyArrow (多线程高性能)",
    "polars": "Polars (Rust引擎)",
}

CSV_PARSER_AVAILABLE: dict[str, bool] = {"pandas": True}

try:
    import pyarrow.csv as pacsv  # type: ignore

    CSV_PARSER_AVAILABLE["pyarrow"] = True
except ImportError:  # pragma: no cover - 可选依赖
    pacsv = None
    CSV_PARSER_AVAILABLE["pyarrow"] = False

try:
    import polars as pl  # type: ignore

    CSV_PARSER_AVAILABLE["polars"] = True
except ImportError:  # pragma: no cover - 可选依赖
    pl = None
    CSV_PARSER_AVAILABLE["polars"] = False

# 设置页面配置
st.set_page_config(page_title="数据预处理", page_icon="📁", layout="wide")

st.title("📁 数据预处理")

# 数据目录设置
DATA_DIR = Path("data")


def ensure_data_dir() -> Path:
    """确保 data 目录存在，仅在首次创建时提示"""
    data_dir = DATA_DIR
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        if not st.session_state.get("_data_dir_notice_shown", False):
            st.session_state["_data_dir_notice_shown"] = True
            st.info(f"已创建数据目录: {data_dir.resolve()}")
    return data_dir


def list_data_folders() -> list[str]:
    """列出 data 目录下的文件夹名称"""
    base_dir = ensure_data_dir()
    return sorted([item.name for item in base_dir.iterdir() if item.is_dir()])


def list_files_in_folder(folder_name: str) -> list[str]:
    """列出指定文件夹中的所有文件"""
    if not folder_name:
        return []
    folder_path = ensure_data_dir() / folder_name
    if not folder_path.exists() or not folder_path.is_dir():
        return []
    return sorted([item.name for item in folder_path.iterdir() if item.is_file()])


def create_new_folder(folder_name: str):
    """在 data 目录下创建新文件夹"""
    is_valid, normalized_or_message = validate_folder_name(folder_name)
    if not is_valid:
        return False, normalized_or_message

    safe_folder_name = normalized_or_message
    folder_path = ensure_data_dir() / safe_folder_name
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=True)
        return True, f"成功创建文件夹: {safe_folder_name}"
    return False, f"文件夹已存在: {safe_folder_name}"


def validate_folder_name(folder_name: str) -> tuple[bool, str]:
    """校验并规范化文件夹名称，避免路径穿越和非法字符"""
    normalized_name = (folder_name or "").strip()
    if not normalized_name:
        return False, "请输入有效的文件夹名称"
    if normalized_name in {".", ".."}:
        return False, "文件夹名称不能为 . 或 .."
    if "/" in normalized_name or "\\" in normalized_name:
        return False, "文件夹名称不能包含路径分隔符"
    if not FOLDER_NAME_PATTERN.fullmatch(normalized_name):
        return False, "文件夹名称仅支持中文、英文、数字、点、下划线和短横线"
    return True, normalized_name


def sanitize_uploaded_filename(file_name: str) -> str:
    """清理上传文件名，移除潜在路径信息"""
    return Path(file_name).name.strip()


def stable_widget_key(prefix: str, *parts: str) -> str:
    """基于输入内容生成稳定的组件 key，避免重复 key 报错"""
    joined = "|".join(parts)
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def read_sdf_file(file_path: Path):
    """一次遍历读取 SDF 文件，减少 I/O 开销"""
    data = []
    all_props = set()

    # 单遍历收集行数据及属性集合
    for i, mol in enumerate(Chem.SDMolSupplier(str(file_path))):
        if mol is None:
            continue

        row = {"Index": i + 1, "SMILES": Chem.MolToSmiles(mol), "MolWt": MolWt(mol)}

        for prop in mol.GetPropNames():
            row[prop] = mol.GetProp(prop)
            all_props.add(prop)

        data.append(row)

    # 补齐缺失列，避免后续 KeyError
    for row in data:
        for prop in all_props:
            row.setdefault(prop, None)

    df = pd.DataFrame(data)
    return df, list(df.columns)


def read_csv_file(file_path: Path, parser: str = "pandas", nrows: int | None = None):
    """根据指定解析器读取CSV文件"""
    try:
        if (
            parser == "pyarrow"
            and CSV_PARSER_AVAILABLE.get("pyarrow")
            and pacsv is not None
        ):
            if nrows is not None:
                df = pd.read_csv(file_path, nrows=nrows)
                return df, list(df.columns)

            table = pacsv.read_csv(file_path.as_posix())
            df = table.to_pandas()
            return df, list(df.columns)

        if parser == "polars" and CSV_PARSER_AVAILABLE.get("polars") and pl is not None:
            df_polars = pl.read_csv(
                file_path.as_posix(), n_rows=nrows, try_parse_dates=True
            )
            df = df_polars.to_pandas()
            return df, list(df.columns)

        df = pd.read_csv(file_path, nrows=nrows)
        return df, list(df.columns)
    except Exception as e:
        st.error(f"读取CSV文件时出错 ({parser}): {str(e)}")
        return None, []


def validate_smiles(
    smiles_series: pd.Series, sample_size: int | None = None, progress_callback=None
):
    """验证SMILES列的有效性，可选抽样并支持进度回调"""

    values = smiles_series.dropna()
    total_count = len(values)
    if total_count == 0:
        return 0, 0, 0

    if sample_size and sample_size < total_count:
        values = values.sample(sample_size, random_state=42)

    evaluated_count = len(values)
    valid_count = 0

    for idx, smiles in enumerate(values, start=1):
        if Chem.MolFromSmiles(str(smiles)) is not None:
            valid_count += 1
        if progress_callback and (idx == evaluated_count or idx % 500 == 0):
            progress_callback(idx / evaluated_count)

    return valid_count, total_count, evaluated_count


def calculate_molecular_weight(smiles_series):
    """根据SMILES计算分子量"""
    mol_weights = []
    for smiles in smiles_series:
        if pd.isna(smiles):
            mol_weights.append(None)
        else:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is not None:
                mol_weights.append(MolWt(mol))
            else:
                mol_weights.append(None)
    return mol_weights


def parse_smiles_validity(smiles_series: pd.Series):
    """解析SMILES并返回分子对象列表及有效性掩码"""
    parsed_mols = []
    valid_flags = []

    for smiles in smiles_series:
        if pd.isna(smiles):
            mol = None
        else:
            mol = Chem.MolFromSmiles(str(smiles))
        parsed_mols.append(mol)
        valid_flags.append(mol is not None)

    valid_mask = pd.Series(valid_flags, index=smiles_series.index, dtype=bool)
    return parsed_mols, valid_mask


def create_standardized_output(
    df: pd.DataFrame,
    id_col: str | None,
    smiles_col: str,
    molwt_col: str | None,
    other_cols: list[str],
    output_path: Path,
    drop_invalid_smiles: bool = True,
):
    """创建标准化输出"""
    parsed_mols = None
    valid_mask = None

    # 在需要时仅解析一次SMILES，复用到过滤与分子量计算
    if drop_invalid_smiles or not (molwt_col and molwt_col in df.columns):
        parsed_mols, valid_mask = parse_smiles_validity(df[smiles_col])

    if drop_invalid_smiles:
        if valid_mask is None:
            _, valid_mask = parse_smiles_validity(df[smiles_col])
        working_df = df.loc[valid_mask].copy()
    else:
        working_df = df.copy()

    # 创建新的DataFrame
    output_df = pd.DataFrame()

    # 标准列
    output_df["ID"] = working_df[id_col] if id_col else range(1, len(working_df) + 1)
    output_df["SMILES"] = working_df[smiles_col]

    # 分子量处理
    if molwt_col and molwt_col in df.columns:
        output_df["MolWt"] = working_df[molwt_col]
    else:
        # 根据SMILES计算分子量
        st.info("正在根据SMILES计算分子量...")
        if parsed_mols is None:
            parsed_mols, valid_mask = parse_smiles_validity(df[smiles_col])

        if drop_invalid_smiles and valid_mask is not None:
            filtered_mols = [
                mol
                for mol, is_valid in zip(parsed_mols, valid_mask.tolist())
                if is_valid
            ]
            output_df["MolWt"] = [MolWt(mol) for mol in filtered_mols]
        else:
            output_df["MolWt"] = [
                MolWt(mol) if mol is not None else None for mol in parsed_mols
            ]

    # 其他选择的列
    for col in other_cols:
        if col in working_df.columns:
            output_df[col] = working_df[col]

    # 保存文件
    output_df.to_csv(output_path, index=False)

    invalid_rows = int((~valid_mask).sum()) if valid_mask is not None else None
    dropped_rows = int(len(df) - len(output_df)) if drop_invalid_smiles else 0
    summary = {
        "drop_invalid_smiles": drop_invalid_smiles,
        "invalid_rows": invalid_rows,
        "dropped_rows": dropped_rows,
    }

    return output_df, summary


# --------------------------------------------------
# ⚡️ 缓存辅助工具：必须位于首次调用之前
# --------------------------------------------------


def file_mtime(path: Path) -> float:
    """获取文件最后修改时间（秒级）"""
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data(show_spinner="⏳ 正在读取数据 ...")
def load_data(file_path: Path, file_ext: str, parser: str, mtime: float):
    """加载数据文件并缓存，mtime 变化自动失效"""
    if file_ext == ".sdf":
        return read_sdf_file(file_path)
    elif file_ext == ".csv":
        return read_csv_file(file_path, parser=parser)
    else:
        return None, []


@st.cache_data(show_spinner="🔍 正在加载预览 ...")
def load_preview_data(
    file_path: Path, file_ext: str, parser: str, mtime: float, limit: int
):
    """加载预览数据，默认只读取指定行数"""
    if file_ext == ".csv":
        df, columns = read_csv_file(file_path, parser=parser, nrows=limit)
        return df, columns
    if file_ext == ".sdf":
        data = []
        all_props = set()
        for i, mol in enumerate(Chem.SDMolSupplier(str(file_path))):
            if mol is None:
                continue
            row = {"Index": i + 1, "SMILES": Chem.MolToSmiles(mol), "MolWt": MolWt(mol)}
            for prop in mol.GetPropNames():
                row[prop] = mol.GetProp(prop)
                all_props.add(prop)
            data.append(row)
            if len(data) >= limit:
                break
        for row in data:
            for prop in all_props:
                row.setdefault(prop, None)
        df = pd.DataFrame(data)
        return df, list(df.columns)
    return None, []


def save_uploaded_file(
    uploaded_file, destination: Path, chunk_size: int = CHUNK_SIZE
) -> None:
    """分块保存上传文件，避免一次性占用过多内存"""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    with destination.open("wb") as output:
        for chunk in iter(lambda: uploaded_file.read(chunk_size), b""):
            if not chunk:
                break
            output.write(chunk)

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)


# 主界面
st.markdown("---")

# 创建三列布局
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.header("📂 文件夹管理")

    # 创建新文件夹
    st.subheader("创建新文件夹")
    new_folder_name = st.text_input("文件夹名称", placeholder="输入新文件夹名称")
    if st.button("创建文件夹"):
        if new_folder_name:
            success, message = create_new_folder(new_folder_name)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)

    # 文件夹选择
    st.subheader("选择工作目录")
    folders = list_data_folders()
    if folders:
        selected_folder = st.selectbox("选择文件夹", options=[""] + folders)
    else:
        selected_folder = ""
        st.info("暂无文件夹，请先创建")

with col2:
    st.header("📤 文件上传")

    if selected_folder:
        st.info(f"当前工作目录: {selected_folder}")

        # 文件上传
        uploaded_file = st.file_uploader(
            "上传文件", type=["csv", "sdf"], help="支持CSV和SDF格式文件"
        )

        if uploaded_file is not None:
            folder_path = ensure_data_dir() / selected_folder
            safe_uploaded_name = sanitize_uploaded_filename(uploaded_file.name)

            if not safe_uploaded_name:
                st.error("上传文件名无效，请重命名后重试。")
            else:
                if safe_uploaded_name != uploaded_file.name:
                    st.warning(
                        f"检测到包含路径信息的文件名，已使用安全名称: {safe_uploaded_name}"
                    )

                destination = folder_path / safe_uploaded_name
                if destination.exists():
                    st.warning(f"文件已存在: {safe_uploaded_name}")
                    overwrite_col, cancel_col = st.columns(2)
                    overwrite_key = stable_widget_key(
                        "overwrite_upload", selected_folder, safe_uploaded_name
                    )
                    cancel_key = stable_widget_key(
                        "cancel_upload", selected_folder, safe_uploaded_name
                    )

                    with overwrite_col:
                        if st.button(
                            "覆盖现有文件", key=overwrite_key, use_container_width=True
                        ):
                            save_uploaded_file(uploaded_file, destination)
                            st.success(f"文件已覆盖上传: {safe_uploaded_name}")
                            st.rerun()
                    with cancel_col:
                        if st.button("取消", key=cancel_key, use_container_width=True):
                            st.info("已取消上传操作。")
                else:
                    save_uploaded_file(uploaded_file, destination)
                    st.success(f"文件已上传: {safe_uploaded_name}")
                    st.rerun()
    else:
        st.warning("请先选择工作目录")

with col3:
    st.header("📄 文件选择")

    if selected_folder:
        files = list_files_in_folder(selected_folder)
        if files:
            selected_file = st.selectbox("选择文件", options=[""] + files)
        else:
            selected_file = ""
            st.info("该文件夹中暂无文件")
    else:
        selected_file = ""
        st.info("请先选择工作目录")

# 文件处理部分
if selected_folder and selected_file:
    st.markdown("---")
    st.header("🔧 数据处理")

    file_path = ensure_data_dir() / selected_folder / selected_file
    file_ext = file_path.suffix.lower()
    file_key = f"{selected_folder}/{selected_file}"
    current_mtime = file_mtime(file_path)

    if st.session_state.get("current_file_key") != file_key:
        for cache_key in (
            "preview_df",
            "preview_columns",
            "full_df",
            "full_columns",
            "validation_summary",
            "validation_cache",
        ):
            st.session_state.pop(cache_key, None)
        st.session_state["current_file_key"] = file_key

    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    st.caption(f"文件大小约 {file_size_mb:.1f} MB ｜ 类型: {file_ext.upper()}")

    if file_ext == ".csv":
        available_parsers = [
            name for name, available in CSV_PARSER_AVAILABLE.items() if available
        ]
        default_parser = st.session_state.get("csv_parser_choice", available_parsers[0])
        if default_parser not in available_parsers:
            default_parser = available_parsers[0]

        parser_index = available_parsers.index(default_parser)
        selected_parser = st.selectbox(
            "CSV 解析器",
            options=available_parsers,
            index=parser_index,
            format_func=lambda x: CSV_PARSER_DISPLAY.get(x, x),
            help="PyArrow/Polars 需要已安装相应依赖，未安装时不显示。",
        )
        st.session_state["csv_parser_choice"] = selected_parser
    else:
        selected_parser = "pandas"

    preview_rows = int(
        st.number_input(
            "预览行数",
            min_value=50,
            max_value=5000,
            value=min(PREVIEW_DEFAULT_ROWS, 2000),
            step=50,
            help="用于快速预览和字段配置，大数据集无需增大此数值。",
        )
    )

    action_cols = st.columns(2)
    with action_cols[0]:
        if st.button("加载预览", use_container_width=True):
            with st.spinner("正在加载预览..."):
                preview_df, preview_columns = load_preview_data(
                    file_path, file_ext, selected_parser, current_mtime, preview_rows
                )
            st.session_state["preview_df"] = preview_df
            st.session_state["preview_columns"] = preview_columns

    with action_cols[1]:
        if st.button("加载全量数据", use_container_width=True):
            with st.spinner("正在加载全量数据（可能耗时）..."):
                full_df, full_columns = load_data(
                    file_path, file_ext, selected_parser, current_mtime
                )
            st.session_state["full_df"] = full_df
            st.session_state["full_columns"] = full_columns

    preview_df = st.session_state.get("preview_df")
    preview_columns = st.session_state.get("preview_columns")
    full_df = st.session_state.get("full_df")
    full_columns = st.session_state.get("full_columns")

    available_columns = full_columns or preview_columns
    df_for_display = full_df if full_df is not None else preview_df

    if preview_df is not None:
        with st.expander("文件预览", expanded=True):
            st.dataframe(preview_df.head(10))
            st.info(f"预览数据：{len(preview_df)} 行，{len(preview_df.columns)} 列")
    else:
        st.info("尚未加载预览，请点击“加载预览”查看列信息。")

    if not available_columns or df_for_display is None:
        st.warning("完成字段映射前需先加载预览数据。")
    else:
        st.subheader("🏷️ 字段映射")

        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("**必需字段**")

            id_options = ["自动生成序号"] + available_columns
            id_col_idx = st.selectbox(
                "ID列",
                options=range(len(id_options)),
                format_func=lambda x: id_options[x],
            )
            id_col = available_columns[id_col_idx - 1] if id_col_idx > 0 else None

            smiles_candidates = [
                col
                for col in available_columns
                if "smil" in col.lower() or "structure" in col.lower()
            ]
            if not smiles_candidates:
                smiles_candidates = available_columns

            smiles_default_index = (
                available_columns.index(smiles_candidates[0])
                if smiles_candidates
                else 0
            )
            smiles_col = st.selectbox(
                "SMILES/结构列", options=available_columns, index=smiles_default_index
            )

            molwt_options = ["根据SMILES计算"] + [
                col
                for col in available_columns
                if "molwt" in col.lower()
                or "weight" in col.lower()
                or "mw" in col.lower()
            ]
            molwt_default_index = 1 if len(molwt_options) > 1 else 0
            selected_molwt_option = st.selectbox(
                "分子量列", options=molwt_options, index=molwt_default_index
            )
            molwt_col = (
                selected_molwt_option
                if selected_molwt_option != "根据SMILES计算"
                else None
            )

        with col_right:
            st.markdown("**其他属性列**")
            other_columns = [
                col
                for col in available_columns
                if col not in [id_col, smiles_col, molwt_col]
            ]
            selected_other_cols = st.multiselect(
                "选择要保留的其他属性",
                options=other_columns,
                help="这些列会出现在输出文件中。",
            )

        if "drop_invalid_smiles" not in st.session_state:
            st.session_state["drop_invalid_smiles"] = True

        data_for_validation = full_df if full_df is not None else preview_df

        if smiles_col and data_for_validation is not None:
            with st.expander("数据验证"):
                total_rows = len(data_for_validation)
                st.info("可选择全量或抽样验证，大型数据集建议抽样以缩短等待时间。")

                validation_mode = st.radio(
                    "验证模式",
                    options=("全量验证", "抽样验证"),
                    index=1 if total_rows > 10000 else 0,
                    horizontal=True,
                )

                sample_size = None
                if validation_mode == "抽样验证" and total_rows > 0:
                    default_sample = min(total_rows, 5000)
                    sample_size = int(
                        st.number_input(
                            "抽样条数",
                            min_value=1,
                            max_value=total_rows,
                            value=max(1, default_sample),
                            step=max(1, max(total_rows // 100, 1)),
                        )
                    )

                validation_cache = st.session_state.setdefault("validation_cache", {})
                data_scope = "full" if full_df is not None else "preview"
                validation_key = json.dumps(
                    {
                        "file": file_key,
                        "mtime": current_mtime,
                        "smiles_column": smiles_col,
                        "mode": "full" if validation_mode == "全量验证" else "sample",
                        "sample_size": int(sample_size) if sample_size else None,
                        "scope": data_scope,
                        "rows": int(total_rows),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )

                run_validation = st.button(
                    "开始验证",
                    key=stable_widget_key(
                        "run_validation",
                        file_key,
                        smiles_col,
                        validation_mode,
                        str(sample_size),
                        data_scope,
                    ),
                    use_container_width=True,
                )

                if run_validation:
                    cached_summary = validation_cache.get(validation_key)
                    if cached_summary is not None:
                        st.session_state["validation_summary"] = cached_summary
                        st.success("已加载缓存验证结果。")
                    else:
                        progress_bar = st.progress(0.0)
                        valid_count, total_count, evaluated_count = validate_smiles(
                            data_for_validation[smiles_col],
                            sample_size=sample_size,
                            progress_callback=progress_bar.progress,
                        )
                        progress_bar.empty()

                        summary_payload = {
                            "mode": (
                                "full" if validation_mode == "全量验证" else "sample"
                            ),
                            "valid_count": int(valid_count),
                            "evaluated_count": int(evaluated_count),
                            "total_count": int(total_count),
                            "sample_size": int(sample_size) if sample_size else None,
                            "scope": data_scope,
                            "smiles_column": smiles_col,
                            "cache_key": validation_key,
                        }
                        validation_cache[validation_key] = summary_payload
                        st.session_state["validation_cache"] = validation_cache
                        st.session_state["validation_summary"] = summary_payload

                summary_payload = st.session_state.get("validation_summary")
                if (
                    summary_payload
                    and summary_payload.get("cache_key") == validation_key
                ):
                    valid_count = int(summary_payload["valid_count"])
                    evaluated_count = int(summary_payload["evaluated_count"])
                    total_count = int(summary_payload["total_count"])

                    if evaluated_count == 0:
                        st.warning("未检测到有效的 SMILES 数据")
                    else:
                        ratio = valid_count / evaluated_count * 100
                        label = "有效SMILES"
                        if evaluated_count != total_count:
                            label += f"（抽样 {evaluated_count} 条）"
                        st.metric(
                            label, f"{valid_count}/{evaluated_count}", f"{ratio:.1f}%"
                        )

                        if evaluated_count < total_count:
                            st.info(
                                f"原始数据共 {total_count} 条，已抽样验证 {evaluated_count} 条。"
                            )

                        if valid_count < evaluated_count:
                            invalid_count = evaluated_count - valid_count
                            if st.session_state.get("drop_invalid_smiles", True):
                                st.warning(
                                    f"发现 {invalid_count} 个无效SMILES，当前设置为生成时跳过。"
                                )
                            else:
                                st.warning(
                                    f"发现 {invalid_count} 个无效SMILES，当前设置为保留。"
                                )
                else:
                    st.caption("点击“开始验证”以执行当前配置的 SMILES 校验。")

        st.subheader("💾 输出设置")
        drop_invalid_smiles = st.checkbox(
            "处理时跳过无效SMILES",
            value=st.session_state.get("drop_invalid_smiles", True),
            key="drop_invalid_smiles",
            help="开启后会在生成标准化文件时移除无法解析为分子的行。",
        )

        base_name = Path(selected_file).stem
        output_filename = f"prepared_{base_name}.csv"
        output_path = file_path.parent / output_filename

        st.text(f"输出文件名: {output_filename}")

        if st.button("🚀 生成标准化文件", type="primary"):
            df_for_processing = full_df
            if df_for_processing is None:
                with st.spinner("正在加载全量数据..."):
                    df_for_processing, full_columns = load_data(
                        file_path, file_ext, selected_parser, current_mtime
                    )
                st.session_state["full_df"] = df_for_processing
                st.session_state["full_columns"] = full_columns

            if df_for_processing is None:
                st.error("加载全量数据失败，请检查文件格式或内容。")
            else:
                validation_summary = st.session_state.get("validation_summary")
                try:
                    with st.spinner("正在处理数据..."):
                        output_df, processing_summary = create_standardized_output(
                            df_for_processing,
                            id_col,
                            smiles_col,
                            molwt_col,
                            selected_other_cols,
                            output_path,
                            drop_invalid_smiles=drop_invalid_smiles,
                        )

                    st.success(f"✅ 成功生成标准化文件: {output_filename}")
                    st.info(
                        f"文件保存位置: {output_path.parent.resolve()} / {output_filename}"
                    )
                    if (
                        processing_summary.get("drop_invalid_smiles")
                        and processing_summary.get("dropped_rows", 0) > 0
                    ):
                        st.warning(
                            f"已跳过 {processing_summary['dropped_rows']} 条无效SMILES记录。"
                        )

                    with st.expander("输出预览", expanded=True):
                        st.dataframe(output_df.head(10))
                        st.info(
                            f"输出文件包含 {len(output_df)} 行，{len(output_df.columns)} 列"
                        )

                    metadata = {
                        "generated_at": datetime.now().isoformat(),
                        "source_file": selected_file,
                        "output_file": output_filename,
                        "working_directory": str(output_path.parent.resolve()),
                        "csv_parser": selected_parser if file_ext == ".csv" else None,
                        "id_column": id_col,
                        "smiles_column": smiles_col,
                        "molwt_column": molwt_col,
                        "other_columns": selected_other_cols,
                        "total_rows": int(len(output_df)),
                        "columns": list(output_df.columns),
                        "validation": validation_summary,
                        "output_options": {"drop_invalid_smiles": drop_invalid_smiles},
                        "processing_summary": processing_summary,
                    }

                    metadata_path = output_path.with_suffix(".meta.json")
                    metadata_path.write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    st.info(
                        f"已写入处理元数据: {metadata_path.name} (目录: {metadata_path.parent.resolve()})"
                    )

                    file_size = output_path.stat().st_size
                    if file_size <= DOWNLOAD_INLINE_THRESHOLD:
                        with output_path.open("rb") as buffer:
                            st.download_button(
                                label="📥 下载标准化文件",
                                data=buffer,
                                file_name=output_filename,
                                mime="text/csv",
                            )
                    else:
                        st.info(
                            "输出文件较大（约 {:.1f} MB），请直接在项目的 `data/{}` 目录获取。".format(
                                file_size / (1024 * 1024), selected_folder
                            )
                        )

                except Exception as e:
                    st.error(f"处理数据时出错: {str(e)}")

# 显示当前状态
if selected_folder:
    with st.sidebar:
        st.subheader("📊 当前状态")
        st.text(f"工作目录: {selected_folder}")
        if selected_file:
            st.text(f"选择文件: {selected_file}")

        files = list_files_in_folder(selected_folder)
        st.text(f"文件数量: {len(files)}")

        # 显示文件列表
        if files:
            st.subheader("📁 文件列表")
            folder_path = ensure_data_dir() / selected_folder
            for file_name in files:
                file_path = folder_path / file_name
                file_size_kb = file_path.stat().st_size / 1024
                st.text(f"• {file_name} ({file_size_kb:.1f}KB)")
