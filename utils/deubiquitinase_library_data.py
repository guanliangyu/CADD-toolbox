"""
Deubiquitinase Focused Library - CSV 数据读取工具
"""

from __future__ import annotations

from collections import Counter
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd


DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "gb18030", "gbk")
DEFAULT_SEPARATORS: tuple[str, ...] = (",", "\t", ";", "|")
PROBE_ROWS = 200
FOLDER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._\-\u4e00-\u9fff]+$")


@dataclass(frozen=True)
class CsvLoadMeta:
    """CSV 读取元信息"""

    source_name: str
    encoding: str
    separator: str
    n_rows: int
    n_cols: int


def ensure_data_dir(data_dir: str | Path = "data") -> Path:
    """确保 data 目录存在"""
    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_data_folders(data_dir: str | Path = "data") -> list[str]:
    """列出 data 目录下的所有子文件夹"""
    base_dir = Path(data_dir)
    if not base_dir.exists():
        return []
    return sorted([item.name for item in base_dir.iterdir() if item.is_dir()])


def list_csv_files_in_folder(folder_name: str, data_dir: str | Path = "data") -> list[str]:
    """列出指定文件夹中的 CSV 文件"""
    if not folder_name:
        return []
    folder_path = Path(data_dir) / folder_name
    if not folder_path.exists() or not folder_path.is_dir():
        return []
    return sorted([item.name for item in folder_path.iterdir() if item.is_file() and item.suffix.lower() == ".csv"])


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


def create_data_subfolder(folder_name: str, data_dir: str | Path = "data") -> Path:
    """在 data 目录下创建子文件夹（已存在则直接返回）"""
    is_valid, normalized_or_message = validate_folder_name(folder_name)
    if not is_valid:
        raise ValueError(normalized_or_message)

    base_dir = ensure_data_dir(data_dir)
    folder_path = base_dir / normalized_or_message
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def sanitize_uploaded_filename(file_name: str | None) -> str:
    """清理上传文件名，移除路径并确保 .csv 后缀"""
    base_name = Path(file_name or "").name.strip()
    if not base_name:
        base_name = "uploaded.csv"

    if not base_name.lower().endswith(".csv"):
        base_name = f"{base_name}.csv"
    return base_name


def _build_unique_file_path(target_path: Path) -> Path:
    """为同名文件生成不冲突的新文件名"""
    stem = target_path.stem
    suffix = target_path.suffix
    counter = 1
    candidate = target_path
    while candidate.exists():
        candidate = target_path.with_name(f"{stem}_{counter}{suffix}")
        counter += 1
    return candidate


def save_uploaded_csv_to_data(
    uploaded_file,
    folder_name: str,
    data_dir: str | Path = "data",
    filename: str | None = None,
    overwrite: bool = False,
) -> Path:
    """将上传 CSV 保存到 data/<folder_name>/ 下"""
    if uploaded_file is None:
        raise ValueError("未提供上传文件")

    folder_path = create_data_subfolder(folder_name, data_dir=data_dir)
    file_name = sanitize_uploaded_filename(filename or getattr(uploaded_file, "name", "uploaded.csv"))
    target_path = folder_path / file_name
    if target_path.exists() and not overwrite:
        target_path = _build_unique_file_path(target_path)

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    if not file_bytes:
        raise ValueError("上传文件为空，无法保存")

    target_path.write_bytes(file_bytes)
    return target_path


def _detect_separator(sample_text: str) -> str:
    """
    基于表头行检测 CSV 分隔符

    优先使用首个非空行中出现次数最多的候选分隔符，
    避免复杂内容下 Sniffer 误判导致慢解析。
    """
    header_line = ""
    for line in sample_text.splitlines():
        stripped = line.strip()
        if stripped:
            header_line = stripped
            break

    if not header_line:
        return ","

    counts = {sep: header_line.count(sep) for sep in DEFAULT_SEPARATORS}
    best_sep, best_count = max(counts.items(), key=lambda item: item[1])
    if best_count > 0:
        return best_sep
    return ","


def _iter_separators(detected_separator: str) -> list[str]:
    """生成分隔符尝试顺序（优先尝试检测结果）"""
    ordered = [detected_separator]
    ordered.extend([sep for sep in DEFAULT_SEPARATORS if sep != detected_separator])
    return ordered


def _looks_like_wrong_separator(df: pd.DataFrame, sample_text: str, separator: str) -> bool:
    """
    判断分隔符是否疑似错误

    经验规则：
    - 解析后只有 1 列；
    - 但表头行中明显包含该分隔符；
    此时通常是分隔符或编码导致的误解析。
    """
    if df.shape[1] != 1:
        return False

    header_line = ""
    for line in sample_text.splitlines():
        stripped = line.strip()
        if stripped:
            header_line = stripped
            break

    if not header_line:
        return False
    return separator in header_line


def _read_csv_bytes(
    csv_bytes: bytes,
    encoding: str,
    separator: str,
    nrows: int | None = None,
    force_python_engine: bool = False,
) -> pd.DataFrame:
    """按给定编码和分隔符读取 CSV 字节内容"""
    csv_stream = BytesIO(csv_bytes)
    read_kwargs = {
        "encoding": encoding,
        "sep": separator,
        "nrows": nrows,
        "low_memory": True,
    }

    if force_python_engine:
        return pd.read_csv(csv_stream, engine="python", **read_kwargs)

    try:
        return pd.read_csv(csv_stream, **read_kwargs)
    except pd.errors.ParserError:
        csv_stream.seek(0)
        return pd.read_csv(csv_stream, engine="python", **read_kwargs)


def _read_csv_path(
    file_path: Path,
    encoding: str,
    separator: str,
    nrows: int | None = None,
    force_python_engine: bool = False,
) -> pd.DataFrame:
    """按给定编码和分隔符读取 CSV 文件路径"""
    read_kwargs = {
        "encoding": encoding,
        "sep": separator,
        "nrows": nrows,
        "low_memory": True,
    }

    if force_python_engine:
        return pd.read_csv(file_path, engine="python", **read_kwargs)

    try:
        return pd.read_csv(file_path, **read_kwargs)
    except pd.errors.ParserError:
        return pd.read_csv(file_path, engine="python", **read_kwargs)


def load_csv_from_bytes(
    csv_bytes: bytes,
    source_name: str = "uploaded.csv",
    nrows: int | None = None,
) -> tuple[pd.DataFrame, CsvLoadMeta]:
    """从字节流读取 CSV，并自动探测编码与分隔符"""
    if not csv_bytes:
        raise ValueError("CSV 文件内容为空，无法读取")

    last_error: Exception | None = None
    sample_bytes = csv_bytes[:8192]

    for encoding in DEFAULT_ENCODINGS:
        try:
            sample_text = sample_bytes.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

        detected_separator = _detect_separator(sample_text)
        for separator in _iter_separators(detected_separator):
            try:
                probe_nrows = PROBE_ROWS if nrows is None else min(PROBE_ROWS, nrows)
                probe_df = _read_csv_bytes(
                    csv_bytes=csv_bytes,
                    encoding=encoding,
                    separator=separator,
                    nrows=probe_nrows,
                    force_python_engine=True,
                )

                if _looks_like_wrong_separator(probe_df, sample_text, separator):
                    raise ValueError(f"分隔符 `{separator}` 解析结果疑似错误")

                if nrows is None or nrows > probe_nrows:
                    df = _read_csv_bytes(
                        csv_bytes=csv_bytes,
                        encoding=encoding,
                        separator=separator,
                        nrows=nrows,
                    )
                else:
                    df = probe_df

                meta = CsvLoadMeta(
                    source_name=source_name,
                    encoding=encoding,
                    separator=separator,
                    n_rows=len(df),
                    n_cols=df.shape[1],
                )
                return df, meta
            except Exception as exc:
                last_error = exc
                continue

    raise ValueError(f"无法解析 CSV 文件，请检查编码或分隔符。原始错误: {last_error}") from last_error


def load_uploaded_csv(uploaded_file, nrows: int | None = None) -> tuple[pd.DataFrame, CsvLoadMeta]:
    """读取 Streamlit 上传的 CSV 文件对象"""
    if uploaded_file is None:
        raise ValueError("未提供上传文件")

    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    csv_bytes = uploaded_file.read()
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)

    source_name = getattr(uploaded_file, "name", "uploaded.csv")
    return load_csv_from_bytes(csv_bytes=csv_bytes, source_name=source_name, nrows=nrows)


def load_csv_from_path(file_path: str | Path, nrows: int | None = None) -> tuple[pd.DataFrame, CsvLoadMeta]:
    """从本地路径读取 CSV 文件"""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    with path.open("rb") as f:
        sample_bytes = f.read(8192)
    if not sample_bytes:
        raise ValueError("CSV 文件内容为空，无法读取")

    last_error: Exception | None = None
    for encoding in DEFAULT_ENCODINGS:
        try:
            sample_text = sample_bytes.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

        detected_separator = _detect_separator(sample_text)
        for separator in _iter_separators(detected_separator):
            try:
                probe_nrows = PROBE_ROWS if nrows is None else min(PROBE_ROWS, nrows)
                probe_df = _read_csv_path(
                    file_path=path,
                    encoding=encoding,
                    separator=separator,
                    nrows=probe_nrows,
                    force_python_engine=True,
                )

                if _looks_like_wrong_separator(probe_df, sample_text, separator):
                    raise ValueError(f"分隔符 `{separator}` 解析结果疑似错误")

                if nrows is None or nrows > probe_nrows:
                    df = _read_csv_path(
                        file_path=path,
                        encoding=encoding,
                        separator=separator,
                        nrows=nrows,
                    )
                else:
                    df = probe_df

                meta = CsvLoadMeta(
                    source_name=path.name,
                    encoding=encoding,
                    separator=separator,
                    n_rows=len(df),
                    n_cols=df.shape[1],
                )
                return df, meta
            except Exception as exc:
                last_error = exc
                continue

    raise ValueError(f"无法解析 CSV 文件，请检查编码或分隔符。原始错误: {last_error}") from last_error


def _strip_outer_quotes(text: str) -> str:
    """去除字符串两端成对引号（支持多层包裹）"""
    cleaned = text.strip()
    while len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _strip_loose_edge_quotes(text: str) -> str:
    """
    去除边界残留引号

    用于处理不规范数据中的 `""USP36` / `USP40""` 等情况。
    仅清理首尾连续引号，不改动中间内容。
    """
    cleaned = text.strip()
    while cleaned.startswith(("'", '"')):
        cleaned = cleaned[1:].strip()
    while cleaned.endswith(("'", '"')):
        cleaned = cleaned[:-1].strip()
    return cleaned


def split_multi_value_field(value, delimiter: str = ";") -> list[str]:
    """
    解析单元格中的多值字段

    规则：
    - 以 delimiter 分隔；
    - 去除外围引号和空白；
    - 丢弃空片段与 'nan' 文本。
    """
    if delimiter is None or delimiter == "":
        raise ValueError("delimiter 不能为空")
    if pd.isna(value):
        return []

    text = _strip_loose_edge_quotes(_strip_outer_quotes(str(value)))
    if not text:
        return []
    if text.lower() == "nan":
        return []

    parts = text.split(delimiter)
    tokens: list[str] = []
    for part in parts:
        token = _strip_loose_edge_quotes(_strip_outer_quotes(part)).strip()
        if not token:
            continue
        if token.lower() == "nan":
            continue
        tokens.append(token)
    return tokens


def extract_first_word(text: str) -> str | None:
    """提取字符串中的第一个单词（按空白字符分隔）"""
    cleaned = _strip_loose_edge_quotes(_strip_outer_quotes(str(text))).strip()
    if not cleaned:
        return None

    parts = re.split(r"\s+", cleaned, maxsplit=1)
    if not parts or not parts[0]:
        return None
    return parts[0]


def analyze_column_field_options(
    series: pd.Series,
    delimiter: str = ";",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    统计某列中可用字段信息

    返回：
    - option_df: 列 `字段值`、`出现行数`；
    - stats: 总行数、非空行数、多值行数、可用字段种类数。
    """
    total_rows = len(series)
    non_empty_rows = 0
    multi_value_rows = 0
    option_counter: Counter[str] = Counter()

    for value in series:
        tokens = split_multi_value_field(value, delimiter=delimiter)
        if not tokens:
            continue
        non_empty_rows += 1
        if len(tokens) > 1:
            multi_value_rows += 1

        # 统计“出现行数”：同一行重复值仅计一次
        for token in set(tokens):
            option_counter[token] += 1

    if option_counter:
        option_df = pd.DataFrame(
            {
                "字段值": list(option_counter.keys()),
                "出现行数": list(option_counter.values()),
            }
        ).sort_values(["出现行数", "字段值"], ascending=[False, True], ignore_index=True)
    else:
        option_df = pd.DataFrame(columns=["字段值", "出现行数"])

    stats = {
        "total_rows": int(total_rows),
        "non_empty_rows": int(non_empty_rows),
        "multi_value_rows": int(multi_value_rows),
        "unique_options": int(len(option_counter)),
    }
    return option_df, stats


def standardize_multi_value_series(
    series: pd.Series,
    mode: str = "first",
    selected_value: str | None = None,
    delimiter: str = ";",
) -> pd.Series:
    """
    将多值字段标准化为单值列

    mode:
    - first: 每行取第一个字段；
    - first_word: 每行取第一个字段的第一个单词；
    - selected: 仅当该行包含 selected_value 时输出 selected_value，否则置空。
    """
    normalized_values: list[str | pd.NA] = []

    for value in series:
        tokens = split_multi_value_field(value, delimiter=delimiter)
        if not tokens:
            normalized_values.append(pd.NA)
            continue

        if mode == "first":
            normalized_values.append(tokens[0])
        elif mode == "first_word":
            first_word = extract_first_word(tokens[0])
            normalized_values.append(first_word if first_word else pd.NA)
        elif mode == "selected":
            if selected_value is None or selected_value == "":
                raise ValueError("mode='selected' 时必须提供 selected_value")
            normalized_values.append(selected_value if selected_value in set(tokens) else pd.NA)
        else:
            raise ValueError(f"不支持的标准化模式: {mode}")

    return pd.Series(normalized_values, index=series.index, dtype="object")
