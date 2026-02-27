"""
结构多样性评估 - 数据读取与缓存工具
"""

from __future__ import annotations

import gc
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st


DEFAULT_META_SAMPLE_ROWS = 5000
MAX_FPS_CACHE_ITEMS = 2


def list_data_folders(data_dir: str = "data"):
    """列出 data 目录下的所有文件夹"""
    if not os.path.exists(data_dir):
        return []
    return [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]


def list_csv_files_in_folder(folder_name: str, data_dir: str = "data"):
    """列出指定文件夹中的所有 CSV 文件"""
    folder_path = os.path.join(data_dir, folder_name)
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.endswith(".csv")]


def get_file_info(file_path: str):
    """获取文件基本信息"""
    if not os.path.exists(file_path):
        return None

    file_size = os.path.getsize(file_path) / (1024 * 1024)
    mod_time = os.path.getmtime(file_path)
    mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
    return {"size_mb": file_size, "modified": mod_time_str}


def _put_cache_item(cache: dict, key: str, value, max_items: int) -> None:
    """向会话缓存写入条目并控制容量"""
    if key in cache:
        cache.pop(key)
    cache[key] = value
    while len(cache) > max_items:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)


def _build_file_signature(file_path: str) -> str:
    """构建用于缓存的文件签名"""
    try:
        stat_info = os.stat(file_path)
        return f"{os.path.abspath(file_path)}|{stat_info.st_mtime_ns}|{stat_info.st_size}"
    except OSError:
        return os.path.abspath(file_path)


def _build_fps_cache_key(
    csv_path: str,
    chunksize: int,
    fp_dtype: str,
    fingerprint_cols,
    meta_mode: str,
    meta_sample_rows: int,
) -> str:
    cols_sig = "auto" if fingerprint_cols is None else ",".join(map(str, fingerprint_cols))
    file_sig = _build_file_signature(csv_path)
    return "|".join(
        [
            file_sig,
            f"chunksize={chunksize}",
            f"dtype={fp_dtype}",
            f"cols={cols_sig}",
            f"meta_mode={meta_mode}",
            f"meta_rows={meta_sample_rows}",
        ]
    )


def build_array_signature(array: np.ndarray) -> str:
    """构建轻量数组签名，用于会话级计算缓存"""
    if array is None:
        return "none"
    if array.size == 0:
        return f"shape={array.shape}|dtype={array.dtype}|empty"

    rows = min(64, array.shape[0])
    cols = min(64, array.shape[1]) if array.ndim == 2 else 1
    sample = array[:rows] if array.ndim == 1 else array[:rows, :cols]
    sample = np.asarray(sample, dtype=np.float32)

    return (
        f"shape={array.shape}|dtype={array.dtype}|"
        f"sum={float(np.sum(sample)):.6f}|mean={float(np.mean(sample)):.6f}"
    )


def read_fps(
    csv_path: str,
    chunksize: int = 200_000,
    fp_dtype: str = "float16",
    fingerprint_cols=None,
    meta_mode: str = "sample",
    meta_sample_rows: int = DEFAULT_META_SAMPLE_ROWS,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """流式加载指纹列，峰值内存≈常数级"""
    try:
        meta_mode = (meta_mode or "sample").lower()
        if meta_mode not in {"none", "sample", "full"}:
            raise ValueError(f"不支持的 meta_mode: {meta_mode}")

        meta_sample_rows = max(1, int(meta_sample_rows))

        sample = pd.read_csv(csv_path, nrows=100, low_memory=True)

        if fingerprint_cols is None:
            keywords = ["fingerprint", "fp", "descriptor", "feature", "bit", "e3fp", "rocs", "usrcat"]
            num_cols = []

            for col in sample.columns:
                if pd.api.types.is_numeric_dtype(sample[col]):
                    num_cols.append(col)
                elif any(keyword.lower() in col.lower() for keyword in keywords):
                    coerced = pd.to_numeric(sample[col], errors="coerce")
                    if coerced.notna().any():
                        num_cols.append(col)
            fingerprint_cols = num_cols

        if not fingerprint_cols:
            raise ValueError("未检测到数值型指纹列")

        meta_cols = [col for col in sample.columns if col not in fingerprint_cols]
        st.info(f"检测到 {len(fingerprint_cols)} 个指纹列，{len(meta_cols)} 个元数据列")

        fp_parts = []
        total_rows = 0
        actual_dtype = fp_dtype
        used_fallback_dtype = False

        progress_bar = st.progress(0)
        status_text = st.empty()

        chunk_iter = pd.read_csv(
            csv_path,
            usecols=fingerprint_cols,
            chunksize=chunksize,
            low_memory=True,
            engine="c",
        )

        for i, chunk in enumerate(chunk_iter, start=1):
            status_text.text(f"处理第 {i} 个数据块，当前行数: {total_rows:,}")

            try:
                numeric_chunk = chunk.apply(pd.to_numeric, errors="coerce")
                numeric_chunk.replace([np.inf, -np.inf], 0, inplace=True)
                numeric_chunk.fillna(0, inplace=True)

                if i == 1:
                    try:
                        chunk_array = numeric_chunk.to_numpy(dtype=actual_dtype, copy=False)
                        if np.isinf(chunk_array).any():
                            raise OverflowError("检测到inf，可能是低精度溢出")
                    except (ValueError, OverflowError, TypeError):
                        actual_dtype = "float32"
                        used_fallback_dtype = True
                        st.warning(f"检测到数据无法稳定转换为 {fp_dtype}，自动切换为 float32")
                        chunk_array = numeric_chunk.to_numpy(dtype=actual_dtype, copy=False)
                else:
                    chunk_array = numeric_chunk.to_numpy(dtype=actual_dtype, copy=False)

                if np.isnan(chunk_array).any() or np.isinf(chunk_array).any():
                    chunk_array = np.nan_to_num(chunk_array, nan=0.0, posinf=0.0, neginf=0.0)

                fp_parts.append(chunk_array)
                total_rows += len(chunk_array)

            except Exception as e:
                st.warning(f"处理第 {i} 个数据块时出错: {str(e)}，跳过此块")
                continue
            finally:
                del chunk
                if "numeric_chunk" in locals():
                    del numeric_chunk
                if i % 8 == 0:
                    gc.collect()

            progress_bar.progress(min(0.95, i / 20))

        progress_bar.progress(0.95)
        status_text.text("合并数据块...")

        if not fp_parts:
            raise ValueError("未成功读取任何有效数据块")

        fps = fp_parts[0] if len(fp_parts) == 1 else np.concatenate(fp_parts, axis=0)
        del fp_parts
        gc.collect()

        progress_bar.progress(1.0)
        status_text.text("读取元数据...")

        if not meta_cols or meta_mode == "none":
            meta = pd.DataFrame(index=np.arange(fps.shape[0]))
        elif meta_mode == "full":
            meta = pd.read_csv(csv_path, usecols=meta_cols, low_memory=True)
        else:
            sampled_rows = min(meta_sample_rows, fps.shape[0])
            meta = pd.read_csv(csv_path, usecols=meta_cols, nrows=sampled_rows, low_memory=True)
            meta.attrs["sampled"] = True
            meta.attrs["sample_rows"] = sampled_rows
            st.info(f"🧾 元数据按样本加载: {sampled_rows:,}/{fps.shape[0]:,} 行")

        progress_bar.empty()
        status_text.empty()

        st.success(f"✅ 流式加载完成: {len(fps):,} 个样本，{len(fingerprint_cols)} 维指纹 ({actual_dtype})")
        st.info(f"📊 内存使用: {fps.nbytes / 1024**2:.1f} MB")
        if used_fallback_dtype:
            st.info(f"💡 数据类型已从 {fp_dtype} 调整为 {actual_dtype} 以确保兼容性")

        return fps, fingerprint_cols, meta

    except Exception as e:
        st.error(f"流式读取指纹数据时出错: {str(e)}")
        return None, None, None


def read_fps_cached(
    csv_path: str,
    chunksize: int = 200_000,
    fp_dtype: str = "float16",
    fingerprint_cols=None,
    meta_mode: str = "sample",
    meta_sample_rows: int = DEFAULT_META_SAMPLE_ROWS,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """带会话缓存的指纹读取"""
    cache = st.session_state.setdefault("fps_cache", {})
    cache_key = _build_fps_cache_key(
        csv_path=csv_path,
        chunksize=chunksize,
        fp_dtype=fp_dtype,
        fingerprint_cols=fingerprint_cols,
        meta_mode=meta_mode,
        meta_sample_rows=meta_sample_rows,
    )

    if cache_key in cache:
        st.info(f"♻️ 命中读取缓存: {os.path.basename(csv_path)}")
        value = cache.pop(cache_key)
        cache[cache_key] = value
        return value

    result = read_fps(
        csv_path=csv_path,
        chunksize=chunksize,
        fp_dtype=fp_dtype,
        fingerprint_cols=fingerprint_cols,
        meta_mode=meta_mode,
        meta_sample_rows=meta_sample_rows,
    )
    if result[0] is not None:
        _put_cache_item(cache, cache_key, result, MAX_FPS_CACHE_ITEMS)
    return result


def subsample_fingerprints(
    fps: np.ndarray,
    max_samples: int,
    label: str,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray | None]:
    """若样本数超过阈值，则随机下采样"""
    if max_samples <= 0 or len(fps) <= max_samples:
        return fps, None

    rng = np.random.default_rng(random_seed)
    indices = np.sort(rng.choice(len(fps), max_samples, replace=False))
    st.warning(
        f"{label} 含 {len(fps):,} 个样本，已随机抽样 {max_samples:,} 个用于兼容模式计算。"
        " 如需完整矩阵，请降低数据集规模或切换优化模式。"
    )
    return fps[indices], indices


def subsample_by_ratio(
    fps: np.ndarray,
    ratio: float,
    label: str,
    meta: pd.DataFrame | None = None,
    random_seed: int = 42,
) -> tuple[np.ndarray, pd.DataFrame | None, np.ndarray | None]:
    """按照指定比例随机抽样指纹数据。ratio=1.0 表示不采样。"""
    if fps is None or len(fps) == 0 or ratio >= 0.999:
        return fps, meta, None

    rng = np.random.default_rng(random_seed)
    sample_size = max(1, int(len(fps) * ratio))
    sample_size = min(sample_size, len(fps))
    indices = np.sort(rng.choice(len(fps), sample_size, replace=False))
    st.info(f"{label} 已按 {ratio*100:.1f}% 比例抽样 {sample_size:,}/{len(fps):,} 个样本")

    fps_sub = fps[indices]
    if meta is not None and len(meta) == len(fps):
        meta = meta.iloc[indices].reset_index(drop=True)
    return fps_sub, meta, indices


def load_fingerprints_from_csv(
    file_path,
    fingerprint_cols=None,
    fp_dtype: str = "float16",
    chunksize: int = 200_000,
    meta_mode: str = "sample",
    meta_sample_rows: int = DEFAULT_META_SAMPLE_ROWS,
):
    """兼容性函数，内部调用缓存版 read_fps"""
    fps, fp_cols, meta = read_fps_cached(
        file_path,
        chunksize=chunksize,
        fp_dtype=fp_dtype,
        fingerprint_cols=fingerprint_cols,
        meta_mode=meta_mode,
        meta_sample_rows=meta_sample_rows,
    )
    if fps is not None:
        return fps, meta, fp_cols
    return None, None, None


def ensure_faiss_compatible(arr: np.ndarray) -> np.ndarray:
    """确保数组与FAISS兼容（C-contiguous + float32）"""
    if not arr.flags["C_CONTIGUOUS"] or arr.dtype != np.float32:
        return np.ascontiguousarray(arr.astype(np.float32))
    return arr
