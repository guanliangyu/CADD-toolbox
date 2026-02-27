"""
Tanimoto 相似性并行计算工具（CPU）。
"""

from __future__ import annotations

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import shared_memory
from typing import Callable

import numpy as np
import psutil

_TANIMOTO_SHARED_FPS = None
_TANIMOTO_SHARED_MEM = None


def get_max_available_cpus() -> int:
    """获取当前进程可用的最大 CPU 数量"""
    try:
        process = psutil.Process()
        if hasattr(process, "cpu_affinity"):
            affinity = process.cpu_affinity()
            if affinity:
                return max(1, len(affinity))
    except Exception:
        pass

    cpu_count = psutil.cpu_count(logical=True) or os.cpu_count() or 1
    return max(1, int(cpu_count))


def _init_tanimoto_worker(
    shm_name: str, shape: tuple[int, int], dtype_str: str
) -> None:
    """初始化 Tanimoto worker 的共享内存视图"""
    global _TANIMOTO_SHARED_FPS, _TANIMOTO_SHARED_MEM
    _TANIMOTO_SHARED_MEM = shared_memory.SharedMemory(name=shm_name)
    _TANIMOTO_SHARED_FPS = np.ndarray(
        shape, dtype=np.dtype(dtype_str), buffer=_TANIMOTO_SHARED_MEM.buf
    )


def _compute_tanimoto_chunk(
    fps: np.ndarray, row_start: int, row_end: int
) -> tuple[int, int, np.ndarray]:
    """计算 [row_start, row_end) 行对应的上三角 Tanimoto 相似性"""
    n_samples = fps.shape[0]
    chunk = np.zeros((row_end - row_start, n_samples), dtype=np.float32)

    for local_idx, row_idx in enumerate(range(row_start, row_end)):
        row = fps[row_idx]
        ref = fps[row_idx:]
        intersection = np.minimum(ref, row).sum(axis=1, dtype=np.float32)
        union = np.maximum(ref, row).sum(axis=1, dtype=np.float32)
        similarities = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 1e-12,
        )
        chunk[local_idx, row_idx:] = similarities.astype(np.float32, copy=False)

    return row_start, row_end, chunk


def _tanimoto_chunk_worker(row_start: int, row_end: int) -> tuple[int, int, np.ndarray]:
    """进程池 worker 入口"""
    global _TANIMOTO_SHARED_FPS
    return _compute_tanimoto_chunk(_TANIMOTO_SHARED_FPS, row_start, row_end)


def _select_multiprocessing_context():
    """优先使用 fork（Linux 下启动开销更低）"""
    try:
        available = mp.get_all_start_methods()
        if "fork" in available:
            return mp.get_context("fork")
    except Exception:
        pass
    return mp.get_context("spawn")


def compute_tanimoto_similarity_matrix(
    fingerprints: np.ndarray,
    n_jobs: int = 1,
    chunk_rows: int = 128,
    progress_callback: Callable[[float], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> np.ndarray:
    """计算完整 Tanimoto 相似性矩阵（支持多 CPU 并行）"""
    fps = np.ascontiguousarray(fingerprints.astype(np.float32, copy=False))
    n_samples = fps.shape[0]

    if n_samples == 0:
        return np.empty((0, 0), dtype=np.float32)

    n_jobs = max(1, int(n_jobs))
    n_jobs = min(n_jobs, get_max_available_cpus())
    chunk_rows = max(16, int(chunk_rows))

    row_ranges = [
        (start, min(start + chunk_rows, n_samples))
        for start in range(0, n_samples, chunk_rows)
    ]
    n_tasks = len(row_ranges)
    similarity_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)

    def _update_progress(done_tasks: int) -> None:
        if progress_callback is not None:
            progress_callback(done_tasks / max(1, n_tasks))

    if n_jobs == 1 or n_tasks == 1:
        for idx, (row_start, row_end) in enumerate(row_ranges, start=1):
            if status_callback is not None:
                status_callback(
                    f"Tanimoto单进程计算: {row_start:,}-{row_end:,}/{n_samples:,}"
                )
            _, _, chunk = _compute_tanimoto_chunk(fps, row_start, row_end)
            similarity_matrix[row_start:row_end, :] = chunk
            _update_progress(idx)
    else:
        shm = None
        try:
            shm = shared_memory.SharedMemory(create=True, size=fps.nbytes)
            shm_view = np.ndarray(fps.shape, dtype=fps.dtype, buffer=shm.buf)
            shm_view[:] = fps

            mp_ctx = _select_multiprocessing_context()
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                mp_context=mp_ctx,
                initializer=_init_tanimoto_worker,
                initargs=(shm.name, fps.shape, fps.dtype.str),
            ) as executor:
                futures = [
                    executor.submit(_tanimoto_chunk_worker, row_start, row_end)
                    for row_start, row_end in row_ranges
                ]
                for idx, future in enumerate(as_completed(futures), start=1):
                    row_start, row_end, chunk = future.result()
                    similarity_matrix[row_start:row_end, :] = chunk
                    if status_callback is not None:
                        status_callback(f"Tanimoto多进程计算进度: {idx}/{n_tasks} 块")
                    _update_progress(idx)
        except Exception as e:
            if status_callback is not None:
                status_callback(f"Tanimoto多进程失败，改用多线程: {e}")
            try:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [
                        executor.submit(
                            _compute_tanimoto_chunk, fps, row_start, row_end
                        )
                        for row_start, row_end in row_ranges
                    ]
                    for idx, future in enumerate(as_completed(futures), start=1):
                        row_start, row_end, chunk = future.result()
                        similarity_matrix[row_start:row_end, :] = chunk
                        if status_callback is not None:
                            status_callback(
                                f"Tanimoto多线程计算进度: {idx}/{n_tasks} 块"
                            )
                        _update_progress(idx)
            except Exception as e2:
                if status_callback is not None:
                    status_callback(f"Tanimoto多CPU失败，回退单进程: {e2}")
                for idx, (row_start, row_end) in enumerate(row_ranges, start=1):
                    _, _, chunk = _compute_tanimoto_chunk(fps, row_start, row_end)
                    similarity_matrix[row_start:row_end, :] = chunk
                    _update_progress(idx)
        finally:
            if shm is not None:
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass

    upper = np.triu(similarity_matrix)
    return upper + upper.T - np.diag(np.diag(upper))
