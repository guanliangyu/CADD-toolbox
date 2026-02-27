"""
结构多样性评估 - 相似性与多样性计算工具
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from sklearn.neighbors import NearestNeighbors

from utils.structure_diversity_data import build_array_signature, ensure_faiss_compatible
from utils.tanimoto_parallel import compute_tanimoto_similarity_matrix

try:
    import faiss

    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False

try:

    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False

try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False

MAX_METRIC_CACHE_ITEMS = 6


def _put_cache_item(cache: dict, key: str, value, max_items: int) -> None:
    """向会话缓存写入条目并控制容量"""
    if key in cache:
        cache.pop(key)
    cache[key] = value
    while len(cache) > max_items:
        oldest_key = next(iter(cache))
        cache.pop(oldest_key, None)


def _build_metric_cache_key(prefix: str, dataset_sig: str, **params) -> str:
    """构建指标缓存键"""
    param_sig = "|".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{prefix}|{dataset_sig}|{param_sig}"


def cached_knn_similarity(
    fps: np.ndarray,
    metric: str = "cosine",
    k: int = 30,
    use_gpu: bool = True,
    dataset_sig: str | None = None,
) -> np.ndarray:
    """带会话缓存的 k-NN 相似度计算"""
    cache = st.session_state.setdefault("metric_cache", {})
    sig = dataset_sig or build_array_signature(fps)
    key = _build_metric_cache_key("knn", sig, metric=metric, k=k, use_gpu=int(use_gpu))

    if key in cache:
        st.info("♻️ 命中缓存: k-NN 相似性")
        value = cache.pop(key)
        cache[key] = value
        return value

    knn = knn_similarity(fps, metric=metric, k=k, use_gpu=use_gpu)
    if knn is not None:
        _put_cache_item(cache, key, knn, MAX_METRIC_CACHE_ITEMS)
    return knn


def cached_pairwise_similarity(
    fps: np.ndarray,
    n_pairs: int = 2_000_000,
    metric: str = "cosine",
    seed: int = 42,
    dataset_sig: str | None = None,
) -> np.ndarray:
    """带会话缓存的随机采样成对相似性计算"""
    cache = st.session_state.setdefault("metric_cache", {})
    sig = dataset_sig or build_array_signature(fps)
    key = _build_metric_cache_key("pair", sig, metric=metric, n_pairs=n_pairs, seed=seed)

    if key in cache:
        st.info("♻️ 命中缓存: 成对采样相似性")
        value = cache.pop(key)
        cache[key] = value
        return value

    rng = np.random.default_rng(seed)
    pair_sim = sample_pairwise(fps, n_pairs=n_pairs, metric=metric, rng=rng)
    _put_cache_item(cache, key, pair_sim, MAX_METRIC_CACHE_ITEMS)
    return pair_sim


def _recommend_gpu_block_rows(n_samples: int, metric: str = "cosine") -> int:
    """根据可用显存估算GPU分块行数"""
    base_default = 1024 if n_samples <= 10_000 else 512
    try:
        import cupy as cp  # local import for safety

        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        per_pair_bytes = 8 if metric == "cosine" else 16
        estimated_rows = int((free_bytes * 0.25) / (max(1, n_samples) * per_pair_bytes))
        return int(np.clip(estimated_rows, 128, 4096))
    except Exception:
        return base_default


def _compute_similarity_matrix_gpu_blockwise(
    fingerprints: np.ndarray,
    metric: str,
    progress_bar=None,
    status_text=None,
) -> np.ndarray:
    """使用GPU分块计算完整相似性矩阵（cosine/euclidean）"""
    import cupy as cp

    fps = np.ascontiguousarray(fingerprints.astype(np.float32, copy=False))
    n_samples = fps.shape[0]
    block_rows = _recommend_gpu_block_rows(n_samples, metric=metric)
    fps_gpu = cp.asarray(fps)

    try:
        if metric == "cosine":
            norms = cp.linalg.norm(fps_gpu, axis=1, keepdims=True)
            norms = cp.maximum(norms, 1e-12)
            fps_gpu = fps_gpu / norms

            sim_matrix = np.empty((n_samples, n_samples), dtype=np.float32)
            for start in range(0, n_samples, block_rows):
                end = min(start + block_rows, n_samples)
                if status_text is not None:
                    status_text.text(f"使用GPU分块计算余弦相似度: {start:,}-{end:,}/{n_samples:,}")
                block = fps_gpu[start:end]
                sim_block = block @ fps_gpu.T
                sim_matrix[start:end] = cp.asnumpy(sim_block)
                if progress_bar is not None:
                    progress_bar.progress(min(0.95, end / max(1, n_samples)))
            return sim_matrix

        if metric == "euclidean":
            sq_norms = cp.sum(fps_gpu * fps_gpu, axis=1)
            dist_matrix = np.empty((n_samples, n_samples), dtype=np.float32)

            for start in range(0, n_samples, block_rows):
                end = min(start + block_rows, n_samples)
                if status_text is not None:
                    status_text.text(f"使用GPU分块计算欧氏距离: {start:,}-{end:,}/{n_samples:,}")
                block = fps_gpu[start:end]
                gram = block @ fps_gpu.T
                block_sq = sq_norms[start:end][:, None]
                dist_sq = cp.maximum(block_sq + sq_norms[None, :] - 2 * gram, 0.0)
                dist_block = cp.sqrt(dist_sq)
                dist_matrix[start:end] = cp.asnumpy(dist_block)
                if progress_bar is not None:
                    progress_bar.progress(min(0.95, end / max(1, n_samples)))

            max_dist = float(np.max(dist_matrix))
            if max_dist <= 1e-12:
                return np.ones_like(dist_matrix, dtype=np.float32)
            return 1 - (dist_matrix / max_dist)

        raise ValueError(f"GPU路径不支持的相似性度量: {metric}")
    finally:
        try:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except Exception:
            pass


def knn_similarity(
    fps: np.ndarray,
    metric: str = "cosine",
    k: int = 30,
    use_gpu: bool = True,
) -> np.ndarray:
    """使用 FAISS 或 sklearn 计算 k 最近邻相似度（不含自身）"""
    try:
        if not FAISS_AVAILABLE:
            st.warning("FAISS不可用，使用sklearn.NearestNeighbors计算k-NN（CPU）")

            n_samples = len(fps)
            if n_samples == 0:
                return np.empty((0, 0), dtype=np.float32)
            if n_samples == 1:
                return np.empty((1, 0), dtype=np.float32)

            k_eff = min(int(k), n_samples - 1)
            nn = NearestNeighbors(
                n_neighbors=k_eff + 1,
                metric="cosine" if metric == "cosine" else "euclidean",
                algorithm="brute" if metric == "cosine" else "auto",
                n_jobs=-1,
            )
            nn.fit(fps)
            distances, _ = nn.kneighbors(fps, return_distance=True)
            distances = distances[:, 1:]

            if metric == "cosine":
                knn_sim = 1.0 - distances
            else:
                max_dist = float(np.max(distances))
                if max_dist <= 1e-12:
                    knn_sim = np.ones_like(distances, dtype=np.float32)
                else:
                    knn_sim = 1.0 - (distances / max_dist)

            return knn_sim.astype(np.float32, copy=False)

        fps_copy = ensure_faiss_compatible(fps)

        if metric == "cosine":
            faiss.normalize_L2(fps_copy)
            if use_gpu and faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatIP(fps_copy.shape[1]))
            else:
                index = faiss.IndexFlatIP(fps_copy.shape[1])
        else:
            if use_gpu and faiss.get_num_gpus() > 0:
                index = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(fps_copy.shape[1]))
            else:
                index = faiss.IndexFlatL2(fps_copy.shape[1])

        index.add(fps_copy)
        sim, _ = index.search(fps_copy, k + 1)

        if metric == "euclidean":
            max_dist = sim.max()
            if max_dist <= 1e-12:
                sim = np.ones_like(sim, dtype=np.float32)
            else:
                sim = 1 - (sim / max_dist)

        return sim[:, 1:]

    except Exception as e:
        st.error(f"k-NN计算出错: {str(e)}")
        return None


def sample_pairwise(
    fps: np.ndarray,
    n_pairs: int = 2_000_000,
    metric: str = "cosine",
    rng=None,
    random_seed: int = 42,
) -> np.ndarray:
    """随机采样计算成对相似性"""
    if rng is None:
        rng = np.random.default_rng(random_seed)

    n_samples = len(fps)
    max_pairs = n_samples * (n_samples - 1) // 2
    n_pairs = min(n_pairs, max_pairs)

    idx1 = rng.integers(0, n_samples, n_pairs, dtype=np.int64)
    idx2 = rng.integers(0, n_samples, n_pairs, dtype=np.int64)

    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]

    if len(idx1) == 0:
        return np.array([])

    fps1, fps2 = fps[idx1], fps[idx2]

    if metric == "cosine":
        norms1 = np.linalg.norm(fps1, axis=1)
        norms2 = np.linalg.norm(fps2, axis=1)
        dot_products = (fps1 * fps2).sum(axis=1)

        norm_products = norms1 * norms2
        valid_mask = norm_products > 1e-10
        similarities = np.zeros(len(fps1))
        similarities[valid_mask] = dot_products[valid_mask] / norm_products[valid_mask]
        return similarities

    distances = np.linalg.norm(fps1 - fps2, axis=1)
    if len(distances) == 0:
        return np.array([])
    max_dist = distances.max()
    if max_dist <= 1e-12:
        return np.ones_like(distances, dtype=np.float32)
    return 1 - (distances / max_dist)


def compute_similarity_matrix_from_fingerprints(
    fingerprints,
    metric: str = "cosine",
    confirm_key_suffix: str = "default",
    force_device: str | None = None,
    tanimoto_n_jobs: int = 1,
    tanimoto_chunk_rows: int = 128,
):
    """针对小数据集计算完整相似性矩阵"""
    n_samples = len(fingerprints)

    if n_samples > 50_000:
        st.warning(f"⚠️ 数据集较大({n_samples:,}个样本)，建议使用k-NN + 采样方法")
        continue_key = f"continue_full_matrix_{confirm_key_suffix}_{n_samples}"
        if not st.button("继续使用完整矩阵计算", key=continue_key):
            return None

    from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text(f"计算 {n_samples}x{n_samples} 相似性矩阵...")

        use_gpu = bool(CUDA_AVAILABLE and CUPY_AVAILABLE)
        if force_device == "cpu":
            use_gpu = False
        elif force_device == "gpu" and not use_gpu:
            st.warning("兼容模式：强制GPU失败，回退CPU计算完整相似性矩阵")

        if metric in {"cosine", "euclidean"}:
            similarity_matrix = None
            if use_gpu:
                try:
                    status_text.text(f"使用GPU计算 {n_samples}x{n_samples} 相似性矩阵...")
                    similarity_matrix = _compute_similarity_matrix_gpu_blockwise(
                        fingerprints,
                        metric=metric,
                        progress_bar=progress_bar,
                        status_text=status_text,
                    )
                    st.info("🚀 已使用GPU分块加速完整相似性矩阵计算")
                except Exception as e:
                    st.warning(f"GPU相似性矩阵计算失败，回退CPU: {str(e)}")

            if similarity_matrix is None:
                if metric == "cosine":
                    similarity_matrix = cosine_similarity(fingerprints)
                else:
                    distances = euclidean_distances(fingerprints)
                    max_dist = np.max(distances)
                    if max_dist <= 1e-12:
                        similarity_matrix = np.ones_like(distances, dtype=np.float32)
                    else:
                        similarity_matrix = 1 - (distances / max_dist)

        elif metric == "tanimoto":
            if force_device == "gpu":
                st.info("Tanimoto 当前使用CPU多进程实现，暂不支持GPU。")
            status_text.text(
                f"计算Tanimoto相似性矩阵 (CPU={tanimoto_n_jobs}, chunk={tanimoto_chunk_rows})..."
            )
            similarity_matrix = compute_tanimoto_similarity_matrix(
                fingerprints,
                n_jobs=tanimoto_n_jobs,
                chunk_rows=tanimoto_chunk_rows,
                progress_callback=lambda p: progress_bar.progress(min(0.95, p)),
                status_callback=status_text.text,
            )
        else:
            raise ValueError(f"不支持的相似性度量: {metric}")

        progress_bar.empty()
        status_text.empty()
        return similarity_matrix

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"计算相似性矩阵时出错: {str(e)}")
        return None


def diversity_stats(knn_sim: np.ndarray, pair_sim: np.ndarray) -> dict[str, float]:
    """基于k-NN和采样相似性计算多样性指标"""
    if knn_sim is None or len(knn_sim) == 0:
        return {}

    nn_max = knn_sim.max(axis=1)
    nn_mean = knn_sim.mean(axis=1)

    stats = {
        "NN_Mean": nn_mean.mean(),
        "NN_Median": np.median(nn_mean),
        "NN_Std": nn_mean.std(),
        "NN_Min": nn_max.min(),
        "NN_Max": nn_max.max(),
        "NN_Q25": np.percentile(nn_max, 25),
        "NN_Q75": np.percentile(nn_max, 75),
    }

    if pair_sim is not None and len(pair_sim) > 0:
        valid_pairs = pair_sim[~np.isnan(pair_sim)]

        if len(valid_pairs) > 0:
            stats.update(
                {
                    "Pair_Mean": valid_pairs.mean(),
                    "Pair_Median": np.median(valid_pairs),
                    "Pair_Std": valid_pairs.std(),
                    "Pair_Min": valid_pairs.min(),
                    "Pair_Max": valid_pairs.max(),
                }
            )

            try:
                hist, _ = np.histogram(valid_pairs, bins=100, range=(0, 1), density=True)
                p = hist / (hist.sum() + 1e-12)
                p = p[p > 1e-12]
                stats["Shannon_Entropy"] = -(p * np.log2(p)).sum()
            except Exception:
                stats["Shannon_Entropy"] = 0.0

    return stats


def calculate_diversity_metrics(sim_matrix, random_seed: int = 42):
    """从完整相似性矩阵计算多样性指标"""
    if sim_matrix is None:
        return {}

    mask = ~np.eye(sim_matrix.shape[0], dtype=bool)
    off_diagonal = sim_matrix[mask]

    np.fill_diagonal(sim_matrix, -1)
    knn_sim = np.sort(sim_matrix, axis=1)[:, -30:]

    n_pairs = min(1_000_000, len(off_diagonal))
    rng = np.random.default_rng(random_seed)
    sampled_pairs = rng.choice(off_diagonal, size=n_pairs, replace=False)

    return diversity_stats(knn_sim, sampled_pairs)
