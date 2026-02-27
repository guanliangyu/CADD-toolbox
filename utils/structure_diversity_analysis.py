"""
结构多样性评估 - 聚类与降维分析工具
"""

from __future__ import annotations

import inspect
import time
import warnings

import numpy as np
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.manifold import TSNE

try:
    import torch

    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except Exception:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False

try:
    from cuml.cluster import DBSCAN as cuDBSCAN
    from cuml.cluster import KMeans as cuKMeans

    CUML_AVAILABLE = True
except Exception:
    cuKMeans = None
    cuDBSCAN = None
    CUML_AVAILABLE = False

try:
    import umap

    UMAP_AVAILABLE = True
except Exception:
    UMAP_AVAILABLE = False

TSNE_ITER_PARAM = "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"


def initialize_cuda():
    """初始化 CUDA 设备并返回设备信息"""
    if not TORCH_AVAILABLE:
        st.sidebar.info("ℹ️ PyTorch未安装，将使用CPU计算")
        return False, "cpu"

    try:
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda" if cuda_available else "cpu")

        if cuda_available:
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1024**2
            gpu_mem_cached = torch.cuda.memory_reserved(0) / 1024**2

            st.sidebar.success("✅ CUDA可用，将使用GPU加速")
            st.sidebar.info(
                f"GPU信息:\n"
                f"- 设备: {gpu_name}\n"
                f"- 总显存: {gpu_mem_total:.1f}MB\n"
                f"- 已分配: {gpu_mem_alloc:.1f}MB\n"
                f"- 已缓存: {gpu_mem_cached:.1f}MB"
            )
        else:
            st.sidebar.info("ℹ️ CUDA不可用，将使用CPU计算")

        return cuda_available, device
    except Exception as e:
        st.sidebar.error(f"GPU初始化错误: {str(e)}")
        return False, "cpu"


def embed_umap(
    fps: np.ndarray,
    n_pca: int = 128,
    n_components: int = 2,
    random_seed: int = 42,
) -> np.ndarray:
    """先PCA降维再UMAP嵌入的优化降维方法"""
    try:
        if CUML_AVAILABLE and CUDA_AVAILABLE:
            st.info(f"🔧 使用 GPU PCA({n_pca}D) + UMAP({n_components}D) 降维...")
            import cupy as cp
            from cuml.decomposition import PCA as cuPCA
            from cuml.manifold import UMAP as cuUMAP

            n_pca_gpu = min(n_pca, fps.shape[1], fps.shape[0] - 1)
            fps_gpu = cp.asarray(fps.astype(np.float32), order="C")
            pca_gpu = cuPCA(n_components=n_pca_gpu, random_state=random_seed)
            x_pca_gpu = pca_gpu.fit_transform(fps_gpu)
            reducer_gpu = cuUMAP(
                n_components=n_components,
                n_neighbors=15,
                min_dist=0.1,
                random_state=random_seed,
            )
            embedding_gpu = reducer_gpu.fit_transform(x_pca_gpu)
            st.success(f"✅ GPU UMAP完成: {n_pca_gpu}D → {embedding_gpu.shape[1]}D")
            return cp.asnumpy(embedding_gpu)

        st.info(f"🔧 使用 IncrementalPCA({n_pca}D) + UMAP({n_components}D) 降维...")

        n_pca_cpu = min(n_pca, fps.shape[1], fps.shape[0] - 1)
        ipca = IncrementalPCA(n_components=n_pca_cpu, batch_size=min(10_000, fps.shape[0]))

        batch_size = min(10_000, fps.shape[0])
        for i in range(0, len(fps), batch_size):
            batch = fps[i : i + batch_size]
            ipca.partial_fit(batch)

        x_pca = ipca.transform(fps)
        st.info(f"✅ PCA完成: {fps.shape[1]}D → {x_pca.shape[1]}D")

        if not UMAP_AVAILABLE:
            st.warning("UMAP不可用，使用PCA降维到2D")
            if x_pca.shape[1] > n_components:
                final_pca = PCA(n_components=n_components, random_state=random_seed)
                return final_pca.fit_transform(x_pca)
            return x_pca[:, :n_components] if x_pca.shape[1] >= n_components else x_pca

        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=15,
            min_dist=0.1,
            metric="euclidean",
            random_state=random_seed,
            n_jobs=1,
        )

        embedding = reducer.fit_transform(x_pca)
        st.success(f"✅ UMAP完成: {x_pca.shape[1]}D → {embedding.shape[1]}D")
        return embedding

    except Exception as e:
        st.error(f"降维失败: {str(e)}")
        st.info("回退到标准PCA降维...")
        try:
            pca = PCA(n_components=n_components, random_state=random_seed)
            return pca.fit_transform(fps)
        except Exception as e2:
            st.error(f"PCA回退也失败: {str(e2)}")
            return None


def perform_optimized_clustering_analysis(
    fps: np.ndarray,
    cluster_method: str = "K-means",
    n_clusters: int = 5,
    eps: float = 0.3,
    min_samples: int = 5,
    use_minibatch: bool = True,
    force_device=None,
    random_seed: int = 42,
):
    """优化的聚类分析（直接基于指纹数据）"""
    np.random.seed(random_seed)
    st.info(f"🔧 使用优化降维 + {cluster_method} 聚类...")

    coords = embed_umap(fps, n_pca=128, n_components=2, random_seed=random_seed)
    if coords is None:
        return None

    coords_gpu = None
    if CUML_AVAILABLE and CUDA_AVAILABLE:
        try:
            import cupy as cp

            coords_gpu = cp.asarray(coords.astype(np.float32))
        except Exception as exc:
            st.warning(f"构建GPU数据失败，回退CPU: {exc}")
            coords_gpu = None

    clusters = None

    if cluster_method == "K-means":
        if coords_gpu is not None:
            try:
                st.info("🚀 使用 cuML KMeans 聚类...")
                kmeans_gpu = cuKMeans(
                    n_clusters=n_clusters,
                    random_state=random_seed,
                    max_iter=300,
                    tol=1e-4,
                )
                clusters = cp.asnumpy(kmeans_gpu.fit_predict(coords_gpu))
            except Exception as exc:
                st.warning(f"cuML KMeans 聚类失败，回退到CPU: {exc}")

        if clusters is None:
            if use_minibatch and len(fps) > 10_000:
                st.info("🚀 使用MiniBatchKMeans进行聚类...")
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    batch_size=min(10_000, len(fps) // 10),
                    random_state=random_seed,
                    n_init=3,
                    max_iter=100,
                )
            else:
                st.info("🔧 使用标准KMeans进行聚类...")
                kmeans = KMeans(
                    n_clusters=n_clusters,
                    random_state=random_seed,
                    n_init=10,
                    max_iter=300,
                )
            clusters = kmeans.fit_predict(coords)

    elif cluster_method == "DBSCAN":
        if coords_gpu is not None:
            try:
                st.info("🚀 使用 cuML DBSCAN 聚类...")
                dbscan_gpu = cuDBSCAN(eps=eps, min_samples=min_samples)
                clusters = cp.asnumpy(dbscan_gpu.fit_predict(coords_gpu))
            except Exception as exc:
                st.warning(f"cuML DBSCAN 聚类失败，回退到CPU: {exc}")

        if clusters is None:
            st.info("🔧 使用DBSCAN进行密度聚类...")
            try:
                import hdbscan

                dbscan = hdbscan.HDBSCAN(
                    min_cluster_size=min_samples,
                    min_samples=min_samples,
                    metric="euclidean",
                )
                clusters = dbscan.fit_predict(coords)
                st.info("✅ 使用HDBSCAN聚类")
            except ImportError:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
                clusters = dbscan.fit_predict(coords)
                st.info("✅ 使用标准DBSCAN聚类")

        n_noise = int(np.sum(clusters == -1))
        n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
        st.info(f"📊 DBSCAN结果: {n_clusters_found}个簇, {n_noise}个噪声点")
    else:
        raise ValueError(f"不支持的聚类方法: {cluster_method}")

    return {"coords": coords, "clusters": clusters, "cluster_method": cluster_method}


def perform_dimensionality_reduction(
    similarity_matrix,
    method: str = "t-SNE",
    perplexity: float = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    force_device=None,
    random_seed: int = 42,
):
    """执行降维操作（兼容模式专用）"""
    warnings.filterwarnings("ignore", category=UserWarning)

    np.random.seed(random_seed)
    debug_info = st.empty()
    start_time = time.time()

    cuda_available, device = initialize_cuda()

    if force_device == "cpu":
        st.info("🔧 Debug模式：降维强制使用CPU")
        cuda_available = False
    elif force_device == "gpu":
        if not cuda_available:
            st.warning("🔧 Debug模式：降维强制GPU失败，回退到CPU")
        else:
            st.info("🔧 Debug模式：降维强制使用GPU")

    distance_matrix = 1 - similarity_matrix
    distance_matrix = np.clip(distance_matrix, 0, 2)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    try:
        if method == "t-SNE":
            debug_info.info("🔧 使用CPU版本t-SNE确保结果一致性...")
            effective_perplexity = min(perplexity, (len(distance_matrix) - 1) // 3)

            tsne_params = {
                "n_components": 2,
                "perplexity": effective_perplexity,
                "random_state": random_seed,
                "metric": "precomputed",
                "init": "random",
                "learning_rate": "auto",
                "verbose": 0,
            }
            tsne_params[TSNE_ITER_PARAM] = 1000
            tsne = TSNE(**tsne_params)
            coords = tsne.fit_transform(distance_matrix)

        elif method == "UMAP":
            debug_info.info("🔧 使用UMAP进行降维...")
            if UMAP_AVAILABLE:
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    metric="precomputed",
                    random_state=random_seed,
                )
                coords = reducer.fit_transform(distance_matrix)
            else:
                st.error("UMAP库未安装，请安装umap-learn")
                return None

        elif method == "PCA":
            if cuda_available and CUML_AVAILABLE:
                debug_info.info("🚀 使用GPU PCA进行降维...")
                try:
                    import cupy as cp
                    from cuml.decomposition import PCA as cuPCA

                    sim_gpu = cp.asarray(np.ascontiguousarray(similarity_matrix.astype(np.float32, copy=False)))
                    pca_gpu = cuPCA(n_components=2, random_state=random_seed)
                    coords_gpu = pca_gpu.fit_transform(sim_gpu)
                    coords = cp.asnumpy(coords_gpu)
                except Exception as e:
                    st.warning(f"GPU PCA失败，回退CPU: {str(e)}")
                    pca = PCA(n_components=2, random_state=random_seed)
                    coords = pca.fit_transform(similarity_matrix)
                finally:
                    try:
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                    except Exception:
                        pass
            else:
                debug_info.info("🔧 使用PCA进行降维...")
                pca = PCA(n_components=2, random_state=random_seed)
                coords = pca.fit_transform(similarity_matrix)

        else:
            raise ValueError(f"不支持的降维方法: {method}")

        if coords.shape[1] != 2:
            raise ValueError(f"降维结果维度不正确: {coords.shape}")

        debug_info.success(f"✅ 降维完成 ({time.time() - start_time:.2f}秒)")
        return coords

    except Exception as e:
        debug_info.error(f"降维过程出错: {str(e)}")
        st.error(f"降维失败: {str(e)}")
        return None

    finally:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()


def perform_clustering_analysis(
    sim_matrix,
    cluster_method: str = "K-means",
    n_clusters: int = 5,
    eps: float = 0.3,
    min_samples: int = 5,
    method: str = "t-SNE",
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    force_device=None,
    random_seed: int = 42,
):
    """基于相似性矩阵的聚类分析"""
    if sim_matrix is None:
        return None

    n_samples = len(sim_matrix)
    if n_samples > 50_000:
        st.warning("⚠️ 大数据集建议使用优化版聚类方法 (perform_optimized_clustering_analysis)")

    np.random.seed(random_seed)

    cuda_available, device = initialize_cuda()
    use_gpu = cuda_available and CUML_AVAILABLE

    if force_device == "cpu":
        st.info("🔧 Debug模式：强制使用CPU计算")
        use_gpu = False
    elif force_device == "gpu":
        if not use_gpu:
            st.warning("🔧 Debug模式：强制GPU失败，回退到CPU")
        else:
            st.info("🔧 Debug模式：强制使用GPU计算")

    coords = perform_dimensionality_reduction(
        sim_matrix,
        method=method,
        perplexity=perplexity if method == "t-SNE" else 30.0,
        n_neighbors=n_neighbors if method == "UMAP" else 15,
        min_dist=min_dist if method == "UMAP" else 0.1,
        force_device=force_device,
        random_seed=random_seed,
    )
    if coords is None:
        return None

    clusters = None

    if cluster_method == "K-means":
        if use_gpu:
            st.info("🚀 使用GPU K-means...")
            try:
                import cupy as cp

                cp.random.seed(random_seed)
                coords_gpu = cp.asarray(coords.astype(np.float32))
                kmeans_gpu = cuKMeans(
                    n_clusters=n_clusters,
                    random_state=random_seed,
                    n_init=10,
                    max_iter=300,
                    tol=1e-4,
                )
                clusters_gpu = kmeans_gpu.fit_predict(coords_gpu)
                clusters = cp.asnumpy(clusters_gpu)
            except Exception as e:
                st.warning(f"GPU K-means失败，回退到CPU: {str(e)}")

        if clusters is None:
            if len(sim_matrix) > 10_000:
                kmeans_cpu = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=random_seed,
                    batch_size=min(5000, len(sim_matrix) // 10),
                    n_init=3,
                    max_iter=100,
                )
            else:
                kmeans_cpu = KMeans(
                    n_clusters=n_clusters,
                    random_state=random_seed,
                    n_init=10,
                    algorithm="lloyd",
                    max_iter=300,
                    tol=1e-4,
                )
            clusters = kmeans_cpu.fit_predict(coords)

    elif cluster_method == "DBSCAN":
        dist_matrix = 1 - sim_matrix
        dist_matrix = np.clip(dist_matrix, 0, 2)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)

        if use_gpu:
            st.info("🚀 使用GPU DBSCAN...")
            try:
                import cupy as cp

                dist_matrix_gpu = cp.asarray(dist_matrix, dtype=cp.float32)
                dbscan_gpu = cuDBSCAN(metric="precomputed", eps=eps, min_samples=min_samples)
                dbscan_clusters_gpu = dbscan_gpu.fit_predict(dist_matrix_gpu)
                clusters = cp.asnumpy(dbscan_clusters_gpu)
            except Exception as e:
                st.warning(f"GPU DBSCAN失败，回退到CPU: {str(e)}")

        if clusters is None:
            dbscan_cpu = DBSCAN(
                metric="precomputed",
                eps=eps,
                min_samples=min_samples,
                algorithm="auto",
                leaf_size=30,
            )
            clusters = dbscan_cpu.fit_predict(dist_matrix)
    else:
        raise ValueError(f"不支持的聚类方法: {cluster_method}")

    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.memory_allocated(device) / 1024**2
        gpu_mem_cached = torch.cuda.memory_reserved(device) / 1024**2
        st.success(
            f"✅ GPU加速聚类分析完成:\n"
            f"- GPU内存使用: {gpu_mem_alloc:.1f}MB\n"
            f"- GPU缓存: {gpu_mem_cached:.1f}MB"
        )

    return {"coords": coords, "clusters": clusters, "cluster_method": cluster_method}
