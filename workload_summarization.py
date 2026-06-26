
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkloadSummarizer:
    def __init__(self, query_vectors: np.ndarray, queries: List[str]):
        self.query_vectors = query_vectors
        self.queries = queries
        self.optimal_k = None
        self.cluster_labels = None
        self.centroids = None
        self.summary_queries = None
        # Store intermediate data for export
        self._inertias = []
        self._silhouette_scores = []
        self._k_values = []

    def determine_optimal_k(self, k_range: Optional[Tuple[int, int]] = None,
                           method: str = 'elbow',
                           output_dir: str = '.') -> int:
        n_queries = len(self.queries)

        if k_range is None:
            min_k = 2
            max_k = min(50, max(n_queries // 10, 3))
            k_range = (min_k, max_k)

        k_values = list(range(k_range[0], k_range[1] + 1))
        self._k_values = k_values

        # Always compute both inertias and silhouette scores
        inertias = []
        silhouette_scores_list = []

        for k in k_values:
            logger.info(f"Fitting K-means with k={k} ...")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.query_vectors)
            inertias.append(float(kmeans.inertia_))
            score = silhouette_score(self.query_vectors, labels)
            silhouette_scores_list.append(float(score))

        self._inertias = inertias
        self._silhouette_scores = silhouette_scores_list

        if method == 'elbow':
            optimal_k = self._find_elbow_point(k_values, inertias)
            self._plot_elbow_curve(k_values, inertias, optimal_k, output_dir)
        elif method == 'silhouette':
            optimal_k = k_values[int(np.argmax(silhouette_scores_list))]
            self._plot_silhouette_scores(k_values, silhouette_scores_list, optimal_k, output_dir)
        else:
            raise ValueError(f"Unknown method: {method}")

        self.optimal_k = optimal_k
        logger.info(f"Optimal K determined: {optimal_k}")
        return optimal_k

    def _find_elbow_point(self, k_values: list, inertias: List[float]) -> int:
        rates = []
        for i in range(1, len(inertias)):
            rate = abs(inertias[i] - inertias[i-1])
            rates.append(rate)

        if len(rates) > 1:
            second_deriv = []
            for i in range(1, len(rates)):
                second_deriv.append(abs(rates[i] - rates[i-1]))

            elbow_idx = int(np.argmax(second_deriv)) + 1
            return k_values[elbow_idx]

        return k_values[len(k_values) // 2]

    def _plot_elbow_curve(self, k_values: list, inertias: List[float],
                         optimal_k: int, output_dir: str = '.'):
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=6)
        plt.axvline(x=optimal_k, color='r', linestyle='--',
                   label=f'Optimal K={optimal_k}')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Sum of Squared Distances (Inertia)')
        plt.title('Elbow Method for Optimal K')
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(output_dir, 'elbow_curve.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Elbow curve saved to {save_path}")

    def _plot_silhouette_scores(self, k_values: list,
                               scores: List[float], optimal_k: int,
                               output_dir: str = '.'):
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, scores, 'go-', linewidth=2, markersize=6)
        plt.axvline(x=optimal_k, color='r', linestyle='--',
                   label=f'Optimal K={optimal_k}')
        plt.xlabel('Number of Clusters (K)')
        plt.ylabel('Silhouette Score')
        plt.title('Silhouette Analysis for Optimal K')
        plt.legend()
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(output_dir, 'silhouette_scores.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Silhouette scores saved to {save_path}")

    def cluster_and_summarize(self, k: Optional[int] = None) -> List[str]:
        if k is None:
            if self.optimal_k is None:
                logger.info("Determining optimal K...")
                k = self.determine_optimal_k()
            else:
                k = self.optimal_k

        logger.info(f"Performing K-means with K={k}")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.cluster_labels = kmeans.fit_predict(self.query_vectors)
        self.centroids = kmeans.cluster_centers_

        summary_queries = []
        summary_indices = []

        for cluster_id in range(k):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_vectors = self.query_vectors[cluster_mask]
            cluster_indices = np.where(cluster_mask)[0]

            if len(cluster_vectors) == 0:
                continue

            centroid = self.centroids[cluster_id]
            distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
            nearest_idx = cluster_indices[int(np.argmin(distances))]

            summary_queries.append(self.queries[nearest_idx])
            summary_indices.append(int(nearest_idx))

        self.summary_queries = summary_queries

        logger.info(f"Workload summarized from {len(self.queries)} to "
                   f"{len(summary_queries)} queries")
        logger.info(f"Compression ratio: "
                   f"{(1 - len(summary_queries)/len(self.queries))*100:.2f}%")

        return summary_queries

    def get_cluster_statistics(self) -> dict:
        if self.cluster_labels is None:
            raise ValueError("Must run cluster_and_summarize() first")

        unique_labels = np.unique(self.cluster_labels)
        stats = {
            'n_clusters': int(len(unique_labels)),
            'cluster_sizes': [],
            'compression_ratio': (1 - len(self.summary_queries) /
                                len(self.queries)) * 100
        }

        for cluster_id in range(len(unique_labels)):
            size = int(np.sum(self.cluster_labels == cluster_id))
            stats['cluster_sizes'].append(size)

        stats['avg_cluster_size'] = float(np.mean(stats['cluster_sizes']))
        stats['std_cluster_size'] = float(np.std(stats['cluster_sizes']))

        return stats

    def get_cluster_details(self, n_sample_queries: int = 10) -> List[Dict]:
        """Return per-cluster details: center query, count, sample queries."""
        if self.cluster_labels is None:
            raise ValueError("Must run cluster_and_summarize() first")

        details = []
        unique_labels = np.unique(self.cluster_labels)

        for cluster_id in range(len(unique_labels)):
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_vectors = self.query_vectors[cluster_mask]

            if len(cluster_vectors) == 0:
                continue

            centroid = self.centroids[cluster_id]
            distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
            sorted_order = np.argsort(distances)

            # Center query (nearest to centroid)
            center_idx = cluster_indices[sorted_order[0]]
            center_query = self.queries[center_idx]

            # Sample queries (closest to centroid)
            sample_count = min(n_sample_queries, len(sorted_order))
            sample_queries = []
            for i in range(sample_count):
                idx = cluster_indices[sorted_order[i]]
                sample_queries.append(self.queries[idx])

            details.append({
                'cluster_id': int(cluster_id),
                'center_query': center_query,
                'query_count': int(len(cluster_indices)),
                'sample_queries': sample_queries,
            })

        return details

    def visualize_clusters_2d(self, method: str = 'pca',
                              output_dir: str = '.') -> Optional[str]:
        if self.cluster_labels is None:
            raise ValueError("Must run cluster_and_summarize() first")

        from sklearn.decomposition import PCA

        os.makedirs(output_dir, exist_ok=True)

        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            vectors_2d = reducer.fit_transform(self.query_vectors)
        else:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            vectors_2d = reducer.fit_transform(self.query_vectors)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1],
                            c=self.cluster_labels, cmap='tab20',
                            alpha=0.6, s=10)
        plt.colorbar(scatter, label='Cluster ID')
        plt.title(f'Query Clustering Visualization ({method.upper()})')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)
        save_path = os.path.join(output_dir, f'cluster_visualization_{method}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Cluster visualization saved to {save_path}")
        return save_path
