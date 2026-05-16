from typing import Tuple

import numpy as np


class UMAP:
    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: int = 2,
        random_state: int = 42,
        min_dist: float = 0.1,
        n_epochs: int = 100,
        learning_rate: float = 1.0,
        verbose: bool = False,
        metric: str = "cosine",
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.random_state = random_state
        self.min_dist = min_dist
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.verbose = verbose

        if random_state is not None:
            np.random.seed(random_state)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        n_samples = len(X)
        knn_indices, knn_distances = self._compute_knn(X)
        graph = self._build_graph(X, knn_indices, knn_distances)
        embedding = self._initialize_embedding(n_samples)
        self.embedding_ = self._optimize_embedding(graph, X, embedding)
        return self.embedding_

    def _compute_knn(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric).fit(X)
        knn_distances, knn_indices = nbrs.kneighbors(X)
        return knn_indices, knn_distances

    def _build_graph(
        self, X: np.ndarray, knn_indices: np.ndarray, knn_distances: np.ndarray
    ) -> np.ndarray:
        n_samples = len(X)
        graph = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j, dist in zip(knn_indices[i], knn_distances[i]):
                weight = np.exp(-dist)
                graph[i, j] = weight
        graph = np.maximum(graph, graph.T)
        return graph

    def _initialize_embedding(self, n_samples: int) -> np.ndarray:
        return np.random.uniform(-10, 10, (n_samples, self.n_components))

    def _optimize_embedding(
        self, graph: np.ndarray, X: np.ndarray, embedding: np.ndarray
    ) -> np.ndarray:
        embedding = embedding.copy()
        n_samples = len(X)
        for epoch in range(self.n_epochs):
            for i in range(n_samples):
                positive = np.where(graph[i] > 0)[0]
                if len(positive) > 0:
                    j = np.random.choice(positive)
                    diff = embedding[i] - embedding[j]
                    dist = np.linalg.norm(diff) + 1e-10
                    embedding[i] -= self.learning_rate * (diff / dist) * 0.1

                k = np.random.randint(0, n_samples)
                diff = embedding[i] - embedding[k]
                dist = np.linalg.norm(diff) + 1e-10
                embedding[i] += self.learning_rate * (diff / (dist**2 + 1)) * 0.01

            if self.verbose and epoch % 50 == 0:
                print(f"Epoch {epoch}/{self.n_epochs}")
        return embedding


if __name__ == "__main__":
    umap = UMAP(n_neighbors=5, n_components=2, random_state=42, verbose=True)
    X = np.random.randn(50, 768)
    Y = umap.fit_transform(X)
    print(f"Input: {X.shape} → Output: {Y.shape} ✓")
