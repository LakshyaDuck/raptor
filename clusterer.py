import numpy as np
from sklearn.mixture import GaussianMixture

from config import RaptorConfig
from models import Node
from umap import UMAP


def _select_n_components(data: np.ndarray, max_k: int) -> int:
    n_samples = len(data)
    if n_samples <= 1:
        return 1
    max_k = min(max_k, n_samples)
    if max_k < 2:
        return 1
    bics = []
    ks = range(1, max_k + 1)
    for k in ks:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(data)
        bics.append(gmm.bic(data))
    return ks[np.argmin(bics)] if bics else 1


def cluster_nodes(nodes: list[Node], config: RaptorConfig) -> list[list[Node]]:
    if not nodes:
        return []

    if len(nodes) == 1:
        return [nodes]

    embeddings = np.array([node.embedding for node in nodes])

    # ===== FIRST PASS: GLOBAL =====
    global_umap = UMAP(
        n_neighbors=min(config.umap_n_neighbors_global, len(nodes) - 1),
        n_components=min(config.umap_n_components, len(nodes) - 1),
        random_state=42,
    )
    global_emb = np.asarray(global_umap.fit_transform(embeddings))
    n_global = _select_n_components(global_emb, config.gmm_max_components)
    gmm_global = GaussianMixture(n_components=n_global, random_state=42)
    gmm_global.fit(global_emb)
    global_probs = gmm_global.predict_proba(global_emb)  # shape (n_nodes, n_global)

    # ===== SECOND PASS: LOCAL =====
    clusters = []

    for global_cluster_j in range(n_global):
        # Soft assignment — include node if probability exceeds threshold
        global_mask = global_probs[:, global_cluster_j] >= config.gmm_soft_threshold
        indices = np.where(global_mask)[0]
        cluster_nodes_list = [nodes[i] for i in indices]
        cluster_embeddings = embeddings[indices]

        if len(cluster_nodes_list) == 0:
            continue

        if len(cluster_nodes_list) == 1:
            clusters.append(cluster_nodes_list)
            continue

        # Local UMAP on this global cluster's nodes
        local_umap = UMAP(
            n_neighbors=min(config.umap_n_neighbors_local, len(cluster_nodes_list) - 1),
            n_components=min(config.umap_n_components, len(cluster_nodes_list) - 1),
            random_state=42,
        )
        local_emb = np.asarray(local_umap.fit_transform(cluster_embeddings))
        n_local = _select_n_components(local_emb, config.gmm_max_components)
        gmm_local = GaussianMixture(n_components=n_local, random_state=42)
        gmm_local.fit(local_emb)
        local_probs = gmm_local.predict_proba(
            local_emb
        )  # shape (n_cluster_nodes, n_local)

        # For each local cluster, soft-assign nodes
        for local_cluster_j in range(n_local):
            local_mask = local_probs[:, local_cluster_j] >= config.gmm_soft_threshold
            final_nodes = [
                cluster_nodes_list[i]
                for i in range(len(cluster_nodes_list))
                if local_mask[i]
            ]
            if len(final_nodes) >= config.min_cluster_size:
                clusters.append(final_nodes)

    return clusters
