import numpy as np

from config import RaptorConfig
from models import Node, RaptorTree


def _all_collapsed_scores(query_embedding, tree, config):
    nodes = [n for n in tree.all_nodes() if n.embedding is not None]
    if not nodes:
        return []
    embeddings = np.array([n.embedding for n in nodes])
    denominator = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    sims = np.dot(embeddings, query_embedding) / (denominator)
    return [(nodes[i], float(sims[i])) for i in range(len(nodes))]


def _all_traversal_scores(query_embedding, tree, config):
    idtonode = {n.node_id: n for n in tree.all_nodes()}
    scores = {}
    nodes = tree.nodes_at(tree.depth)
    layer = tree.depth
    while layer >= 0 and nodes:
        embeddings = np.array([n.embedding for n in nodes])
        denominator = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(
            query_embedding
        )
        denominator = np.where(denominator == 0, 1e-10, denominator)
        sims = np.dot(embeddings, query_embedding) / (denominator)
        sorted_indices = np.argsort(sims)[::-1]
        top_indices = sorted_indices[: config.gmm_max_components]
        for i in top_indices:
            node = nodes[i]
            score = float(sims[i])
            if node.node_id not in scores or score > scores[node.node_id][1]:
                scores[node.node_id] = (node, score)
        new_nodes = []
        for i in top_indices:
            for child_id in nodes[i].children_ids:
                if child_id in idtonode:
                    new_nodes.append(idtonode[child_id])
        nodes = new_nodes
        layer -= 1
    result = list(scores.values())
    all_nodes = tree.all_nodes()
    scored_ids = {n.node_id for n, _ in result}
    for node in all_nodes:
        if node.node_id not in scored_ids:
            result.append((node, 0.0))
    return result


def collapsed_retrieval(
    query_embedding, tree: RaptorTree, config: RaptorConfig
) -> list[Node]:
    nodes = tree.all_nodes()
    nodes = [n for n in nodes if n.embedding is not None]
    if not nodes:
        return []
    embeddings = np.array([node.embedding for node in nodes])
    denominator = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    similarities = np.dot(embeddings, query_embedding) / (denominator)

    sorted_indices = np.argsort(similarities)[::-1]
    length = 0
    i = 0
    retrieved = []
    while length < config.retrieval_token_budget and i < len(nodes):
        length += len(nodes[sorted_indices[i]].text.split())
        retrieved.append(nodes[sorted_indices[i]])
        i += 1
    return retrieved


def tree_traverse(
    query_embedding, tree: RaptorTree, config: RaptorConfig
) -> list[Node]:
    retrieved = []
    nodes = tree.nodes_at(tree.depth)
    if not nodes:
        return []
    idtonode = {node.node_id: node for node in tree.all_nodes()}
    embeddings = np.array([node.embedding for node in nodes])
    denominator = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    similarities = np.dot(embeddings, query_embedding) / denominator
    sorted_indices = np.argsort(similarities)[::-1]
    reversed_indices = sorted_indices[: config.gmm_max_components]
    retrieved.extend([nodes[i] for i in reversed_indices])
    layer = tree.depth - 1
    while layer >= 0:
        new_nodes = []
        for i in reversed_indices:
            node = idtonode[nodes[i].node_id]
            for id in node.children_ids:
                new_nodes.append(idtonode[id])
        nodes = new_nodes
        if not nodes:
            break
        embeddings = np.array([node.embedding for node in nodes])
        denominator = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(
            query_embedding
        )
        denominator = np.where(denominator == 0, 1e-10, denominator)
        similarities = np.dot(embeddings, query_embedding) / denominator
        sorted_indices = np.argsort(similarities)[::-1]
        reversed_indices = sorted_indices[: config.gmm_max_components]
        retrieved.extend([nodes[i] for i in reversed_indices])
        layer -= 1
    return retrieved
