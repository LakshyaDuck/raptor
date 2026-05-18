import numpy as np

from config import RaptorConfig
from models import Node, RaptorTree


def collapsed_retrieval(
    query_embedding, tree: RaptorTree, config: RaptorConfig
) -> list[Node]:
    nodes = tree.all_nodes()
    nodes = [n for n in nodes if n.embedding is not None]
    assert config.embedding_dim == nodes[0].embedding.shape[0]
    embeddings = np.zeros((len(nodes), config.embedding_dim))
    for i, node in enumerate(nodes):
        embeddings[i] = node.embedding
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

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
    reversed = []
    nodes = tree.nodes_at(tree.depth)
    idtonode = {node.node_id: node for node in tree.all_nodes()}
    similarities = np.dot(
        np.array([node.embedding for node in nodes]), query_embedding
    ) / (
        np.linalg.norm(np.array([node.embedding for node in nodes]), axis=1)
        * np.linalg.norm(query_embedding)
    )
    sorted_indices = np.argsort(similarities)[::-1]
    reversed.extend([nodes[i] for i in sorted_indices[: config.gmm_max_components]])
    layer = tree.depth - 1
    while layer >= 0:
        new_nodes = []
        for i in sorted_indices:
            node = idtonode[nodes[i].node_id]
            for id in node.children_ids:
                new_nodes.append(idtonode[id])
        nodes = new_nodes
        similarities = np.dot(
            np.array([node.embedding for node in nodes]), query_embedding
        ) / (
            np.linalg.norm(np.array([node.embedding for node in nodes]), axis=1)
            * np.linalg.norm(query_embedding)
        )
        sorted_indices = np.argsort(similarities)[::-1]
        reversed.extend([nodes[i] for i in sorted_indices[: config.gmm_max_components]])
        layer -= 1
    return reversed
