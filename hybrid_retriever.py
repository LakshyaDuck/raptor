from rank_bm25 import BM25Okapi

from config import RaptorConfig
from models import Node, RaptorTree
from retriever import _all_collapsed_scores, _all_traversal_scores


def _normalize_scores(node_scores: list[tuple[Node, float]]) -> dict[Node, float]:
    if not node_scores:
        return {}

    values = [score for _, score in node_scores]
    min_score = min(values)
    max_score = max(values)
    if max_score == min_score:
        return {node: 0.0 for node, _ in node_scores}

    return {
        node: (score - min_score) / (max_score - min_score)
        for node, score in node_scores
    }


def _nodes_within_token_budget(
    scored_nodes: list[tuple[Node, float]], token_budget: int
) -> list[tuple[Node, float]]:
    retrieved = []
    used_tokens = 0
    for node, score in scored_nodes:
        node_tokens = len(node.text.split())
        if used_tokens + node_tokens > token_budget:
            continue
        retrieved.append((node, score))
        used_tokens += node_tokens
    return retrieved


def hybrid_retrieve(
    query, query_emb, tree: RaptorTree, config: RaptorConfig
) -> list[tuple[Node, float]]:
    all_nodes = tree.all_nodes()
    if not all_nodes:
        return []

    dense_scores = _all_collapsed_scores(query_emb, tree, config)
    tree_scores = _all_traversal_scores(query_emb, tree, config)

    corpus = [node.text.lower().split() for node in all_nodes]
    bm25 = BM25Okapi(corpus)
    bm25_raw_scores = bm25.get_scores(query.lower().split())
    bm25_scores = _normalize_scores(
        [(node, float(bm25_raw_scores[i])) for i, node in enumerate(all_nodes)]
    )
    dense_scores = _normalize_scores(dense_scores)
    tree_scores = _normalize_scores(tree_scores)

    scores = {}
    for node in all_nodes:
        scores[node] = {"node": node, "dense": 0.0, "bm25": 0.0, "tree": 0.0}
    for node, score in dense_scores.items():
        scores[node]["dense"] = score
    for node, score in tree_scores.items():
        scores[node]["tree"] = score
    for node, score in bm25_scores.items():
        scores[node]["bm25"] = score

    final_scores = []
    for node in all_nodes:
        score = (
            config.dense_weight * scores[node]["dense"]
            + config.tree_weight * scores[node]["tree"]
            + config.bm25_weight * scores[node]["bm25"]
        )
        final_scores.append((node, score))

    final_scores.sort(key=lambda item: item[1], reverse=True)
    return _nodes_within_token_budget(final_scores, config.retrieval_token_budget)
