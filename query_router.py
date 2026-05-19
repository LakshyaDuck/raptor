from config import RaptorConfig
from embedder import Embedder
from hybrid_retriever import hybrid_retrieve
from models import Node, RaptorTree
from provider import LocalProvider
from retriever import collapsed_retrieval, tree_traverse


class QueryRouter:
    def __init__(
        self,
        config: RaptorConfig,
        tree: RaptorTree,
        embedder: Embedder,
        local: LocalProvider,
    ):
        self.config = config
        self.provider = local
        self.embedder = embedder

    def classify(self, query: str) -> str:
        prompt = "Classify the following query as one of the following categories: Factual, Systhesis and Multi-Hop\nWhere:\nFactual: the query is a factual question\nSysthesis: the query is a synthesis question\nMulti-Hop: the query requires multiple hops to answer"
        return self.provider.complete(prompt)

    def route_query(self, query: str, tree: RaptorTree, config: RaptorConfig) -> str:
        category = self.classify(query)
        query_embedding = self.embedder.embed_nodes(
            [Node(text=query, layer=0, embedding=None, children_ids=[], metadata={})]
        )
        if category == "Factual":
            nodes = collapsed_retrieval(query_embedding[0].embedding, tree, config)
            prompt = (
                f"Answer the following factual question: {query}\n from nodes: {nodes}"
            )
            return self.provider.complete(prompt)
        elif category == "Multi-Hop":
            nodes = tree_traverse(query_embedding[0].embedding, tree, config)
            prompt = (
                f"Answer the following multi-hop question: {query} from nodes: {nodes}"
            )
            return self.provider.complete(prompt)
        elif category == "Systhesis":
            nodes = hybrid_retrieve(query, query_embedding[0].embedding, tree, config)
            prompt = (
                f"Answer the following synthesis question: {query} from nodes: {nodes}"
            )
            return self.provider.complete(prompt)
