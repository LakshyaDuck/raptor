from chunker import chunk_text
from config import RaptorConfig
from embedder import Embedder
from eval import EvalPipeline
from models import Node, RaptorTree
from provider import AnthropicProvider, LocalProvider
from query_router import QueryRouter
from summarizer import Summarizer
from tree_builder import TreeBuilder


class Raptor:
    def __init__(
        self,
        config: RaptorConfig,
        embedder: Embedder,
        summarizer: Summarizer,
        tree_builder: TreeBuilder,
        query_router: QueryRouter,
        eval_pipeline: EvalPipeline,
        provider: LocalProvider,
        eval_provider: AnthropicProvider,
    ):
        self.config = config
        self.embedder = embedder
        self.summarizer = summarizer
        self.tree_builder = tree_builder
        self.query_router = query_router
        self.eval_pipeline = eval_pipeline
        self.provider = provider
        self.eval_provider = eval_provider

    def ingest(self, document: list[str]):
        text = " ".join(document)
        chunks = chunk_text(text, self.config)
        nodes = []
        for chunk in chunks:
            nodes.append(
                Node(text=chunk, layer=0, children_ids=[], embedding=None, metadata={})
            )
        nodes = self.embedder.embed_nodes(nodes)
        self.tree = self.tree_builder.build_tree(nodes)

    def query(self, query: str):
        if not self.tree:
            return []
        return self.query_router.route_query(query, self.tree, self.config)

    def evaluate(self, query: str, answer: str, nodes: list[Node]):
        return self.eval_pipeline.evaluate(query, answer, nodes)
