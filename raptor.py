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
        config,
        embedder,
        summarizer,
        tree_builder,
        query_router,
        eval_pipeline,
        local_provider,
        anthropic_provider,
    ):
        self.config = config
        self.embedder = embedder
        self.summarizer = summarizer
        self.tree_builder = tree_builder
        self.query_router = query_router
        self.eval_pipeline = eval_pipeline
        self.local_provider = local_provider
        self.anthropic_provider = anthropic_provider
        self.tree: RaptorTree | None = None

    def ingest(self, document: list[str]):
        text = " ".join(document)
        nodes = chunk_text(text, self.config)
        nodes = self.embedder.embed_nodes(nodes)
        self.tree = self.tree_builder.build_tree(nodes)
        print(len(self.tree.all_nodes()))
        print(self.tree.depth)
        print(len(self.tree.nodes_at(self.tree.depth)))

    def query(self, query: str):
        if not self.tree:
            return []
        return self.query_router.route_query(query, self.tree, self.config)

    def evaluate(self, query: str, answer: str, nodes: list[Node]):
        return self.eval_pipeline.evaluate(query, answer, nodes)
