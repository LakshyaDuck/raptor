from clusterer import cluster_nodes
from config import RaptorConfig
from embedder import Embedder
from models import Node, RaptorTree
from summarizer import Summarizer


class TreeBuilder:
    def __init__(
        self, config: RaptorConfig, embedder: Embedder, summarizer: Summarizer
    ):
        self.config = config
        self.embedder = embedder
        self.summarizer = summarizer

    def build_tree(self, leaf_nodes: list[Node]) -> RaptorTree:
        self.tree = RaptorTree()
        for node in leaf_nodes:
            self.tree.add_node(node)
        current_nodes = leaf_nodes
        layer = 0
        while True:
            clusters = cluster_nodes(current_nodes, self.config)
            if (
                len(clusters) == 1
                or len(clusters) == len(current_nodes)
                or layer >= self.config.max_tree_layers
            ):
                break
            nodes = []
            for cluster in clusters:
                summary_text = self.summarizer.summarize(cluster)
                node = Node(
                    text=summary_text,
                    layer=layer + 1,
                    embedding=None,
                    children_ids=[node.node_id for node in cluster],
                    metadata={},
                )
                nodes.append(node)
            self.embedder.embed_nodes(nodes)
            for node in nodes:
                self.tree.add_node(node)
            current_nodes = nodes
            layer += 1
        return self.tree
