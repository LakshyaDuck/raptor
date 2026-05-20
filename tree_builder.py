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
            # Step 1: Try to cluster
            clusters = cluster_nodes(current_nodes, self.config)

            # Step 2: Check if clustering was effective
            # If no clustering happened, stop
            if (
                not clusters
                or len(clusters) == 1
                or len(clusters)
                == len(
                    current_nodes
                )  # Clustering failed (each node is its own cluster)
                or layer >= self.config.max_tree_layers
            ):
                break  # ← Exit AFTER failed clustering, before creating nodes

            # Step 3: Clustering worked, create parent nodes
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

            # Step 4: Embed the NEW parent nodes
            self.embedder.embed_nodes(nodes)

            # Step 5: Add them to tree
            for node in nodes:
                self.tree.add_node(node)

            # Step 6: Set up for next iteration
            current_nodes = nodes
            layer += 1

        return self.tree
