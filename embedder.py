from sentence_transformers import SentenceTransformer

from config import RaptorConfig
from models import Node


class Embedder:
    def __init__(self, config: RaptorConfig):
        self.model = SentenceTransformer(config.embedding_model)

    def embed_nodes(self, nodes: list[Node]) -> list[Node]:

        embeddings = self.model.encode(node.text for node in nodes)

        for i, node in enumerate(nodes):
            node.embedding = embeddings[i]

        return nodes
