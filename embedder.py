from sentence_transformers import SentenceTransformer

from config import RaptorConfig
from models import Node


class Embedder:
    def __init__(self, config: RaptorConfig):
        self.model = config.embedding_model

    def embed_nodes(self, nodes: list[Node]) -> list[Node]:
        model = SentenceTransformer(self.model)
        for node in nodes:
            text = node.text
            embeddings = model.encode(text)
            node.embedding = embeddings

        return nodes
