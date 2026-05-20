from sentence_transformers import SentenceTransformer

from config import RaptorConfig
from models import Node


class Embedder:
    def __init__(self, config: RaptorConfig):
        self.model = SentenceTransformer(
            config.embedding_model,
            device=config.embedding_device,
            local_files_only=config.embedding_local_files_only,
        )
        self.batch_size = config.embedding_batch_size

    def embed_nodes(self, nodes: list[Node]) -> list[Node]:

        if not nodes:
            return []
        text = [node.text for node in nodes]
        embeddings = self.model.encode(text, batch_size=self.batch_size)

        for i, node in enumerate(nodes):
            node.embedding = embeddings[i]

        return nodes
