class RaptorConfig:
    def __init__(self):

        # chunking
        self.chunk_size = 100
        self.chunk_overlap = 15

        # embedding
        self.embedding_model = "multi-qa-mpnet-base-cos-v1"
        self.embedding_batch_size = 32

        # umap — two values, not one (paper correction)
        self.umap_n_neighbors_global = 50
        self.umap_n_neighbors_local = 10
        self.umap_n_components = 10

        # gmm
        self.gmm_max_components = 10
        self.gmm_soft_threshold = 0.1

        # summarization
        self.summarization_model = "qwen3.5:9b"
        self.max_tokens_per_cluster = 3500
        self.summary_max_tokens = 256

        # tree
        self.max_tree_layers = 4
        self.min_cluster_size = 2

        # retrieval
        self.retrieval_token_budget = 256
        self.bm25_weight = 0.25
        self.dense_weight = 0.25
        self.tree_weight = 0.5

        # providers
        self.ollama_base_url = "http://localhost:11434/v1"
        self.anthropic_model = "claude-haiku-4-5-20251001"

        # eval
        self.eval_max_tokens = 10

    def __repr__(self):
        attrs = vars(self)
        lines = [f"  {k} = {v!r}" for k, v in attrs.items()]
        return "RaptorConfig(\n" + "\n".join(lines) + "\n)"
