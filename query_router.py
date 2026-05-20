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
        embedder: Embedder,
        local: LocalProvider,
    ):
        self.config = config
        self.provider = local
        self.embedder = embedder

    def classify(self, query: str) -> str:
        prompt = f"""Classify the following query as one of the following categories: Factual, Synthesis, Multi-Hop
Where:
Factual: the query is a factual question
Synthesis: the query is a synthesis question
Multi-Hop: the query requires multiple hops to answer
Query: {query}
Answer in only one category name, no explanation."""
        return self.provider.complete(prompt)

    def _normalize_category(self, category: str) -> str:
        normalized = category.strip().lower().replace("_", "-")
        if "multi" in normalized and "hop" in normalized:
            return "multi-hop"
        if "fact" in normalized:
            return "factual"
        if "synth" in normalized or "systh" in normalized:
            return "synthesis"
        return "synthesis"

    def _format_nodes(self, nodes) -> str:
        lines = []
        for index, item in enumerate(nodes, start=1):
            node = item[0] if isinstance(item, tuple) else item
            lines.append(f"[{index}] {node.text}")
        return "\n".join(lines)

    def _answer_prompt(self, query: str, nodes, mode: str) -> str:
        return f"""Use the context to answer the question.

Question: {query}

Context:
{self._format_nodes(nodes)}

Answer:"""

    def route_query(self, query: str, tree: RaptorTree, config: RaptorConfig) -> str:
        try:
            print(f"\n[route_query] Starting...")

            # Step 1: Classify
            print(f"[route_query] Step 1: Classifying query...")
            category_raw = self.classify(query)
            print(f"[route_query] Raw category: '{category_raw}'")

            if category_raw is None:
                print(f"[route_query] ERROR: classify() returned None!")
                return "ERROR: Failed to classify query"

            category = self._normalize_category(category_raw)
            print(f"[route_query] Normalized category: {category}")

            # Step 2: Embed query
            print(f"[route_query] Step 2: Embedding query...")
            query_node = Node(
                text=query, layer=0, embedding=None, children_ids=[], metadata={}
            )
            embedded_nodes = self.embedder.embed_nodes([query_node])

            if not embedded_nodes or embedded_nodes[0].embedding is None:
                print(f"[route_query] ERROR: Failed to embed query!")
                return "ERROR: Failed to embed query"

            query_embedding = embedded_nodes[0].embedding
            print(f"[route_query] Query embedded: {query_embedding.shape}")

            # Step 3: Retrieve nodes
            print(f"[route_query] Step 3: Retrieving nodes (category={category})...")
            nodes = None

            if category == "factual":
                print(f"[route_query] Using collapsed_retrieval...")
                nodes = collapsed_retrieval(query_embedding, tree, config)
            elif category == "multi-hop":
                print(f"[route_query] Using tree_traverse...")
                nodes = tree_traverse(query_embedding, tree, config)
            else:
                print(f"[route_query] Using hybrid_retrieve...")
                nodes = hybrid_retrieve(query, query_embedding, tree, config)

            print(f"[route_query] Retrieved nodes: {nodes}")

            if nodes is None:
                print(f"[route_query] ERROR: Retriever returned None!")
                return "ERROR: No relevant nodes found"

            if isinstance(nodes, list) and len(nodes) == 0:
                print(f"[route_query] WARNING: No nodes retrieved")
                return "ERROR: No relevant nodes found"

            print(f"[route_query] Retrieved {len(nodes)} nodes")

            # Step 4: Generate answer
            print(f"[route_query] Step 4: Generating answer prompt...")
            prompt = self._answer_prompt(query, nodes, category)
            print(f"[route_query] Prompt created: {len(prompt)} chars")
            print(f"[route_query] Prompt preview: {prompt[:300]}")

            print(f"[route_query] Step 5: Calling provider.complete()...")
            answer = self.provider.complete(prompt)
            print(f"[route_query] Answer received: {answer}")

            if answer is None:
                print(f"[route_query] ERROR: provider.complete() returned None!")
                return "ERROR: Failed to generate answer"

            print(f"[route_query] ✓ Success! Answer length: {len(answer)}")
            return answer

        except Exception as e:
            print(f"[route_query] ❌ EXCEPTION: {e}")
            import traceback

            traceback.print_exc()
            return f"ERROR: {e}"
