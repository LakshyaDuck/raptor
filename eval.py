import re

from config import RaptorConfig
from models import Node
from provider import AnthropicProvider, LLMProvider


class EvalPipeline:
    def __init__(self, config: RaptorConfig, provider: LLMProvider | None = None):
        self.config = config
        self.provider = provider or AnthropicProvider(config)

    def faithfulness(self, query: str, answer: str, nodes: list[Node]) -> float:
        prompt = f"I want you to act as a faithfulness evaluator who returns a score between 0 and 1 for the faithfulness of the answer to the query. Definition of faithfulness here is the groundedness of the answer to the query in the nodes of the tree. No explanation needed. Just return the score. Query: {query}\nAnswer: {answer}\nNodes:\n{self._format_nodes(nodes)}"
        score = self.provider.complete(prompt)
        return self._parse_score(score)

    def context_precision(self, query: str, nodes: list[Node]) -> float:
        prompt = f"I want you to act as a context precision evaluator who returns a score between 0 and 1 for the precision of the context used to answer the query. Definition of precision here is the relevance of the nodes to the question. No explanation needed. Just return the score. Query: {query}\nNodes:\n{self._format_nodes(nodes)}"
        score = self.provider.complete(prompt)
        return self._parse_score(score)

    def answer_relevance(self, query: str, answer: str) -> float:
        prompt = f"I want you to act as an answer relevance evaluator who returns a score between 0 and 1 for the relevance of the answer to the query. Definition of relevance here is the answer's ability to address the query. No explanation needed. Just return the score. Query: {query}\nAnswer: {answer}"
        score = self.provider.complete(prompt)
        return self._parse_score(score)

    def _parse_score(self, score: str) -> float:
        match = re.search(r"[-+]?(?:\d*\.\d+|\d+)", score)
        if not match:
            return 0.0
        value = float(match.group(0))
        return max(0.0, min(1.0, value))

    def _format_nodes(self, nodes: list[Node]) -> str:
        lines = []
        for index, item in enumerate(nodes, start=1):
            node = item[0] if isinstance(item, tuple) else item
            lines.append(f"[{index}] {node.text}")
        return "\n".join(lines)

    def evaluate(self, query: str, answer: str, nodes: list[Node]) -> dict[str, float]:
        faithfulness = self.faithfulness(query, answer, nodes)
        context_precision = self.context_precision(query, nodes)
        answer_relevance = self.answer_relevance(query, answer)
        return {
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "answer_relevance": answer_relevance,
            "score": (faithfulness + context_precision + answer_relevance) / 3,
        }
