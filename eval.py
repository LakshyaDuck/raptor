from config import RaptorConfig
from models import Node
from provider import LocalProvider


class EvalPipeline:
    def __init__(self, config: RaptorConfig):
        self.config = config
        self.provider = LocalProvider(config)

    def faithfulness(self, query: str, answer: str, nodes: list[Node]) -> float:
        prompt = f"I want you to act as a faithfulness evaluator who returns a score between 0 and 1 for the faithfulness of the answer to the query. Definition of faithfulness here is the groundedness of the answer to the query in the nodes of the tree. No explanation needed. Just return the score. Query: {query}\nAnswer: {answer}\nNodes: {nodes}"
        score = self.provider.complete(prompt)
        return float(score) or 0.0

    def context_precision(self, query: str, nodes: list[Node]) -> float:
        prompt = f"I want you to act as a context precision evaluator who returns a score between 0 and 1 for the precision of the context used to answer the query. Definition of precision here is the relevance of the nodes to the question. No explanation needed. Just return the score. Query: {query}\nNodes: {nodes}"
        score = self.provider.complete(prompt)
        return float(score) or 0.0

    def answer_relevance(self, query: str, answer: str) -> float:
        prompt = f"I want you to act as an answer relevance evaluator who returns a score between 0 and 1 for the relevance of the answer to the query. Definition of relevance here is the answer's ability to address the query. No explanation needed. Just return the score. Query: {query}\nAnswer: {answer}"
        score = self.provider.complete(prompt)
        return float(score) or 0.0

    def evaluate(self, query: str, answer: str, nodes: list[Node]) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness(query, answer, nodes),
            "context_precision": self.context_precision(query, nodes),
            "answer_relevance": self.answer_relevance(query, answer),
            "score": (
                self.faithfulness(query, answer, nodes)
                + self.context_precision(query, nodes)
                + self.answer_relevance(query, answer)
            )
            / 3,
        }
