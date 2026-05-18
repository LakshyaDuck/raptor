from config import RaptorConfig
from models import Node
from provider import LLMProvider


class Summarizer:
    def __init__(self, config: RaptorConfig, provider: LLMProvider):
        self.config = config
        self.provider = provider

    def summarize(self, nodes: list[Node]) -> str:
        string = ""
        for i in nodes:
            string = string + "\n\n" + i.text

        if len(string.split()) < self.config.max_tokens_per_cluster:
            prompt = f"""summarize the following passage concisely, preserve key facts, no hallucination:\nPASSAGES:\n{string}\nSUMMARY:"""
            summary = self.provider.complete(prompt)
            return summary
        else:
            summaries = []
            while len(string.split()) > self.config.max_tokens_per_cluster:
                st = " ".join(string.split()[: self.config.max_tokens_per_cluster])
                prompt = f"""summarize the following passage concisely, preserve key facts, no hallucination:\nPASSAGES:\n{st}\nSUMMARY:"""
                summaries.append(self.provider.complete(prompt))
                string = " ".join(string.split()[self.config.max_tokens_per_cluster :])
            prompt = f"""summarize the following passage concisely, preserve key facts, no hallucination:\nPASSAGES:\n{string}\nSUMMARY:"""
            summaries.append(self.provider.complete(prompt))
            return "\n\n".join(summaries)
