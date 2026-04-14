from abc import ABC, abstractmethod

import numpy as np
from anthropic import Anthropic
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from config import RaptorConfig


class LLMProvider(ABC):
    def __init__(self, config: RaptorConfig):
        super().__init__()

    @abstractmethod
    def complete(self, prompt: str) -> str:
        pass


class LocalProvider(LLMProvider):
    def __init__(self, config: RaptorConfig):
        super().__init__(config)
        self.client = OpenAI(base_url=config.ollama_base_url, api_key="ollama")
        self.model = config.summarization_model
        self.max_tokens = config.summary_max_tokens

    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    def __init__(self, config: RaptorConfig):
        super().__init__(config)
        self.model = config.anthropic_model
        self.client = Anthropic()
        self.max_tokens = config.eval_max_tokens

    def complete(self, prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class EmbeddingProvider:
    def __init__(self, config: RaptorConfig):
        self.config = config
        self.model = SentenceTransformer(config.embedding_model)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts)
        return vectors
