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
        try:
            print(f"\n[LocalProvider] START")
            print(f"[LocalProvider] Model: {self.model}")
            print(f"[LocalProvider] Prompt length: {len(prompt)}")

            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            print(f"[LocalProvider] Got response")
            print(f"[LocalProvider] Response object: {response}")

            if not response.choices:
                print(f"[LocalProvider] ERROR: No choices in response")
                return ""

            content = response.choices[0].message.content
            print(f"[LocalProvider] Content: {repr(content)}")
            print(f"[LocalProvider] Content type: {type(content).__name__}")
            print(f"[LocalProvider] END\n")

            return content if content else ""

        except Exception as e:
            print(f"[LocalProvider] EXCEPTION: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()
            return ""


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
        self.model = SentenceTransformer(
            config.embedding_model,
            device=config.embedding_device,
            local_files_only=config.embedding_local_files_only,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, batch_size=self.config.embedding_batch_size)
        return vectors
