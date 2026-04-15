import re

from config import RaptorConfig
from models import Node


def split_sentences(text: str, max_words: int) -> list[str]:
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    # Break long sentences
    result = []
    for sentence in sentences:
        words = sentence.split()
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i : i + max_words])
            result.append(chunk)

    return result


def accumulate_sentences(sentences: list[str], size: int, overlap: int) -> list[str]:
    chunks = []
    current_chunk_words = []

    for sentence in sentences:
        sentence_words = sentence.split()

        if len(current_chunk_words) + len(sentence_words) <= size:
            current_chunk_words.extend(sentence_words)
        else:
            chunks.append(" ".join(current_chunk_words))

            overlap_start = max(0, len(current_chunk_words) - overlap)
            overlap_words = current_chunk_words[overlap_start:]

            current_chunk_words = overlap_words + sentence_words

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks


def chunk_text(text: str, config: RaptorConfig, metadata: dict = {}) -> list[Node]:
    sentences = split_sentences(text, config.chunk_size)
    chunks = accumulate_sentences(sentences, config.chunk_size, config.chunk_overlap)
    nodes = []
    for chunk in chunks:
        nodes.append(
            Node(
                text=chunk, layer=0, children_ids=[], embedding=None, metadata=metadata
            )
        )
    return nodes
