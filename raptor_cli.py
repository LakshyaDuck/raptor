import argparse
import sys

from config import RaptorConfig
from embedder import Embedder
from eval import EvalPipeline
from models import RaptorTree
from provider import AnthropicProvider, LocalProvider
from query_router import QueryRouter
from raptor import Raptor
from summarizer import Summarizer
from tree_builder import TreeBuilder


def build_raptor(config: RaptorConfig) -> Raptor:
    local_provider = LocalProvider(config)
    eval_provider = AnthropicProvider(config)
    embedder = Embedder(config)
    summarizer = Summarizer(config, local_provider)
    tree_builder = TreeBuilder(config, embedder, summarizer)
    query_router = QueryRouter(config, RaptorTree(), embedder, local_provider)
    eval_pipeline = EvalPipeline(config, eval_provider)
    return Raptor(
        config,
        embedder,
        summarizer,
        tree_builder,
        query_router,
        eval_pipeline,
        local_provider,
        eval_provider,
    )


def read_document(args: argparse.Namespace) -> list[str]:
    if args.text:
        return [args.text]
    if args.file:
        with open(args.file, "r", encoding="utf-8") as file:
            return [file.read()]
    return [sys.stdin.read()]


def make_config(args: argparse.Namespace) -> RaptorConfig:
    config = RaptorConfig()
    config.embedding_device = args.embedding_device
    config.summarization_model = args.ollama_model
    config.chunk_size = args.chunk_size
    config.chunk_overlap = args.chunk_overlap
    config.max_tree_layers = args.max_tree_layers
    config.retrieval_token_budget = args.retrieval_token_budget
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a RAPTOR ingest/query smoke flow."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--file", help="Path to a text document to ingest.")
    input_group.add_argument("--text", help="Inline text to ingest.")
    parser.add_argument("--query", required=True, help="Question to ask after ingest.")
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the answer with Anthropic."
    )
    parser.add_argument(
        "--embedding-device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for sentence-transformer embeddings.",
    )
    parser.add_argument(
        "--allow-embedding-downloads",
        action="store_true",
        help="Allow Hugging Face network access if the embedding model is not cached.",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen3.5:9b",
        help="Ollama model name for local generation.",
    )
    parser.add_argument("--chunk-size", type=int, default=80)
    parser.add_argument("--chunk-overlap", type=int, default=10)
    parser.add_argument("--max-tree-layers", type=int, default=2)
    parser.add_argument("--retrieval-token-budget", type=int, default=256)
    args = parser.parse_args()

    config = make_config(args)
    config.embedding_local_files_only = not args.allow_embedding_downloads
    raptor = build_raptor(config)
    document = read_document(args)

    print("Ingesting document...")
    raptor.ingest(document)
    print(f"Tree depth: {raptor.tree.depth}")
    print(f"Total nodes: {len(raptor.tree.all_nodes())}")

    print("\nAnswer:")
    answer = raptor.query(args.query)
    print(answer)

    if args.eval:
        print("\nEvaluation:")
        scores = raptor.evaluate(args.query, answer, raptor.tree.all_nodes())
        for name, score in scores.items():
            print(f"{name}: {score:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
