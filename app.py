import json
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple

import gradio as gr

from config import RaptorConfig
from embedder import Embedder
from eval import EvalPipeline
from provider import AnthropicProvider, LocalProvider
from query_router import QueryRouter
from raptor import Raptor
from summarizer import Summarizer
from tree_builder import TreeBuilder


class RaptorManager:
    """Manages file ingestion, tree building, and querying"""

    def __init__(self):
        self.raptor: Optional[Raptor] = None
        self.current_file_hash: Optional[str] = None
        self.trees_dir = Path("saved_trees")
        self.trees_dir.mkdir(exist_ok=True)

    # ===== FILE EXTRACTION =====

    def extract_text(self, file_path: str) -> Tuple[str, bool]:
        """Extract text from .txt, .pdf, or .docx files"""
        try:
            file_ext = Path(file_path).suffix.lower()

            if file_ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                    return text, True

            elif file_ext == ".pdf":
                try:
                    import PyPDF2

                    with open(file_path, "rb") as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                        return text, True
                except ImportError:
                    return "ERROR: PyPDF2 not installed. Run: pip install PyPDF2", False

            elif file_ext in [".docx", ".doc"]:
                try:
                    from docx import Document

                    doc = Document(file_path)
                    text = "\n".join([para.text for para in doc.paragraphs])
                    return text, True
                except ImportError:
                    return (
                        "ERROR: python-docx not installed. Run: pip install python-docx",
                        False,
                    )

            else:
                return (
                    f"ERROR: Unsupported file type '{file_ext}'. Supported: .txt, .pdf, .docx",
                    False,
                )

        except Exception as e:
            return f"ERROR: Failed to extract text: {e}", False

    def get_file_hash(self, file_path: str) -> str:
        """Generate hash of file for caching"""
        import hashlib

        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()[:12]
        except:
            return hashlib.md5(file_path.encode()).hexdigest()[:12]

    def save_tree(self, raptor: Raptor, file_hash: str, file_name: str) -> bool:
        """Save tree to disk"""
        try:
            tree_path = self.trees_dir / f"tree_{file_hash}.pkl"
            metadata_path = self.trees_dir / f"meta_{file_hash}.json"

            # Save tree
            with open(tree_path, "wb") as f:
                pickle.dump(raptor.tree, f)

            # Save metadata
            tree_depth = raptor.tree.depth if raptor.tree else 0
            total_nodes = len(raptor.tree.all_nodes()) if raptor.tree else 0

            metadata = {
                "file_name": file_name,
                "file_hash": file_hash,
                "tree_depth": tree_depth,
                "total_nodes": total_nodes,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            return True
        except Exception as e:
            print(f"Error saving tree: {e}")
            return False

    def load_tree(self, file_hash: str) -> Optional[object]:
        """Load tree from disk"""
        try:
            tree_path = self.trees_dir / f"tree_{file_hash}.pkl"
            if tree_path.exists():
                with open(tree_path, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            print(f"Error loading tree: {e}")
        return None

    def load_metadata(self, file_hash: str) -> Optional[dict]:
        """Load metadata from disk"""
        try:
            metadata_path = self.trees_dir / f"meta_{file_hash}.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
        return None

    def list_saved_trees(self) -> list:
        """List all saved trees"""
        trees = []
        try:
            for metadata_file in self.trees_dir.glob("meta_*.json"):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    trees.append(metadata)
        except:
            pass
        return trees

    def clear_all_caches(self) -> str:
        """Delete all cached trees"""
        try:
            import shutil

            if self.trees_dir.exists():
                shutil.rmtree(self.trees_dir)
                self.trees_dir.mkdir(exist_ok=True)
                return "✓ All caches cleared successfully"
            return "No caches to clear"
        except Exception as e:
            return f"ERROR: Failed to clear caches: {e}"

    def clear_single_cache(self, file_hash: str) -> str:
        """Delete a single cached tree"""
        try:
            tree_path = self.trees_dir / f"tree_{file_hash}.pkl"
            metadata_path = self.trees_dir / f"meta_{file_hash}.json"

            if tree_path.exists():
                tree_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()

            return "✓ Cache cleared"
        except Exception as e:
            return f"ERROR: {e}"


# Initialize manager
manager = RaptorManager()


# ===== MAIN FUNCTIONS =====


def ingest_and_query(file_obj, query_text: str) -> Tuple[str, str, str, str, str, str]:
    """
    Main function: ingest file, build/load tree, query, return results with timings

    Returns:
    - answer: the query response
    - setup_time: time to initialize components
    - extraction_time: time to extract text
    - ingestion_time: time to build/load tree
    - query_time: time to process query
    - status: ingestion status message
    """

    try:
        # Get the actual file path from Gradio file object
        if file_obj is None:
            return "ERROR: No file selected", "0s", "0s", "0s", "0s", "ERROR: No file"

        file_path = file_obj.name  # This is the actual file path

        # ===== SETUP =====
        tic = time.perf_counter()
        config = RaptorConfig()
        embedder = Embedder(config)
        local_provider = LocalProvider(config)
        anthropic_provider = AnthropicProvider(config)
        summarizer = Summarizer(config, local_provider)
        tree_builder = TreeBuilder(config, embedder, summarizer)
        query_router = QueryRouter(config, embedder, local_provider)
        eval_pipeline = EvalPipeline(config, local_provider)
        tac = time.perf_counter()
        setup_time = f"{tac - tic:.2f}s"

        # ===== EXTRACT TEXT =====
        tic = time.perf_counter()
        text, success = manager.extract_text(file_path)
        if not success:
            return text, setup_time, "0s", "0s", "0s", text
        tac = time.perf_counter()
        extraction_time = f"{tac - tic:.2f}s"

        # ===== CHECK FOR CACHED TREE =====
        file_hash = manager.get_file_hash(file_path)
        file_name = Path(file_path).name

        tic = time.perf_counter()

        # Try loading from cache
        tree = manager.load_tree(file_hash)
        metadata = manager.load_metadata(file_hash)

        if tree and metadata:
            # Use cached tree
            raptor = Raptor(
                config,
                embedder,
                summarizer,
                tree_builder,
                QueryRouter(config, embedder, local_provider),
                eval_pipeline,
                local_provider,
                anthropic_provider,
            )
            raptor.tree = tree
            ingestion_status = f"✓ Loaded cached tree from '{file_name}' (Depth: {metadata['tree_depth']}, Nodes: {metadata['total_nodes']})"
        else:
            # Build new tree
            raptor = Raptor(
                config,
                embedder,
                summarizer,
                tree_builder,
                QueryRouter(config, embedder, local_provider),
                eval_pipeline,
                local_provider,
                anthropic_provider,
            )
            raptor.ingest([text])

            # Save for next time
            manager.save_tree(raptor, file_hash, file_name)

            tree_depth = raptor.tree.depth if raptor.tree else 0
            total_nodes = len(raptor.tree.all_nodes()) if raptor.tree else 0
            ingestion_status = f"✓ Built new tree for '{file_name}' (Depth: {tree_depth}, Nodes: {total_nodes})"

        tac = time.perf_counter()
        ingestion_time = f"{tac - tic:.2f}s"

        # ===== QUERY =====
        tic = time.perf_counter()
        answer = raptor.query(query_text)
        tac = time.perf_counter()
        query_time = f"{tac - tic:.2f}s"

        answer_str = str(answer) if answer else "No answer generated"

        return (
            answer_str,
            setup_time,
            extraction_time,
            ingestion_time,
            query_time,
            ingestion_status,
        )

    except Exception as e:
        import traceback

        error_msg = f"ERROR: {e}\n{traceback.format_exc()}"
        return error_msg, "0s", "0s", "0s", "0s", error_msg


def get_saved_trees_display() -> str:
    """Get formatted list of saved trees"""
    trees = manager.list_saved_trees()

    if not trees:
        return "No saved trees yet."

    md = "## Cached Trees\n\n"
    for i, tree in enumerate(trees, 1):
        md += f"**{i}. {tree['file_name']}**\n"
        md += f"- Depth: {tree['tree_depth']}\n"
        md += f"- Nodes: {tree['total_nodes']}\n\n"

    return md


def clear_caches_callback() -> str:
    """Clear all caches and return status"""
    result = manager.clear_all_caches()
    return result


# ===== GRADIO INTERFACE =====

with gr.Blocks(title="RAPTOR Document QA") as demo:
    gr.Markdown("# RAPTOR Document QA System")
    gr.Markdown("Upload a document and ask questions. Trees are cached for reuse.")

    with gr.Tabs():
        # Tab 1: Query
        with gr.Tab("Query Document"):
            with gr.Column():
                gr.Markdown("### Upload & Query")
                gr.Markdown("1. Upload a document (.txt, .pdf, or .docx)")
                gr.Markdown("2. Ask your question")

                file_input = gr.File(
                    label="Upload Document", file_types=[".txt", ".pdf", ".docx"]
                )
                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about the document...",
                    lines=3,
                )
                query_btn = gr.Button("Query", variant="primary", size="lg")

                answer_output = gr.Textbox(label="Answer", interactive=False, lines=6)

                with gr.Row():
                    setup_output = gr.Textbox(label="Setup Time", interactive=False)
                    extract_output = gr.Textbox(label="Extract Time", interactive=False)
                    ingest_output = gr.Textbox(label="Ingest Time", interactive=False)
                    query_time_output = gr.Textbox(
                        label="Query Time", interactive=False
                    )

                status_output = gr.Textbox(label="Status", interactive=False, lines=2)

                query_btn.click(
                    ingest_and_query,
                    inputs=[file_input, query_input],
                    outputs=[
                        answer_output,
                        setup_output,
                        extract_output,
                        ingest_output,
                        query_time_output,
                        status_output,
                    ],
                )

        # Tab 2: Saved Trees
        with gr.Tab("Saved Trees"):
            with gr.Column():
                gr.Markdown("### Cached Trees")
                gr.Markdown(
                    "Trees are automatically cached after first ingestion for fast reuse."
                )

                with gr.Row():
                    refresh_btn = gr.Button("Refresh", variant="secondary")
                    clear_btn = gr.Button("Clear All Caches", variant="stop")

                saved_display = gr.Markdown()
                clear_status = gr.Textbox(label="Status", interactive=False)

                refresh_btn.click(get_saved_trees_display, outputs=[saved_display])
                clear_btn.click(clear_caches_callback, outputs=[clear_status])
                demo.load(get_saved_trees_display, outputs=[saved_display])


if __name__ == "__main__":
    demo.launch()
