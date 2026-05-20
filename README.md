RAPTOR

1. **Leaf Construction:** Source texts are parsed, segmented into granular chunks, and embedded using state-of-the-art dense vector models.
2. **Recursive Clustering:** Chunks are clustered using Gaussian Mixture Models (GMMs) with soft clustering capabilities, allowing individual text segments to belong to multiple global themes.
3. **Abstractive Summarization:** A Large Language Model (LLM) generates high-level abstractive summaries for each cluster.
4. **Tree Progression:** Summaries are re-embedded and recursively clustered, creating a multi-tier tree structure until a stable, single root cluster is achieved.

---

## ⚡ Interruptible Deployment Design (Vast.ai Optimization)

Deploying on spot/interruptible instances requires shifting from an in-memory state paradigm to a **transactional, persistent state model**. This implementation addresses this via:

* **Atomic State Serialization:** The system architecture enforces strict boundaries between tree levels. Once a layer finishes clustering and summarizing, its entire state is committed to disk immediately.
* **Idempotent Execution:** The initialization script checks for pre-existing state directories. If an instance is killed mid-run and restarted, the pipeline skips completed layers and seamlessly resumes computing exactly where it left off.
* **External Storage Synchronization (Hooks):** Includes native integration hooks to instantly push serialized components to persistent remote targets (S3, GCP Buckets, or remote SFTP storage) upon layer completion.

---

## 🛠️ Installation & Setup

### Prerequisites
* NVIDIA GPU (Compute Capability 7.0+ recommended for efficient embedding/inference)
* CUDA Toolkit 11.8+ / 12.1+
* Python 3.10+
* Local AI model from Ollama, default model hermes3:8b

### Clone and Install Dependencies
```bash
git clone [https://github.com/LakshyaDuck/raptor.git](https://github.com/LakshyaDuck/raptor.git)
cd raptor

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install core and acceleration dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Start the Web App
python app.py```
