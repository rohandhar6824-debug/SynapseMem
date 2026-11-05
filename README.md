<div align="center">

# ⚡ SynapseMem

**A zero-dependency, temporal belief-graph memory engine designed to eliminate chronological amnesia and context bloat in local edge language models (SLMs).**

[![PyPI version](https://img.shields.io/pypi/v/synapsemem-ai.svg?color=blue)](https://pypi.org/project/synapsemem-ai/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23970380.svg)](https://doi.org/10.5281/zenodo.23970380)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hardware](https://img.shields.io/badge/Hardware-Edge%20%2F%20CPU%20Friendly-purple)](#-benchmark-results)
[![Ollama Compatible](https://img.shields.io/badge/Ollama-Native-black)](https://ollama.ai)

📄 **Preprint:** [Read the Research Paper (PDF)](paper/SynapseMem_Preprint.pdf) • [Zenodo Record](https://doi.org/10.5281/zenodo.23970380)

<br/>

[Architecture](#-architecture) • [Key Innovations](#-key-innovations) • [Benchmarks](#-benchmark-results) • [Quickstart](#-quickstart) • [Ollama Integration](#-local-agent-integration-ollama) • [Citation](#-citation)

</div>

---

## 🔬 The Core Problem: Chronological Amnesia in RAG

When an autonomous AI agent operates over days, weeks, or months, environment states drift constantly:
- **Day 1:** `"The production database runs on port 5432."`
- **Day 5:** `"We migrated the production database to port 6432."`

Standard **Vector RAG** (Retrieval-Augmented Generation) converts text into static embeddings and retrieves chunks based purely on semantic similarity. Because both statements discuss database ports, standard cosine similarity retrieves **both** chunks with near-identical scores. 

Small Language Models (SLMs, e.g., 1B–3B parameter models running on edge CPUs) cannot reliably resolve this contradiction. They frequently hallucinate obsolete values, compromise between the two, or slow down dramatically as conversation logs balloon context windows at an $O(N)$ rate.

---

## 🏗️ Architecture

SynapseMem replaces unstructured vector dumps with an embedded, deterministic temporal belief graph backed by standard SQLite:

```text
User / Sensor Input
       │
       ▼
┌─────────────────────────┐
│   CQRS Query Gater      │ ──[ Interrogative ]──► Read-Only Context Retrieval
└──────────┬──────────────┘
           │ (Declarative Fact)
           ▼
┌─────────────────────────┐
│ Semantic Singularity    │ ──► Predicate conflict detected?
│       Gate              │     ├── YES: Invalidate old state (active=0, set forward pointer)
└──────────┬──────────────┘     └── NO:  Append / update belief
           ▼
┌─────────────────────────┐
│   Ebbinghaus Retention  │ ──► Recalculate decay R(t) = exp(-Δt · λ / S)
│       Engine            │     Prune entropy below threshold τ
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│ Partitioned Injection   │ ──► Inject: [ACTIVE MEMORY] vs [RESOLVED PAST CHANGES]
└─────────────────────────┘
```

## 💡 Key Innovations

1. **Semantic Singularity Gate:** Stateful predicates (`uses_*`, `has_*`, `is_*`, `deployed_*`) are mutually exclusive. When a new belief arrives for a singular slot, previous values are deterministically marked inactive and archived without requiring secondary LLM arbitration passes.
2. **Ebbinghaus Memory Decay:** Memory retention decays over time via $R(t) = \exp(-\Delta t \cdot \lambda / S)$, while repeated references reinforce synaptic strength $S$. Transient noise fades automatically while core operational parameters remain consolidated.
3. **Partitioned Prompt Injection:** Structured prompt sections strictly separate current operational truth (`### VERIFIED ACTIVE MEMORY`) from superseded context (`### RESOLVED STATE CHANGES (DO NOT USE OLD VALUES)`).
4. **CQRS Prompt Gating:** Interrogative inquiries (questions) are routed through a read-only path, preventing user queries from corrupting or mutating state representations.
5. **Zero External Dependencies:** Built purely on Python standard library modules (`sqlite3`, `math`, `re`, `time`). No Docker, vector databases, or external services required.

---

## 📊 Benchmark Results

Evaluated locally against `qwen2.5:3b` on CPU via Ollama across sequential state-inversion tasks:

| Metric | Naive History Baseline | SynapseMem (Ours) | Delta |
| :--- | :--- | :--- | :--- |
| **State-Inversion Accuracy** | 100%* | **100%** | Parity |
| **Injected Context Footprint** | 455 words | **234 words** | **-48.6%** |
| **Average Query Latency (CPU)** | 0.36 s | **0.15 s** | **~58% Faster** |
| **Context Scaling Curve** | $O(N)$ Linear Bloat | **$O(1)$ Constant Ceiling** | **Bounded** |

*\*Note: Baseline accuracy collapses rapidly as conversation length exceeds the context boundary, while SynapseMem maintains a constant ~25-word context footprint per entity inquiry.*

---

## 🚀 Quickstart

### Installation

```bash
pip install synapsemem-ai
```

### Basic Usage

```python
from synapsemem import SynapseMemoryEngine

# Initialize memory engine with local SQLite storage
memory = SynapseMemoryEngine(db_path="agent_memory.db")

# Ingest initial belief
memory.ingest_belief(
    subject="auth_service",
    predicate="uses_port",
    obj="8000"
)

# Later: state changes (automatically archives port 8000)
memory.ingest_belief(
    subject="auth_service",
    predicate="uses_port",
    obj="9090"
)

# Retrieve deterministic, conflict-free context
context = memory.retrieve_active_context(subject="auth_service")
print(context)
```

### Local Agent Integration (Ollama)

```python
from synapsemem import SynapseMemoryEngine, OllamaBridge

# Connect SynapseMem to your local Ollama instance
bridge = OllamaBridge(
    model="qwen2.5:3b",
    db_path="local_agent.db"
)

# Ingest state change and execute prompt with O(1) bounded memory
response = bridge.chat("We migrated payment_gateway to AWS us-east-1.")
print(response)

response = bridge.chat("Where is payment_gateway deployed?")
print(response)
```

## 📜 Citation

If you use SynapseMem in your research or applications, please cite the preprint:

```bibtex
@article{dhar2026synapsemem,
  title={SynapseMem: Deterministic Temporal Belief Invalidation and Ebbinghaus Memory Consolidation for Edge Language Models},
  author={Dhar, Rohan},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.23970380},
  url={[https://doi.org/10.5281/zenodo.23970380](https://doi.org/10.5281/zenodo.23970380)}
}
```

## 📄 License
Distributed under the MIT License. 