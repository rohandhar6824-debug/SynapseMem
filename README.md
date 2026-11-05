<div align="center">

# ⚡ SynapseMem

**A zero-dependency, temporal belief-graph memory engine designed to eliminate chronological amnesia and context bloat in local edge language models (SLMs).**

[![PyPI version](https://img.shields.io/pypi/v/synapsemem-ai.svg?color=blue)](https://pypi.org/project/synapsemem-ai/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Hardware](https://img.shields.io/badge/Hardware-Edge%20%2F%20CPU%20Friendly-purple)](#-benchmark-results)
[![Ollama Compatible](https://img.shields.io/badge/Ollama-Native-black)](https://ollama.ai)

📄 **Preprint:** [Read the Research Paper (PDF)](paper/SynapseMem_Preprint.pdf)

- [Architecture](#-architecture)
- [Key Innovations](#-key-innovations)
- [Benchmarks](#-benchmark-results)
- [Quickstart](#-quickstart)
- [Ollama Integration](#-local-agent-integration-ollama)
- [Citation](#-citation)

</div>

---

## 🔬 The Core Problem: Chronological Amnesia in RAG

When an autonomous AI agent operates over days, weeks, or months, environment states drift constantly:
- **Day 1:** `"The production database runs on port 5432."`
- **Day 5:** `"We migrated the production database to port 6432."`

Standard **Vector RAG** (Retrieval-Augmented Generation) converts text into static embeddings and retrieves chunks based purely on semantic similarity. Because both statements discuss database ports, standard cosine similarity retrieves **both** chunks with near-identical scores. 

Small Language Models (SLMs, e.g., 1B–3B parameter models running on edge CPUs) cannot reliably resolve this contradiction. They frequently hallucinate obsolete values, compromise between the two, or slow down dramatically as conversation logs balloon context windows at an $O(N)$ rate.