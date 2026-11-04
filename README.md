# ⚡ SynapseMem

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Hardware](https://img.shields.io/badge/Hardware-Edge%20%2F%20CPU%20Friendly-green)](#benchmarks)

**SynapseMem** is a lightweight, zero-dependency, temporal belief-graph memory engine designed to eliminate **chronological amnesia** and context bloat in small, local language models (SLMs).

Traditional Vector RAG dumps conflicting raw chunks into the prompt context, degrading accuracy and slowing down CPU inference. SynapseMem converts interactions into atomic beliefs, deterministically resolves state transitions over time, and injects clean, conflict-free context into edge models.

---

## 🔬 The Problem: Chronological Amnesia in RAG

When an autonomous agent runs over days or weeks, states drift:
* **Monday:** `"The production server is on port 8000."`
* **Friday:** `"We migrated the server to port 9090."`

Standard vector similarity retrieves **both** chunks with equal semantic weight. A 3B parameter model given both inputs often hallucinates stale data.