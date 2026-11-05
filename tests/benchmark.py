import os
import sys
import time
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synapsemem import SynapseMemoryEngine, extract_triplets

MODEL = "qwen2.5:3b"

# Multi-turn state mutations with realistic operational chatter
TEST_SCENARIO = [
    # Initial Facts (Round 1)
    ("The database port is 5432.", False),
    ("The frontend framework is React.", False),
    ("The active deployment region is us-east-1.", False),
    ("The primary cache layer is Redis.", False),
    ("The background worker status is idle.", False),
    # Intermediate State Shifts (Round 2)
    ("Update: The database port is 5433 for staging tests.", False),
    ("Update: The frontend framework is Angular temporarily.", False),
    ("Update: The active deployment region is eu-west-1 now.", False),
    ("Update: The primary cache layer is Hazelcast now.", False),
    ("Update: The background worker status is warming_up now.", False),
    # Final State Mutations (Round 3)
    ("Update: The database port is 6432 now.", False),
    ("Update: The frontend framework is Vue now.", False),
    ("Update: The active deployment region is ap-south-1 now.", False),
    ("Update: The primary cache layer is Memcached now.", False),
    ("Update: The background worker status is processing now.", False),
    # Evaluation Queries & Ground Truths
    ("What port is the database using right now?", True, "6432"),
    ("What is the current frontend framework?", True, "Vue"),
    ("Which deployment region is active right now?", True, "ap-south-1"),
    ("What primary cache layer is currently active?", True, "Memcached"),
    ("What is the current status of the background worker?", True, "processing"),
]


def run_naive_baseline():
    """Dumps full raw conversation history into context."""
    print("\n[1/2] Running Naive History Baseline...")
    history = []
    results = []
    start_total_time = time.time()
    total_tokens_approx = 0

    for item in TEST_SCENARIO:
        text = item[0]
        is_query = item[1]

        if not is_query:
            history.append(f"User: {text}")
        else:
            expected = item[2]
            prompt = "\n".join(history) + f"\nUser: {text}\nAnswer with only the direct answer word/value:"
            total_tokens_approx += len(prompt.split())

            t0 = time.time()
            resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
            latency = time.time() - t0

            output = resp["message"]["content"].strip()
            is_correct = expected.lower() in output.lower()
            results.append({"correct": is_correct, "latency": latency, "output": output, "expected": expected})

    return results, total_tokens_approx, (time.time() - start_total_time)


def run_synapsemem_pipeline():
    """Runs through SynapseMemoryEngine with schema-grounded extraction and conflict resolution."""
    print("\n[2/2] Running SynapseMem Pipeline...")
    db_path = "benchmark_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    engine = SynapseMemoryEngine(db_path=db_path)
    results = []
    start_total_time = time.time()
    total_tokens_approx = 0
    t_clock = time.time()

    for item in TEST_SCENARIO:
        text = item[0]
        is_query = item[1]
        t_clock += 60

        if not is_query:
            schema = engine.get_active_schema()
            facts = extract_triplets(text, model=MODEL, active_schema=schema)
            for f in facts:
                if isinstance(f, dict) and all(k in f for k in ("subject", "predicate", "object")):
                    explicit_sing = f.get("is_singular") if isinstance(f.get("is_singular"), bool) else None
                    engine.ingest_belief(
                        str(f["subject"]),
                        str(f["predicate"]),
                        str(f["object"]),
                        now=t_clock,
                        is_singular=explicit_sing,
                    )
        else:
            expected = item[2]
            injected_context = engine.retrieve_active_context(text, now=t_clock)
            prompt = (
                f"Use this verified system state:\n{injected_context}\n\n"
                f"Question: {text}\nAnswer with only the direct answer word/value:"
            )
            total_tokens_approx += len(prompt.split())

            t0 = time.time()
            resp = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Rely strictly on the verified system state."},
                    {"role": "user", "content": prompt},
                ],
            )
            latency = time.time() - t0

            output = resp["message"]["content"].strip()
            is_correct = expected.lower() in output.lower()
            results.append({"correct": is_correct, "latency": latency, "output": output, "expected": expected})

    if os.path.exists(db_path):
        os.remove(db_path)

    return results, total_tokens_approx, (time.time() - start_total_time)


def print_comparison(naive_res, syn_res, naive_tokens, syn_tokens):
    n_correct = sum(1 for r in naive_res if r["correct"])
    s_correct = sum(1 for r in syn_res if r["correct"])
    total_q = len(naive_res)

    print("\n" + "=" * 60)
    print("           BENCHMARK RESULTS: TEMPORAL ACCURACY           ")
    print("=" * 60)
    print(f"{'Metric':<30} | {'Naive Baseline':<12} | {'SynapseMem':<12}")
    print("-" * 60)
    print(f"{'Accuracy (Correct / Total)':<30} | {f'{n_correct}/{total_q} ({n_correct/total_q*100:.0f}%)':<12} | {f'{s_correct}/{total_q} ({s_correct/total_q*100:.0f}%)':<12}")
    print(f"{'Total Injected Words (Proxy)':<30} | {naive_tokens:<12} | {syn_tokens:<12}")

    n_avg_lat = sum(r["latency"] for r in naive_res) / total_q
    s_avg_lat = sum(r["latency"] for r in syn_res) / total_q
    print(f"{'Avg Query Latency (sec)':<30} | {f'{n_avg_lat:.2f}s':<12} | {f'{s_avg_lat:.2f}s':<12}")
    print("=" * 60)

    print("\nDetail Breakdown (Query vs Expected vs Synapse Output):")
    for idx, (n, s) in enumerate(zip(naive_res, syn_res), 1):
        status = "PASS" if s["correct"] else "FAIL"
        print(f" Q{idx}: Expected '{s['expected']}' | Synapse: '{s['output']}' [{status}] | Naive: '{n['output']}'")


if __name__ == "__main__":
    naive_results, naive_tokens, _ = run_naive_baseline()
    synapse_results, synapse_tokens, _ = run_synapsemem_pipeline()
    print_comparison(naive_results, synapse_results, naive_tokens, synapse_tokens)