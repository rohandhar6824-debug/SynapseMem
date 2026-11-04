import time
import json
import ollama
import re
from synapse_core import SynapseMemoryEngine

MODEL = "qwen2.5:3b"

# Test scenario: 5 entities undergoing state mutations over time
TEST_SCENARIO = [
    # Initial Facts
    ("The database port is 5432.", False),
    ("The frontend framework is React.", False),
    ("The active deployment region is us-east-1.", False),
    ("The primary cache layer is Redis.", False),
    ("The background worker status is idle.", False),
    # State Updates (Mutations)
    ("Update: The database port is 6432 now.", False),
    ("Update: The frontend framework is Vue now.", False),
    ("Update: The active deployment region is ap-south-1 now.", False),
    ("Update: The primary cache layer is Memcached now.", False),
    ("Update: The background worker status is processing now.", False),
    # Evaluation Queries & Ground Truths
    ("What port is the database using right now?", True, "6432"),
    ("What is the current frontend framework?", True, "Vue"),
    ("Which region is active right now?", True, "ap-south-1"),
    ("What cache layer is currently active?", True, "Memcached"),
    ("What is the current status of the background worker?", True, "processing"),
]

def run_naive_baseline():
    """Simulates standard memory: dumps full raw conversation history into context."""
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
            resp = ollama.chat(model=MODEL, messages=[{'role': 'user', 'content': prompt}])
            latency = time.time() - t0
            
            output = resp['message']['content'].strip()
            is_correct = expected.lower() in output.lower()
            results.append({"correct": is_correct, "latency": latency, "output": output, "expected": expected})

    return results, total_tokens_approx, (time.time() - start_total_time)


def run_synapsemem_pipeline():
    """Runs through SynapseMemoryEngine with conflict resolution."""
    print("\n[2/2] Running SynapseMem Pipeline...")
    engine = SynapseMemoryEngine()
    results = []
    start_total_time = time.time()
    total_tokens_approx = 0

    # Strict triplet extraction for benchmark updates
    extractor_prompt = (
        "Extract declarations into JSON: {\"facts\": [{\"subject\": ..., \"predicate\": ..., \"object\": ...}]}.\n"
        "Normalize predicates to: 'uses_port', 'uses_framework', 'deployed_in', 'uses_cache', 'has_status'."
    )

    t_clock = time.time()

    for item in TEST_SCENARIO:
        text = item[0]
        is_query = item[1]
        t_clock += 60  # Step forward time logically

        if not is_query:
            # Extraction & ingestion
            try:
                resp = ollama.chat(
                    model=MODEL,
                    messages=[
                        {'role': 'system', 'content': extractor_prompt},
                        {'role': 'user', 'content': text}
                    ],
                    format='json'
                )
                data = json.loads(resp['message']['content'])
                facts = data.get('facts', [])
                for f in facts:
                    engine.ingest_belief(f['subject'], f['predicate'], f['object'], now=t_clock)
            except Exception:
                pass
        else:
            expected = item[2]
            tokens = re.findall(r'\b[a-zA-Z0-9_-]+\b', text.lower())
            retrieved = []
            for tok in tokens:
                if len(tok) >= 3:
                    ctx = engine.retrieve_active_context(tok, now=t_clock)
                    if "(None)" not in ctx:
                        retrieved.append(ctx)

            injected_context = "\n".join(sorted(set(retrieved)))
            prompt = (
                f"Use this verified system state:\n{injected_context}\n\n"
                f"Question: {text}\nAnswer with only the direct answer word/value:"
            )
            total_tokens_approx += len(prompt.split())

            t0 = time.time()
            resp = ollama.chat(
                model=MODEL,
                messages=[
                    {'role': 'system', 'content': 'Rely strictly on the verified system state.'},
                    {'role': 'user', 'content': prompt}
                ]
            )
            latency = time.time() - t0

            output = resp['message']['content'].strip()
            is_correct = expected.lower() in output.lower()
            results.append({"correct": is_correct, "latency": latency, "output": output, "expected": expected})

    return results, total_tokens_approx, (time.time() - start_total_time)


def print_comparison(naive_res, syn_res, naive_tokens, syn_tokens):
    n_correct = sum(1 for r in naive_res if r["correct"])
    s_correct = sum(1 for r in syn_res if r["correct"])
    total_q = len(naive_res)

    print("\n" + "="*60)
    print("           BENCHMARK RESULTS: TEMPORAL ACCURACY           ")
    print("="*60)
    print(f"{'Metric':<30} | {'Naive Baseline':<12} | {'SynapseMem':<12}")
    print("-" * 60)
    print(f"{'Accuracy (Correct / Total)':<30} | {f'{n_correct}/{total_q} ({n_correct/total_q*100:.0f}%)':<12} | {f'{s_correct}/{total_q} ({s_correct/total_q*100:.0f}%)':<12}")
    print(f"{'Total Injected Words (Proxy)':<30} | {naive_tokens:<12} | {syn_tokens:<12}")
    
    n_avg_lat = sum(r['latency'] for r in naive_res) / total_q
    s_avg_lat = sum(r['latency'] for r in syn_res) / total_q
    print(f"{'Avg Query Latency (sec)':<30} | {f'{n_avg_lat:.2f}s':<12} | {f'{s_avg_lat:.2f}s':<12}")
    print("="*60)

    print("\nDetail Breakdown (Query vs Expected vs Synapse Output):")
    for idx, (n, s) in enumerate(zip(naive_res, syn_res), 1):
        status = "PASS" if s["correct"] else "FAIL"
        print(f" Q{idx}: Expected '{s['expected']}' | Synapse: '{s['output']}' [{status}] | Naive: '{n['output']}'")


if __name__ == "__main__":
    naive_results, naive_tokens, _ = run_naive_baseline()
    synapse_results, synapse_tokens, _ = run_synapsemem_pipeline()
    print_comparison(naive_results, synapse_results, naive_tokens, synapse_tokens)