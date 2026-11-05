import os
import sys
from pathlib import Path

# Ensure local package is importable when running directly from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synapsemem import SynapseMemoryEngine


def simulate_scale():
    db_file = "stress_test.db"
    if os.path.exists(db_file):
        os.remove(db_file)

    engine = SynapseMemoryEngine(db_path=db_file)

    print("\n" + "=" * 65)
    print("      SYNAPSEMEM SCALING TEST: CONSTANT CONTEXT VS LINEAR BLOAT")
    print("=" * 65)
    print(f"{'Turn':<6} | {'Simulated Action':<28} | {'Naive Words':<12} | {'Synapse Words':<12}")
    print("-" * 65)

    naive_history = []
    services = ["api_gateway", "payment_worker", "auth_proxy", "search_indexer", "cache_cluster"]
    ports = [8000, 3000, 4000, 9200, 6379]

    turn = 1
    # Round 1: Initial deployment
    for svc, port in zip(services, ports):
        msg = f"The {svc} is initialized on port {port}."
        naive_history.append(f"User: {msg}")
        engine.ingest_belief(svc, "uses_port", str(port))

        ctx = engine.retrieve_active_context(subject=svc)
        naive_len = sum(len(m.split()) for m in naive_history)
        syn_len = len(ctx.split())
        print(f"{turn:<6} | Deploy {svc:<21} | {naive_len:<12} | {syn_len:<12}")
        turn += 1

    # Round 2: Re-configurations & state shifts (5 updates)
    for svc, port in zip(services, [8080, 3001, 4001, 9201, 6380]):
        msg = f"Update: Migration complete. The {svc} now uses port {port}."
        naive_history.append(f"User: {msg}")
        engine.ingest_belief(svc, "uses_port", str(port))

        ctx = engine.retrieve_active_context(subject=svc)
        naive_len = sum(len(m.split()) for m in naive_history)
        syn_len = len(ctx.split())
        print(f"{turn:<6} | Shift {svc:<22} | {naive_len:<12} | {syn_len:<12}")
        turn += 1

    print("=" * 65)
    print("Takeaway: Naive context balloons linearly; SynapseMem stays flat.")


if __name__ == "__main__":
    simulate_scale()