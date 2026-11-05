import os
import sys
import time
from pathlib import Path

import ollama

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synapsemem import SynapseMemoryEngine, extract_triplets

MODEL = "qwen2.5:3b"

# Generate a chaotic sequence of 50+ turns with conflicting updates and noise
CHAOTIC_SCENARIO = [
    # Round 1: Initial Deployments
    ("The primary database runs on port 5432.", False),
    ("The auth service uses port 8000.", False),
    ("The caching layer is Redis.", False),
    ("The active cloud region is us-east-1.", False),
    ("The frontend framework is React.", False),
    
    # Noise / Distractors
    ("The weather outside is sunny today.", False),
    ("Developer Rohan drank coffee at 10 AM.", False),
    ("The team meeting is scheduled for Friday.", False),
    
    # Round 2: First Wave of Migrations
    ("Update: The primary database migrated to port 6432.", False),
    ("Update: The auth service now uses port 9000.", False),
    ("Update: The caching layer is switched to Memcached.", False),
    ("Update: The active cloud region shifted to eu-west-1.", False),
    ("Update: The frontend framework is now Vue.", False),
    
    # More Noise
    ("Someone brought donuts to the office.", False),
    ("The build pipeline finished in 4 minutes.", False),
    
    # Round 3: Second Wave of Rapid Migrations (Stress Test)
    ("Update: Wait, change database port back to 5432 for legacy support.", False),
    ("Update: The auth service moved to port 4000.", False),
    ("Update: We are using Redis again for the cache layer.", False),
    ("Update: Cloud region moved to ap-south-1.", False),
    ("Update: Frontend framework updated to Svelte.", False),
    
    # Final Verification Queries & Expected Truths (The Ultimate Test)
    ("What port is the primary database running on right now?", True, "5432"),
    ("Which port does the auth service use currently?", True, "4000"),
    ("What is our active caching layer right now?", True, "redis"),
    ("Which cloud region is active?", True, "ap-south-1"),
    ("What is our current frontend framework?", True, "svelte"),
]

def run_stress_test():
    print("\n" + "="*70)
    print("      SYNAPSEMEM STRESS-TO-FAILURE & CHAOS RESILIENCE TEST")
    print("="*70)
    
    db_path = "stress_failure.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    engine = SynapseMemoryEngine(db_path=db_path)
    t_clock = time.time()
    
    results = []
    query_count = 0
    correct_count = 0
    
    for idx, item in enumerate(CHAOTIC_SCENARIO, 1):
        text = item[0]
        is_query = item[1]
        t_clock += 30  # Step time forward
        
        if not is_query:
            # Ingestion phase
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
                        is_singular=explicit_sing
                    )
            print(f"[Turn {idx:02d}] Ingested background noise/update.")
        else:
            # Evaluation query phase
            query_count += 1
            expected = item[2]
            
            injected_context = engine.retrieve_active_context(text, now=t_clock)
            prompt = (
                f"Use this verified system state:\n{injected_context}\n\n"
                f"Question: {text}\nAnswer with only the direct answer word/value:"
            )
            
            resp = ollama.chat(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "Rely strictly on the verified system state."},
                    {"role": "user", "content": prompt}
                ]
            )
            output = resp["message"]["content"].strip()
            is_correct = expected.lower() in output.lower()
            if is_correct:
                correct_count += 1
                
            status = "PASS" if is_correct else "FAIL"
            print(f"[Query {query_count}] Expected: '{expected}' | Model Output: '{output}' [{status}]")
            results.append({"correct": is_correct, "expected": expected, "output": output})

    print("="*70)
    print(f"STRESS TEST SUMMARY: {correct_count}/{query_count} Passed ({(correct_count/query_count)*100:.1f}%)")
    print("="*70)
    
    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    run_stress_test()