import ollama
import json
import time
import re
from synapse_core import SynapseMemoryEngine

memory = SynapseMemoryEngine()
MODEL = "qwen2.5:3b"

QUESTION_STARTERS = (
    "what", "when", "where", "which", "who", "whom", "whose", 
    "why", "how", "is", "are", "was", "were", "do", "does", 
    "did", "can", "could", "should", "would", "tell me"
)

def is_query_only(text: str) -> bool:
    """Detects if an input is purely a question to avoid state poisoning."""
    cleaned = text.strip().lower()
    if cleaned.endswith("?"):
        return True
    return any(cleaned.startswith(w + " ") or cleaned == w for w in QUESTION_STARTERS)

def extract_triplets(user_text: str) -> list:
    """Extracts facts only from declarative statements."""
    system_prompt = (
        "Extract new factual declarations from the user text as JSON: {\"facts\": [{\"subject\": ..., \"predicate\": ..., \"object\": ...}]}.\n"
        "RULES:\n"
        "- Only extract facts if the user is making a clear, declarative statement.\n"
        "- If the text is a question, query, or lacks concrete statements, return {\"facts\": []}.\n"
        "- Keep predicates short (e.g., 'uses_port', 'has_status')."
    )
    
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_text}
            ],
            format='json'
        )
        data = json.loads(response['message']['content'])
        
        if isinstance(data, dict):
            for key in ['facts', 'triplets', 'data', 'items']:
                if key in data and isinstance(data[key], list):
                    return data[key]
            if 'subject' in data and 'predicate' in data and 'object' in data:
                return [data]
            return []
        elif isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"[Extractor Debug] Parse error: {e}")
        return []

def chat_with_memory():
    print(f"SynapseMem Agent Online using {MODEL}. Type 'exit' to quit.\n" + "="*55)
    
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['exit', 'quit']:
            break
            
        t_now = time.time()
        
        # 1. State Mutation Gate: Only extract if it is NOT a read-only query
        if is_query_only(user_input):
            print("[Read-Only Query: Memory ingestion skipped]")
        else:
            triplets = extract_triplets(user_input)
            if triplets:
                print(f"[Extracted {len(triplets)} fact(s)]")
                for t in triplets:
                    if isinstance(t, dict) and all(k in t for k in ['subject', 'predicate', 'object']):
                        print("  " + memory.ingest_belief(str(t['subject']), str(t['predicate']), str(t['object']), now=t_now))
            else:
                print("[Extracted 0 facts]")

        # 2. Context Retrieval (Strip punctuation)
        tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', user_input.lower())
        retrieved_segments = []
        for token in tokens:
            if len(token) >= 3:
                ctx = memory.retrieve_active_context(token, now=t_now)
                if "(None)" not in ctx:
                    retrieved_segments.append(ctx)
                     
        injected_context = "\n\n".join(sorted(set(retrieved_segments)))
        
        # 3. Prompt Construction
        if injected_context:
            prompt = (
                f"Use this verified system memory to answer the user query.\n\n"
                f"{injected_context}\n\n"
                f"User Question: {user_input}"
            )
        else:
            prompt = user_input
            
        print("\n[Thinking with Context...]")
        response = ollama.chat(
            model=MODEL,
            messages=[
                {
                    'role': 'system', 
                    'content': "You are an accurate technical assistant. Rely strictly on the provided verified system memory for factual questions."
                },
                {'role': 'user', 'content': prompt}
            ]
        )
        
        print(f"\nAgent: {response['message']['content']}")

if __name__ == "__main__":
    chat_with_memory()