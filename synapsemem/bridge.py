import json
import re
import time
from typing import Dict, List, Optional

import ollama

try:
    from .engine import SynapseMemoryEngine
except ImportError:
    from engine import SynapseMemoryEngine

DEFAULT_MODEL = "qwen2.5:3b"

INTERROGATIVE_PREFIXES = (
    "what", "when", "where", "which", "who", "whom", "whose",
    "why", "how", "is", "are", "was", "were", "does", "did"
)

DECLARATIVE_OVERRIDES = (
    "update:", "note that", "remember that", "set ", "migrated",
    "changed to", "is now", "are now", "uses port", "runs on"
)


def is_query_only(text: str) -> bool:
    """
    CQRS Gate: Returns True if input is purely an interrogative query,
    preventing read-only questions from mutating memory state.
    """
    cleaned = text.strip().lower()
    if not cleaned:
        return True

    # If the user explicitly includes declarative update phrases, allow extraction
    if any(marker in cleaned for marker in DECLARATIVE_OVERRIDES):
        return False

    if cleaned.endswith("?"):
        return True

    return any(cleaned.startswith(w + " ") or cleaned == w for w in INTERROGATIVE_PREFIXES)


def extract_triplets(
    user_text: str,
    model: str = DEFAULT_MODEL,
    active_schema: Optional[Dict[str, List[str]]] = None
) -> List[Dict]:
    """Extracts structured (subject, predicate, object, is_singular) facts from declarative statements."""
    schema_hint = ""
    if active_schema:
        existing_subj = ", ".join(active_schema.get("subjects", [])[:20])
        existing_pred = ", ".join(active_schema.get("predicates", [])[:20])
        if existing_subj or existing_pred:
            schema_hint = (
                f"\nEXISTING ENTITIES (reuse exact name if referring to the same entity): [{existing_subj}]\n"
                f"EXISTING PREDICATES (reuse exact name if referring to the same relation): [{existing_pred}]\n"
            )

    system_prompt = (
        "Extract factual declarations from the user text as JSON:\n"
        "{\"facts\": [{\"subject\": \"...\", \"predicate\": \"...\", \"object\": \"...\", \"is_singular\": true}]}\n"
        "RULES:\n"
        "- Only extract facts from clear, declarative statements.\n"
        "- If the input is a question or lacks concrete facts, return {\"facts\": []}.\n"
        "- Use snake_case for subjects (e.g., 'database', 'frontend_framework', 'background_worker') and strip 'the'.\n"
        "- Use short snake_case predicates (e.g., 'uses_port', 'uses_framework', 'deployed_in', 'uses_cache', 'has_status').\n"
        "- Set 'is_singular' to true if the entity can only have ONE active value for this predicate at a time "
        "(e.g., current port, status, active region), or false if it can have multiple at once (e.g., skills, tags)."
        f"{schema_hint}"
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            format="json",
        )
        data = json.loads(response["message"]["content"])

        if isinstance(data, dict):
            for key in ("facts", "triplets", "data", "items"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            if all(k in data for k in ("subject", "predicate", "object")):
                return [data]
            return []
        elif isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"[Extractor Debug] Parse error: {e}")
        return []


class OllamaBridge:
    """Connects SynapseMemoryEngine to a local Ollama model with CQRS routing."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        db_path: str = "synapse_memory.db",
        engine: Optional[SynapseMemoryEngine] = None
    ):
        self.model = model
        self.memory = engine or SynapseMemoryEngine(db_path=db_path)

    def chat(self, user_input: str, now: Optional[float] = None, verbose: bool = False) -> str:
        t_now = now or time.time()
        user_input = user_input.strip()
        if not user_input:
            return ""

        # 1. CQRS Write Path: Only ingest facts on declarative inputs
        if is_query_only(user_input):
            if verbose:
                print("[Read-Only Query: Memory ingestion skipped]")
        else:
            schema = self.memory.get_active_schema()
            triplets = extract_triplets(user_input, model=self.model, active_schema=schema)
            if verbose:
                print(f"[Extracted {len(triplets)} fact(s)]")
            for t in triplets:
                if isinstance(t, dict) and all(k in t for k in ("subject", "predicate", "object")):
                    explicit_sing = t.get("is_singular") if isinstance(t.get("is_singular"), bool) else None
                    status = self.memory.ingest_belief(
                        str(t["subject"]),
                        str(t["predicate"]),
                        str(t["object"]),
                        now=t_now,
                        is_singular=explicit_sing,
                    )
                    if verbose:
                        print("  " + status)

        # 2. CQRS Read Path: Single pass stopword-filtered context retrieval
        injected_context = self.memory.retrieve_active_context(user_input, now=t_now)

        # 3. Prompt Construction
        if "(None)" not in injected_context:
            prompt = (
                f"Use this verified system memory to answer the user query.\n\n"
                f"{injected_context}\n\n"
                f"User Input: {user_input}"
            )
        else:
            prompt = user_input

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an accurate technical assistant. Rely strictly on the "
                        "provided verified system memory for factual questions."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"].strip()


def chat_with_memory(model: str = DEFAULT_MODEL, db_path: str = "synapse_memory.db") -> None:
    bridge = OllamaBridge(model=model, db_path=db_path)
    print(f"SynapseMem Agent Online using {model}. Type 'exit' to quit.\n" + "=" * 55)

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            break

        print("\n[Processing with SynapseMem...]")
        reply = bridge.chat(user_input, verbose=True)
        print(f"\nAgent: {reply}")


if __name__ == "__main__":
    chat_with_memory()