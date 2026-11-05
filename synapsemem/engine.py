import math
import re
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Set

STOPWORDS: Set[str] = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "can", "could", "should", "would", "will", "shall", "may", "might",
    "the", "and", "for", "with", "from", "into", "onto", "upon", "about",
    "above", "below", "under", "over", "after", "before", "between",
    "now", "right", "currently", "current", "active", "using", "used",
    "tell", "please", "show", "give", "status", "port", "value", "state"
}


@dataclass
class BeliefNode:
    subject: str
    predicate: str
    obj: str
    created_at: float
    last_accessed: float
    strength: float = 1.0
    is_active: bool = True
    superseded_by: Optional[str] = None

    def retention_score(self, current_time: float, decay_rate: float = 0.05) -> float:
        if not self.is_active:
            return 0.0
        elapsed_hours = max(0.0, (current_time - self.last_accessed) / 3600.0)
        effective_stability = self.strength / decay_rate
        return math.exp(-elapsed_hours / effective_stability)


class SynapseMemoryEngine:
    def __init__(self, db_path: str = "synapse_memory.db", plural_predicates: Optional[Set[str]] = None):
        self.db_path = db_path
        self.plural_predicates: Set[str] = plural_predicates or {
            "has_tag", "has_tags", "contains", "includes", "depends_on",
            "has_member", "has_members", "features", "has_skill", "has_skills",
            "has_role", "has_roles", "supports", "connects_to", "has_dependency"
        }
        self._init_db()

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Opens a SQLite connection, commits on success, and explicitly closes the file handle (Windows-safe)."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initializes the persistent SQLite schema."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS beliefs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_accessed REAL NOT NULL,
                    strength REAL NOT NULL,
                    is_active INTEGER NOT NULL,
                    superseded_by TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_subj_pred ON beliefs(subject, predicate)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_active ON beliefs(is_active)")

    @staticmethod
    def normalize_entity(text: str) -> str:
        """Canonicalizes entity names by stripping leading articles and normalizing spaces."""
        cleaned = text.strip().lower()
        cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned)
        cleaned = re.sub(r"[\s\-]+", "_", cleaned)
        cleaned = re.sub(r"[^\w_]", "", cleaned)
        return cleaned or text.strip().lower()

    @staticmethod
    def normalize_predicate(predicate: str) -> str:
        """Canonicalizes predicate strings into snake_case."""
        cleaned = predicate.strip().lower()
        cleaned = re.sub(r"[\s\-]+", "_", cleaned)
        cleaned = re.sub(r"[^\w_]", "", cleaned)
        return cleaned or predicate.strip().lower()

    def _is_singular(self, predicate: str, explicit_singular: Optional[bool] = None) -> bool:
        """Determines whether a predicate represents a mutually exclusive (1:1) state."""
        if explicit_singular is not None:
            return explicit_singular

        if predicate in self.plural_predicates:
            return False

        plural_suffixes = ("_tags", "_skills", "_members", "_items", "_roles", "_list", "_dependencies")
        if predicate.endswith(plural_suffixes):
            return False

        singular_prefixes = (
            "is_", "is", "has_", "uses_", "deployed_", "current_",
            "set_", "runs_on", "located_in", "hosted_on", "listening_on"
        )
        singular_suffixes = (
            "_status", "_port", "_id", "_region", "_version", "_state",
            "_framework", "_cache", "_host", "_url", "_mode", "_tier", "_env"
        )
        return predicate.startswith(singular_prefixes) or predicate.endswith(singular_suffixes)

    def get_active_schema(self) -> Dict[str, List[str]]:
        """Returns distinct active subjects and predicates to ground LLM extraction."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT subject FROM beliefs WHERE is_active = 1 LIMIT 50")
            subjects = [row[0] for row in cursor.fetchall()]
            cursor.execute("SELECT DISTINCT predicate FROM beliefs WHERE is_active = 1 LIMIT 50")
            predicates = [row[0] for row in cursor.fetchall()]
        return {"subjects": subjects, "predicates": predicates}

    def ingest_belief(
        self,
        subject: str,
        predicate: str,
        obj: str,
        now: Optional[float] = None,
        is_singular: Optional[bool] = None
    ) -> str:
        now = now or time.time()
        subj_norm = self.normalize_entity(subject)
        pred_norm = self.normalize_predicate(predicate)
        obj_norm = str(obj).strip()
        singular = self._is_singular(pred_norm, explicit_singular=is_singular)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, object, strength FROM beliefs
                WHERE subject = ? AND predicate = ? AND is_active = 1
            """, (subj_norm, pred_norm))
            active_rows = cursor.fetchall()

            for row_id, existing_obj, strength in active_rows:
                # Reinforce exact match
                if existing_obj.lower() == obj_norm.lower():
                    new_strength = strength + 1.0
                    cursor.execute("""
                        UPDATE beliefs SET last_accessed = ?, strength = ? WHERE id = ?
                    """, (now, new_strength, row_id))
                    return f"[REINFORCED] ({subj_norm}, {pred_norm}, {existing_obj}) | Strength: {new_strength:.1f}"

                # Supersede old value if singular (1:1 state)
                elif singular:
                    cursor.execute("""
                        UPDATE beliefs SET is_active = 0, superseded_by = ? WHERE id = ?
                    """, (obj_norm, row_id))

            cursor.execute("""
                INSERT INTO beliefs (subject, predicate, object, created_at, last_accessed, strength, is_active, superseded_by)
                VALUES (?, ?, ?, ?, ?, 1.0, 1, NULL)
            """, (subj_norm, pred_norm, obj_norm, now, now))
            return f"[COMMITTED]  ({subj_norm}, {pred_norm}, {obj_norm})"

    def retrieve_active_context(
        self,
        entity_query: Optional[str] = None,
        subject: Optional[str] = None,
        min_retention: float = 0.2,
        now: Optional[float] = None
    ) -> str:
        """
        Retrieves active and superseded beliefs matching an entity name or natural language query.
        Supports both `entity_query=...` and `subject=...`.
        """
        now = now or time.time()
        raw_query = (entity_query if entity_query is not None else subject) or ""
        raw_query = raw_query.strip().lower()
        if not raw_query:
            return "### VERIFIED ACTIVE MEMORY:\n(None)"

        norm_full = self.normalize_entity(raw_query)
        raw_tokens = [
            t for t in re.findall(r"\b[a-z0-9_]+\b", raw_query)
            if len(t) >= 3 and t not in STOPWORDS
        ]
        candidate_terms = set(raw_tokens)
        if norm_full and norm_full not in STOPWORDS:
            candidate_terms.add(norm_full)

        if not candidate_terms:
            return "### VERIFIED ACTIVE MEMORY:\n(None)"

        active_facts: List[str] = []
        historical_shifts: List[str] = []
        touched_ids: List[int] = []

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, subject, predicate, object, created_at, last_accessed, strength, is_active, superseded_by
                FROM beliefs
            """)
            rows = cursor.fetchall()

            for row in rows:
                r_id, subj, pred, obj_val, c_at, l_acc, strength, is_active, superseded = row
                subj_tokens = set(subj.lower().split("_")) | {subj.lower()}
                obj_tokens = set(re.findall(r"\b[a-z0-9_]+\b", obj_val.lower())) | {obj_val.lower()}
                pred_tokens = set(pred.lower().split("_"))

                matched = any(
                    term in subj_tokens
                    or term in obj_tokens
                    or term in pred_tokens
                    or (len(term) >= 4 and term in subj.lower())
                    for term in candidate_terms
                )
                if not matched:
                    continue

                node = BeliefNode(subj, pred, obj_val, c_at, l_acc, strength, bool(is_active), superseded)
                if node.is_active:
                    score = node.retention_score(now)
                    if score >= min_retention:
                        touched_ids.append(r_id)
                        active_facts.append(f"- {subj} {pred} {obj_val}")
                elif node.superseded_by:
                    historical_shifts.append(f"- [PAST] {subj} {pred} was '{obj_val}' (now '{node.superseded_by}')")

            if touched_ids:
                placeholders = ",".join("?" for _ in touched_ids)
                cursor.execute(
                    f"UPDATE beliefs SET last_accessed = ? WHERE id IN ({placeholders})",
                    [now, *touched_ids]
                )

        context_lines = ["### VERIFIED ACTIVE MEMORY:"] + (active_facts or ["(None)"])
        if historical_shifts:
            context_lines += ["\n### RESOLVED STATE CHANGES (DO NOT USE OLD VALUES):"] + historical_shifts[-4:]
        return "\n".join(context_lines)

    def reset_memory(self) -> None:
        """Wipes the database for clean benchmarking."""
        with self._connect() as conn:
            conn.cursor().execute("DROP TABLE IF EXISTS beliefs")
        self._init_db()