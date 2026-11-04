import time
import math
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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
    def __init__(self, db_path: str = "synapse_memory.db", plural_predicates: Optional[set] = None):
        self.db_path = db_path
        self.plural_predicates = plural_predicates or {
            "has_tag", "contains", "includes", "depends_on", "has_member", "features"
        }
        self._init_db()

    def _init_db(self):
        """Initializes the persistent SQLite schema."""
        with sqlite3.connect(self.db_path) as conn:
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
            conn.commit()

    def _is_singular(self, predicate: str) -> bool:
        if predicate in self.plural_predicates:
            return False
        state_prefixes = ("is", "has_", "uses_", "deployed_", "current_", "set_")
        return predicate.startswith(state_prefixes) or predicate.endswith(("_status", "_port", "_id"))

    def ingest_belief(self, subject: str, predicate: str, obj: str, now: Optional[float] = None) -> str:
        now = now or time.time()
        subj_norm = subject.strip().lower()
        pred_norm = predicate.strip().lower()
        obj_norm = obj.strip()
        is_singular = self._is_singular(pred_norm)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check for existing active beliefs
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
                    conn.commit()
                    return f"[REINFORCED] ({subj_norm}, {pred_norm}, {existing_obj}) | Strength: {new_strength:.1f}"
                
                # Supersede old value if singular
                elif is_singular:
                    cursor.execute("""
                        UPDATE beliefs SET is_active = 0, superseded_by = ? WHERE id = ?
                    """, (obj_norm, row_id))

            # Insert new active belief
            cursor.execute("""
                INSERT INTO beliefs (subject, predicate, object, created_at, last_accessed, strength, is_active, superseded_by)
                VALUES (?, ?, ?, ?, ?, 1.0, 1, NULL)
            """, (subj_norm, pred_norm, obj_norm, now, now))
            conn.commit()
            return f"[COMMITTED]  ({subj_norm}, {pred_norm}, {obj_norm})"

    def retrieve_active_context(self, entity_query: str, min_retention: float = 0.2, now: Optional[float] = None) -> str:
        now = now or time.time()
        query_norm = entity_query.strip().lower()
        active_facts = []
        historical_shifts = []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Find any records touching this entity
            cursor.execute("""
                SELECT id, subject, predicate, object, created_at, last_accessed, strength, is_active, superseded_by
                FROM beliefs
                WHERE subject LIKE ? OR object LIKE ?
            """, (f"%{query_norm}%", f"%{query_norm}%"))
            rows = cursor.fetchall()

            for row in rows:
                r_id, subj, pred, obj_val, c_at, l_acc, strength, is_active, superseded = row
                node = BeliefNode(subj, pred, obj_val, c_at, l_acc, strength, bool(is_active), superseded)
                
                if node.is_active:
                    score = node.retention_score(now)
                    if score >= min_retention:
                        # Touch access time in DB
                        cursor.execute("UPDATE beliefs SET last_accessed = ? WHERE id = ?", (now, r_id))
                        active_facts.append(f"- {subj} {pred} {obj_val}")
                elif node.superseded_by:
                    historical_shifts.append(f"- [PAST] {subj} {pred} was '{obj_val}' (now '{node.superseded_by}')")
            conn.commit()

        context_lines = ["### VERIFIED ACTIVE MEMORY:"] + (active_facts or ["(None)"])
        if historical_shifts:
            context_lines += ["\n### RESOLVED STATE CHANGES (DO NOT USE OLD VALUES):"] + historical_shifts[-4:]
        return "\n".join(context_lines)

    def reset_memory(self):
        """Utility to wipe database for clean benchmarking."""
        with sqlite3.connect(self.db_path) as conn:
            conn.cursor().execute("DROP TABLE IF EXISTS beliefs")
        self._init_db()