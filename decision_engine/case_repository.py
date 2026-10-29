import sqlite3
import json
import uuid
from datetime import datetime
import os

class CaseRepository:
    """Manages multi-user cases and judge evaluations using SQLite."""
    
    def __init__(self, db_path="decision_cases.db"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    evaluation_type TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    judge_name TEXT NOT NULL,
                    evaluation_json TEXT NOT NULL,
                    submitted_at TEXT NOT NULL,
                    FOREIGN KEY (case_id) REFERENCES cases (case_id)
                )
            """)
            conn.commit()

    def create_case(self, title, evaluation_type, config_data, description=""):
        """Creates a new evaluation case and returns its unique case_id."""
        case_id = str(uuid.uuid4())[:8]  # short readable 8-character ID
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO cases (case_id, title, description, evaluation_type, config_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (case_id, title, description, evaluation_type, json.dumps(config_data), now)
            )
            conn.commit()
        return case_id

    def get_case(self, case_id):
        """Retrieves case metadata and configuration by case_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "case_id": row["case_id"],
                "title": row["title"],
                "description": row["description"],
                "evaluation_type": row["evaluation_type"],
                "config": json.loads(row["config_json"]),
                "created_at": row["created_at"]
            }

    def list_cases(self):
        """Lists all existing cases ordered by creation date."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT case_id, title, evaluation_type, created_at FROM cases ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def submit_evaluation(self, case_id, judge_name, evaluation_data):
        """Stores or updates an evaluation from a judge."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Check if this judge already submitted for this case
            cursor.execute(
                "SELECT id FROM evaluations WHERE case_id = ? AND judge_name = ?",
                (case_id, judge_name)
            )
            existing = cursor.fetchone()
            if existing:
                cursor.execute(
                    """
                    UPDATE evaluations
                    SET evaluation_json = ?, submitted_at = ?
                    WHERE id = ?
                    """,
                    (json.dumps(evaluation_data), now, existing["id"])
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO evaluations (case_id, judge_name, evaluation_json, submitted_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (case_id, judge_name, json.dumps(evaluation_data), now)
                )
            conn.commit()
        return True

    def get_case_evaluations(self, case_id):
        """Retrieves all submitted evaluations for a case."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT judge_name, evaluation_json, submitted_at FROM evaluations WHERE case_id = ? ORDER BY submitted_at ASC",
                (case_id,)
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "judge_name": r["judge_name"],
                    "evaluation": json.loads(r["evaluation_json"]),
                    "submitted_at": r["submitted_at"]
                })
            return results
