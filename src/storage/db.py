# src/storage/db.py
"""
SQLite-based storage layer for AML workflow.
Manages cases, audit events, and review history with schema versioning.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager
import threading

# Use thread-local storage for SQLite connections (thread-safe)
_thread_local = threading.local()


class DatabaseError(Exception):
    """Custom exception for database operations."""
    pass


class WorkflowDB:
    """
    SQLite database for AML workflow state management.
    
    Tables:
    - cases: Core case records with scoring metadata
    - audit_events: Immutable audit trail of all actions
    - review_history: Manual reviewer decisions and notes
    - schema_version: Schema versioning for migrations
    """
    
    def __init__(self, db_path: str = "data/aml_workflow.db"):
        """Initialize database connection and create tables if needed."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        with self._get_connection() as conn:
            self._initialize_schema(conn)
    
    @contextmanager
    def _get_connection(self):
        """Get thread-safe database connection."""
        if not hasattr(_thread_local, 'connection') or _thread_local.connection is None:
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row  # Return rows as dicts
            _thread_local.connection = conn
        
        try:
            yield _thread_local.connection
        except sqlite3.Error as e:
            raise DatabaseError(f"Database error: {e}")
    
    def _initialize_schema(self, conn: sqlite3.Connection):
        """Create database tables if they don't exist."""
        cursor = conn.cursor()
        
        # Schema version table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY,
                version INTEGER NOT NULL UNIQUE,
                applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Get current version
        cursor.execute("SELECT MAX(version) FROM schema_version")
        result = cursor.fetchone()
        current_version = result[0] if result[0] is not None else 0
        
        # Apply migrations
        if current_version < 1:
            self._migrate_v1(cursor)
            cursor.execute("INSERT INTO schema_version (version) VALUES (1)")
        
        conn.commit()
    
    def _migrate_v1(self, cursor: sqlite3.Cursor):
        """Schema v1: Initial schema with cases, audit, review tables."""
        
        # Cases table
        cursor.execute("""
            CREATE TABLE cases (
                case_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL UNIQUE,
                model_version TEXT NOT NULL,
                threshold_version TEXT NOT NULL,
                score REAL NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'queued_for_review',
                feature_contract_version TEXT NOT NULL,
                raw_features TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT valid_status CHECK (review_status IN (
                    'queued_for_review', 'approved', 'rejected', 'escalated'
                ))
            )
        """)
        
        cursor.execute("""
            CREATE INDEX idx_cases_status ON cases(review_status)
        """)
        
        cursor.execute("""
            CREATE INDEX idx_cases_created_at ON cases(created_at)
        """)
        
        # Audit events table
        cursor.execute("""
            CREATE TABLE audit_events (
                event_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                actor TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE,
                CONSTRAINT valid_event_type CHECK (event_type IN (
                    'SCORE_CREATED', 'REVIEW_SUBMITTED', 'STATUS_CHANGED', 
                    'CASE_ESCALATED', 'CASE_APPROVED', 'CASE_REJECTED'
                ))
            )
        """)
        
        cursor.execute("""
            CREATE INDEX idx_audit_case_id ON audit_events(case_id)
        """)
        
        cursor.execute("""
            CREATE INDEX idx_audit_timestamp ON audit_events(timestamp)
        """)
        
        # Review history table
        cursor.execute("""
            CREATE TABLE review_history (
                review_id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                note TEXT,
                previous_status TEXT,
                new_status TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (case_id) REFERENCES cases(case_id) ON DELETE CASCADE,
                CONSTRAINT valid_decision CHECK (decision IN (
                    'APPROVED', 'REJECTED', 'ESCALATED'
                ))
            )
        """)
        
        cursor.execute("""
            CREATE INDEX idx_review_case_id ON review_history(case_id)
        """)
        
        cursor.execute("""
            CREATE INDEX idx_review_reviewer ON review_history(reviewer_id)
        """)
    
    def create_case(
        self,
        case_id: str,
        request_id: str,
        model_version: str,
        threshold_version: str,
        feature_contract_version: str,
        score: float,
        raw_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Create a new case record.
        
        Args:
            case_id: Unique case identifier (UUID)
            request_id: Unique request identifier (UUID)
            model_version: Version of model used for scoring
            threshold_version: Version of threshold used for decision
            feature_contract_version: Version of feature contract
            score: Numerical risk score [0, 1]
            raw_features: Dictionary of all input features
        
        Returns:
            Dictionary with created case details
        
        Raises:
            DatabaseError: If case already exists or insert fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO cases (
                        case_id, request_id, model_version, threshold_version,
                        feature_contract_version, score, raw_features, review_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued_for_review')
                """, (
                    case_id,
                    request_id,
                    model_version,
                    threshold_version,
                    feature_contract_version,
                    score,
                    json.dumps(raw_features),
                ))
                
                conn.commit()
                
                # Log audit event
                self.log_audit_event(
                    case_id=case_id,
                    event_type="SCORE_CREATED",
                    actor="system",
                    details=json.dumps({
                        "score": score,
                        "model_version": model_version,
                        "threshold_version": threshold_version
                    })
                )
                
                return self.get_case(case_id)
        
        except sqlite3.IntegrityError as e:
            raise DatabaseError(f"Failed to create case: {e}")
    
    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a case by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT case_id, request_id, model_version, threshold_version,
                       feature_contract_version, score, review_status, 
                       created_at, updated_at
                FROM cases WHERE case_id = ?
            """, (case_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            return dict(row)
    
    def update_case_status(
        self,
        case_id: str,
        new_status: str,
    ) -> Dict[str, Any]:
        """
        Update case review status.
        
        Args:
            case_id: Case identifier
            new_status: New status (queued_for_review, approved, rejected, escalated)
        
        Returns:
            Updated case record
        
        Raises:
            DatabaseError: If case not found or status invalid
        """
        valid_statuses = {'queued_for_review', 'approved', 'rejected', 'escalated'}
        if new_status not in valid_statuses:
            raise DatabaseError(f"Invalid status: {new_status}. Must be one of {valid_statuses}")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get current status
                cursor.execute("SELECT review_status FROM cases WHERE case_id = ?", (case_id,))
                result = cursor.fetchone()
                if result is None:
                    raise DatabaseError(f"Case {case_id} not found")
                
                current_status = result[0]
                
                # Update status
                cursor.execute("""
                    UPDATE cases 
                    SET review_status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE case_id = ?
                """, (new_status, case_id))
                
                conn.commit()
                
                # Log audit event
                self.log_audit_event(
                    case_id=case_id,
                    event_type="STATUS_CHANGED",
                    actor="system",
                    details=json.dumps({
                        "previous_status": current_status,
                        "new_status": new_status
                    })
                )
                
                return self.get_case(case_id)
        
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to update case status: {e}")
    
    def log_audit_event(
        self,
        case_id: str,
        event_type: str,
        actor: Optional[str] = None,
        details: Optional[str] = None,
        event_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Create an immutable audit event.
        
        Args:
            case_id: Associated case ID
            event_type: Type of event (e.g., 'SCORE_CREATED', 'REVIEW_SUBMITTED')
            actor: Who triggered the event (user ID or 'system')
            details: JSON string with event metadata
            event_id: UUID for event (generated if not provided)
        
        Returns:
            Dictionary with created event details
        """
        from uuid import uuid4
        
        if event_id is None:
            event_id = str(uuid4())
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO audit_events (
                        event_id, case_id, event_type, actor, details
                    ) VALUES (?, ?, ?, ?, ?)
                """, (event_id, case_id, event_type, actor, details))
                
                conn.commit()
                
                return {
                    "event_id": event_id,
                    "case_id": case_id,
                    "event_type": event_type,
                    "actor": actor,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to log audit event: {e}")
    
    def get_audit_trail(self, case_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all audit events for a case, sorted by timestamp.
        
        Args:
            case_id: Case identifier
        
        Returns:
            List of audit events in chronological order
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT event_id, case_id, event_type, actor, details, timestamp
                FROM audit_events
                WHERE case_id = ?
                ORDER BY timestamp ASC
            """, (case_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def record_review(
        self,
        review_id: str,
        case_id: str,
        reviewer_id: str,
        decision: str,
        note: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Record a manual review decision.
        
        Args:
            review_id: Unique review identifier
            case_id: Associated case ID
            reviewer_id: ID of reviewer
            decision: Decision (APPROVED, REJECTED, ESCALATED)
            note: Optional reviewer note
        
        Returns:
            Dictionary with review record
        
        Raises:
            DatabaseError: If case not found or insert fails
        """
        valid_decisions = {'APPROVED', 'REJECTED', 'ESCALATED'}
        if decision not in valid_decisions:
            raise DatabaseError(f"Invalid decision: {decision}")
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get current status
                cursor.execute("SELECT review_status FROM cases WHERE case_id = ?", (case_id,))
                result = cursor.fetchone()
                if result is None:
                    raise DatabaseError(f"Case {case_id} not found")
                
                previous_status = result[0]
                
                # Map decision to new status
                status_map = {
                    'APPROVED': 'approved',
                    'REJECTED': 'rejected',
                    'ESCALATED': 'escalated'
                }
                new_status = status_map[decision]
                
                # Record review
                cursor.execute("""
                    INSERT INTO review_history (
                        review_id, case_id, reviewer_id, decision, note,
                        previous_status, new_status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (review_id, case_id, reviewer_id, decision, note, 
                      previous_status, new_status))
                
                # Update case status
                cursor.execute("""
                    UPDATE cases
                    SET review_status = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE case_id = ?
                """, (new_status, case_id))
                
                conn.commit()
                
                # Log audit event
                self.log_audit_event(
                    case_id=case_id,
                    event_type="REVIEW_SUBMITTED",
                    actor=reviewer_id,
                    details=json.dumps({
                        "decision": decision,
                        "note": note,
                        "previous_status": previous_status,
                        "new_status": new_status
                    })
                )
                
                return {
                    "review_id": review_id,
                    "case_id": case_id,
                    "reviewer_id": reviewer_id,
                    "decision": decision,
                    "note": note,
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to record review: {e}")
    
    def get_cases_by_status(self, status: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get cases filtered by review status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT case_id, request_id, model_version, score, 
                       review_status, created_at
                FROM cases
                WHERE review_status = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (status, limit))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_case_count_by_status(self) -> Dict[str, int]:
        """Get count of cases by each status."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT review_status, COUNT(*) as count
                FROM cases
                GROUP BY review_status
            """)
            
            rows = cursor.fetchall()
            return {row[0]: row[1] for row in rows}
    
    def close(self):
        """Close database connection."""
        if hasattr(_thread_local, 'connection') and _thread_local.connection:
            _thread_local.connection.close()
            _thread_local.connection = None


# Global database singleton
_db_instance: Optional[WorkflowDB] = None


def get_db() -> WorkflowDB:
    """Get or create global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = WorkflowDB()
    return _db_instance