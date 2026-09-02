"""
src/models/database.py
----------------------
SQLite database initialization and persistence layer for custom sign profiles and recordings.
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from src.config import CUSTOM_DATA_DIR, DB_PATH
from src.models.schemas import CustomSign, CustomSignSample, CustomSignSampleInput, SampleTypeEnum


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with row factory configured."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Initialize SQLite database tables if they do not exist."""
    with get_db_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS custom_signs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            label TEXT NOT NULL,
            language TEXT NOT NULL DEFAULT 'CUSTOM',
            description TEXT DEFAULT '',
            sample_count INTEGER DEFAULT 0,
            sample_type_summary TEXT DEFAULT 'mixed',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS custom_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sign_id TEXT NOT NULL,
            sample_type TEXT NOT NULL,
            features_json TEXT,
            frames_json TEXT,
            motion_energy REAL DEFAULT 0.0,
            created_at REAL NOT NULL,
            FOREIGN KEY(sign_id) REFERENCES custom_signs(id) ON DELETE CASCADE
        )
        """)
        conn.commit()


def create_custom_sign(sign_id: str, user_id: str, label: str, description: str,
                       samples: List[CustomSignSampleInput]) -> CustomSign:
    """Creates a new custom sign record with associated landmark samples."""
    init_db()
    now = time.time()
    st_types = set([s.sample_type.value for s in samples])
    summary = list(st_types)[0] if len(st_types) == 1 else 'mixed'
    sample_objects: List[CustomSignSample] = []

    with get_db_connection() as conn:
        conn.execute("""
        INSERT INTO custom_signs (id, user_id, label, language, description, sample_count, sample_type_summary, created_at, updated_at)
        VALUES (?, ?, ?, 'CUSTOM', ?, ?, ?, ?, ?)
        """, (sign_id, user_id, label, description, len(samples), summary, now, now))

        for s in samples:
            sample_id = str(uuid.uuid4())
            feats_json = json.dumps(s.features) if s.features is not None else None
            frames_json = json.dumps(s.frames) if s.frames is not None else None
            conn.execute("""
            INSERT INTO custom_samples (sign_id, sample_type, features_json, frames_json, motion_energy, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (sign_id, s.sample_type.value, feats_json, frames_json, s.motion_energy, now))
            sample_objects.append(CustomSignSample(
                sample_id=sample_id,
                sample_type=s.sample_type,
                features=s.features,
                frames=s.frames,
                motion_energy=s.motion_energy,
                created_at=str(now)
            ))

        conn.commit()

    return CustomSign(
        id=sign_id,
        user_id=user_id,
        label=label,
        language='CUSTOM',
        description=description,
        sample_type_summary=summary,
        samples=sample_objects,
        created_at=now,
        updated_at=now,
        sample_count=len(samples),
        is_trained=False
    )


def list_custom_signs(user_id: Optional[str] = None) -> List[CustomSign]:
    """List custom signs, optionally filtered by user_id."""
    init_db()
    signs = []
    with get_db_connection() as conn:
        query = "SELECT * FROM custom_signs"
        params = ()
        if user_id is not None:
            query += " WHERE user_id = ?"
            params = (user_id,)
        rows = conn.execute(query, params).fetchall()

        for row in rows:
            sign_id = row['id']
            sample_rows = conn.execute("SELECT * FROM custom_samples WHERE sign_id = ?", (sign_id,)).fetchall()
            samples = [
                CustomSignSample(
                    sample_id=str(s['id']),
                    sample_type=SampleTypeEnum(s['sample_type']),
                    features=json.loads(s['features_json']) if s['features_json'] else None,
                    frames=json.loads(s['frames_json']) if s['frames_json'] else None,
                    motion_energy=s['motion_energy'],
                    created_at=str(s['created_at'])
                ) for s in sample_rows
            ]
            signs.append(CustomSign(
                id=sign_id,
                user_id=row['user_id'],
                label=row['label'],
                language=row['language'] if 'language' in row.keys() else 'CUSTOM',
                description=row['description'],
                sample_type_summary=row['sample_type_summary'] if 'sample_type_summary' in row.keys() else 'mixed',
                samples=samples,
                created_at=row['created_at'],
                updated_at=row['updated_at'],
                sample_count=row['sample_count'],
                is_trained=False
            ))
    return signs


def get_custom_sign(sign_id: str) -> Optional[CustomSign]:
    """Retrieve a single custom sign by ID."""
    init_db()
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM custom_signs WHERE id = ?", (sign_id,)).fetchone()
        if not row:
            return None
        sample_rows = conn.execute("SELECT * FROM custom_samples WHERE sign_id = ?", (sign_id,)).fetchall()
        samples = [
            CustomSignSample(
                sample_id=str(s['id']),
                sample_type=SampleTypeEnum(s['sample_type']),
                features=json.loads(s['features_json']) if s['features_json'] else None,
                frames=json.loads(s['frames_json']) if s['frames_json'] else None,
                motion_energy=s['motion_energy'],
                created_at=str(s['created_at'])
            ) for s in sample_rows
        ]
        return CustomSign(
            id=row['id'],
            user_id=row['user_id'],
            label=row['label'],
            language=row['language'] if 'language' in row.keys() else 'CUSTOM',
            description=row['description'],
            sample_type_summary=row['sample_type_summary'] if 'sample_type_summary' in row.keys() else 'mixed',
            samples=samples,
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            sample_count=row['sample_count'],
            is_trained=False
        )


def update_custom_sign(sign_id: str, label: Optional[str] = None, description: Optional[str] = None) -> Optional[CustomSign]:
    """Update custom sign label and/or description."""
    init_db()
    sign = get_custom_sign(sign_id)
    if not sign:
        return None

    new_label = label if label is not None else sign.label
    new_desc = description if description is not None else sign.description
    now = time.time()

    with get_db_connection() as conn:
        conn.execute("""
        UPDATE custom_signs SET label = ?, description = ?, updated_at = ? WHERE id = ?
        """, (new_label, new_desc, now, sign_id))
        conn.commit()

    return get_custom_sign(sign_id)


def add_samples_to_custom_sign(sign_id: str, new_samples: List[CustomSignSampleInput]) -> Optional[CustomSign]:
    """Appends new sample recordings to an existing custom sign."""
    init_db()
    sign = get_custom_sign(sign_id)
    if not sign:
        return None

    now = time.time()
    with get_db_connection() as conn:
        for s in new_samples:
            feats_json = json.dumps(s.features) if s.features is not None else None
            frames_json = json.dumps(s.frames) if s.frames is not None else None
            conn.execute("""
            INSERT INTO custom_samples (sign_id, sample_type, features_json, frames_json, motion_energy, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (sign_id, s.sample_type.value, feats_json, frames_json, s.motion_energy, now))

        new_count = sign.sample_count + len(new_samples)
        conn.execute("""
        UPDATE custom_signs SET sample_count = ?, updated_at = ? WHERE id = ?
        """, (new_count, now, sign_id))
        conn.commit()

    return get_custom_sign(sign_id)


def delete_custom_sign(sign_id: str) -> bool:
    """Deletes a custom sign and all associated samples."""
    init_db()
    with get_db_connection() as conn:
        res = conn.execute("DELETE FROM custom_signs WHERE id = ?", (sign_id,))
        conn.execute("DELETE FROM custom_samples WHERE sign_id = ?", (sign_id,))
        conn.commit()
        return res.rowcount > 0