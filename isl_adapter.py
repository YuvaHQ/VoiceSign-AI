"""
src/ingestion/isl_adapter.py
----------------------------
ISL dataset adapter (INCLUDE metadata JSON import).
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional
from src.config import ISL_DATA_DIR, TOTAL_FRAME_FEATURES
from src.ingestion.base_adapter import BaseDatasetAdapter
from src.models.schemas import SignLanguageEnum


class ISLDatasetAdapter(BaseDatasetAdapter):
    def __init__(self, target_dir: Optional[Path] = None):
        super().__init__(target_dir=target_dir or ISL_DATA_DIR, language=SignLanguageEnum.ISL)

    def ingest_include_metadata_json(self, json_path: Path) -> Dict[str, Any]:
        if not json_path.exists():
            return {'success': False, 'error': f'File not found: {json_path}'}
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        imported = 0
        for item in data:
            word = item.get('word', item.get('sign', 'Unknown'))
            frames = item.get('frames', item.get('landmarks', []))
            if isinstance(frames, list) and frames and isinstance(frames[0], list):
                ok, _ = self.save_dynamic_sample(frames, word)
                if ok: imported += 1
            elif isinstance(frames, list) and len(frames) == TOTAL_FRAME_FEATURES:
                ok, _ = self.save_static_sample(frames, word)
                if ok: imported += 1
        return {'success': True, 'imported': imported, 'total_imported': imported}