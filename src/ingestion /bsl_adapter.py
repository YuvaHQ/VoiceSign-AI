"""
src/ingestion/bsl_adapter.py
----------------------------
BSL dataset adapter (BSL-1K annotations JSON import).
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional
from src.config import BSL_DATA_DIR, TOTAL_FRAME_FEATURES
from src.ingestion.base_adapter import BaseDatasetAdapter
from src.models.schemas import SignLanguageEnum


class BSLDatasetAdapter(BaseDatasetAdapter):
    def __init__(self, target_dir: Optional[Path] = None):
        super().__init__(target_dir=target_dir or BSL_DATA_DIR, language=SignLanguageEnum.BSL)

    def ingest_bsl1k_annotations_json(self, json_path: Path) -> Dict[str, Any]:
        if not json_path.exists():
            return {'success': False, 'error': f'File not found: {json_path}'}
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        imported = 0
        for item in data:
            sign = item.get('sign', item.get('label', 'Unknown'))
            lms = item.get('landmarks', item.get('frames', []))
            if isinstance(lms, list) and lms and isinstance(lms[0], list):
                ok, _ = self.save_dynamic_sample(lms, sign)
                if ok: imported += 1
            elif isinstance(lms, list) and len(lms) == TOTAL_FRAME_FEATURES:
                ok, _ = self.save_static_sample(lms, sign)
                if ok: imported += 1
        return {'success': True, 'imported': imported, 'total_imported': imported}
