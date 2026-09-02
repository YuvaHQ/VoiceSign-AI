"""
src/ingestion/asl_adapter.py
----------------------------
ASL dataset adapter (WLASL JSON import).
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional
from src.config import ASL_DATA_DIR, TOTAL_FRAME_FEATURES
from src.ingestion.base_adapter import BaseDatasetAdapter
from src.models.schemas import SignLanguageEnum


class ASLDatasetAdapter(BaseDatasetAdapter):
    def __init__(self, target_dir: Optional[Path] = None):
        super().__init__(target_dir=target_dir or ASL_DATA_DIR, language=SignLanguageEnum.ASL)

    def ingest_wlasl_json(self, json_path: Path) -> Dict[str, Any]:
        if not json_path.exists():
            return {'success': False, 'error': f'File not found: {json_path}'}
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        imported = 0
        for entry in data:
            gloss = entry.get('gloss', entry.get('word', 'Unknown'))
            instances = entry.get('instances', [entry])
            for inst in instances:
                if 'frames' in inst and isinstance(inst['frames'], list) and len(inst['frames']) > 1:
                    ok, _ = self.save_dynamic_sample(inst['frames'], gloss)
                    if ok: imported += 1
                elif 'landmarks' in inst:
                    ok, _ = self.save_static_sample(inst['landmarks'], gloss)
                    if ok: imported += 1
                elif 'frames' in inst and isinstance(inst['frames'], list) and len(inst['frames']) == TOTAL_FRAME_FEATURES:
                    ok, _ = self.save_static_sample(inst['frames'], gloss)
                    if ok: imported += 1
        return {'success': True, 'imported': imported, 'total_imported': imported}