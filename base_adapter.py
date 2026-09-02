"""
src/ingestion/base_adapter.py
-----------------------------
Base dataset adapter handling sample persistence, deduplication, and CSV importing.
"""
import csv
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import uuid
import numpy as np

from src.config import SEQUENCE_LENGTH, TOTAL_FRAME_FEATURES
from src.landmarks.normalizer import normalize_landmarks, validate_landmarks
from src.landmarks.sequence import compute_motion_energy, create_dynamic_sequence, create_static_sample, resample_sequence
from src.models.schemas import DynamicSignSample, SampleMetadata, SampleTypeEnum, SignLanguageEnum, StaticSignSample


class BaseDatasetAdapter:
    def __init__(self, target_dir: Optional[Path] = None, language: SignLanguageEnum = SignLanguageEnum.ASL):
        self.data_dir = Path(target_dir) if target_dir is not None else Path('data') / language.value.lower()
        self.language = language
        self.static_file = self.data_dir / 'static_samples.jsonl'
        self.dynamic_file = self.data_dir / 'dynamic_samples.jsonl'
        self.meta_file = self.data_dir / 'metadata.json'
        self._seen_hashes: Set[str] = set()
        self._init_storage()

    def _init_storage(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.static_file.exists():
            with open(self.static_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            if 'features' in d:
                                self._seen_hashes.add(self._hash_sample(d['features']))
                        except Exception: pass
        if self.dynamic_file.exists():
            with open(self.dynamic_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            if 'frames' in d:
                                self._seen_hashes.add(self._hash_sample(d['frames']))
                        except Exception: pass

    def _hash_sample(self, features: Union[List[float], List[List[float]]]) -> str:
        arr = np.asarray(features, dtype=np.float32).round(4)
        return hashlib.md5(arr.tobytes()).hexdigest()

    def save_static_sample(self, features_126: List[float], label: str) -> Tuple[bool, str]:
        if not label or not label.strip():
            return False, 'Missing or empty label'
        ok, msg = validate_landmarks(features_126)
        if not ok:
            return False, msg

        h = self._hash_sample(features_126)
        if h in self._seen_hashes:
            return False, f"Duplicate static sample rejected for '{label}'"
        self._seen_hashes.add(h)

        sample = create_static_sample(features_126, label=label, language=self.language)
        with open(self.static_file, 'a', encoding='utf-8') as f:
            f.write(sample.model_dump_json() + chr(10))

        sample_id = str(uuid.uuid4())
        return True, sample_id

    def save_dynamic_sample(self, frames: List[List[float]], label: str) -> Tuple[bool, str]:
        if not label or not label.strip():
            return False, 'Missing or empty label'
        if not frames:
            return False, 'No frames provided'

        h = self._hash_sample(frames)
        if h in self._seen_hashes:
            return False, f"Duplicate dynamic sample rejected for '{label}'"
        self._seen_hashes.add(h)

        sample = create_dynamic_sequence(frames, label=label, language=self.language)
        with open(self.dynamic_file, 'a', encoding='utf-8') as f:
            f.write(sample.model_dump_json() + chr(10))

        sample_id = str(uuid.uuid4())
        return True, sample_id

    def add_static_sample(self, sample: StaticSignSample) -> bool:
        ok, _ = self.save_static_sample(sample.features, sample.label)
        return ok

    def add_dynamic_sample(self, sample: DynamicSignSample) -> bool:
        ok, _ = self.save_dynamic_sample(sample.frames, sample.label)
        return ok

    def get_static_samples(self) -> List[StaticSignSample]:
        if not self.static_file.exists():
            return []
        samples = []
        with open(self.static_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(StaticSignSample.model_validate_json(line))
        return samples

    def get_dynamic_samples(self) -> List[DynamicSignSample]:
        if not self.dynamic_file.exists():
            return []
        samples = []
        with open(self.dynamic_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(DynamicSignSample.model_validate_json(line))
        return samples

    def get_status(self) -> Dict[str, Any]:
        st = self.get_static_samples()
        dyn = self.get_dynamic_samples()
        labels = sorted(list(set([s.label for s in st] + [d.label for d in dyn])))
        st_labels = sorted(list(set([s.label for s in st])))
        dyn_labels = sorted(list(set([d.label for d in dyn])))
        return {
            'language': self.language.value,
            'sample_count': len(st) + len(dyn),
            'static_sample_count': len(st),
            'dynamic_sample_count': len(dyn),
            'distinct_labels_count': len(labels),
            'labels': labels,
            'static_classes': st_labels,
            'dynamic_classes': dyn_labels,
            'last_updated': time.time(),
        }

    def ingest_from_csv(self, csv_path: Path) -> Dict[str, Any]:
        if not csv_path.exists():
            return {'success': False, 'error': f'File not found: {csv_path}'}
        imported = 0
        warnings = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for row in reader:
                if not row: continue
                label = row[-1]
                values = [float(x) for x in row[:-1]]
                if len(values) == TOTAL_FRAME_FEATURES:
                    ok, msg = self.save_static_sample(values, label)
                    if ok: imported += 1
                    else: warnings.append(msg)
                elif len(values) == SEQUENCE_LENGTH * TOTAL_FRAME_FEATURES:
                    frames = [values[i*TOTAL_FRAME_FEATURES:(i+1)*TOTAL_FRAME_FEATURES] for i in range(SEQUENCE_LENGTH)]
                    ok, msg = self.save_dynamic_sample(frames, label)
                    if ok: imported += 1
                    else: warnings.append(msg)
        return {'success': True, 'total_imported': imported, 'warnings': warnings}