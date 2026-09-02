"""
src/ingestion/dataset_manager.py
--------------------------------
Orchestrates multi-language dataset adapters (ASL, ISL, BSL, and CUSTOM).
"""
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
from src.config import ASL_DATA_DIR, BSL_DATA_DIR, CUSTOM_DATA_DIR, ISL_DATA_DIR
from src.ingestion.asl_adapter import ASLDatasetAdapter
from src.ingestion.bsl_adapter import BSLDatasetAdapter
from src.ingestion.isl_adapter import ISLDatasetAdapter
from src.models.database import list_custom_signs
from src.models.schemas import DatasetStatus, DatasetsStatusResponse, SignLanguageEnum


class DatasetManager:
    def __init__(self, data_dir: Optional[Path] = None):
        self.asl_adapter = ASLDatasetAdapter(data_dir / 'asl' if data_dir else ASL_DATA_DIR)
        self.isl_adapter = ISLDatasetAdapter(data_dir / 'isl' if data_dir else ISL_DATA_DIR)
        self.bsl_adapter = BSLDatasetAdapter(data_dir / 'bsl' if data_dir else BSL_DATA_DIR)
        self.adapters = {
            SignLanguageEnum.ASL: self.asl_adapter,
            SignLanguageEnum.ISL: self.isl_adapter,
            SignLanguageEnum.BSL: self.bsl_adapter,
        }

    def get_adapter(self, language: SignLanguageEnum):
        return self.adapters.get(language)

    def get_dataset_status(self, language: SignLanguageEnum) -> DatasetStatus:
        if language == SignLanguageEnum.CUSTOM:
            custom_signs = list_custom_signs()
            c_st_classes = set()
            c_dyn_classes = set()
            c_st_count = 0
            c_dyn_count = 0
            for s in custom_signs:
                for sample in s.samples:
                    if sample.sample_type.value == 'static':
                        c_st_count += 1
                        c_st_classes.add(s.label)
                    else:
                        c_dyn_count += 1
                        c_dyn_classes.add(s.label)
            all_labels = sorted(list(c_st_classes.union(c_dyn_classes)))
            return DatasetStatus(
                language=SignLanguageEnum.CUSTOM,
                sample_count=c_st_count + c_dyn_count,
                static_samples_count=c_st_count,
                dynamic_samples_count=c_dyn_count,
                distinct_labels_count=len(all_labels),
                labels=all_labels,
                static_classes=sorted(list(c_st_classes)),
                dynamic_classes=sorted(list(c_dyn_classes)),
                last_updated=str(time.time()),
            )
        adapter = self.get_adapter(language)
        if not adapter:
            raise ValueError(f'Unknown language: {language}')
        st = adapter.get_status()
        return DatasetStatus(
            language=language,
            sample_count=st['sample_count'],
            static_samples_count=st['static_sample_count'],
            dynamic_samples_count=st['dynamic_sample_count'],
            distinct_labels_count=st['distinct_labels_count'],
            labels=st['labels'],
            static_classes=st['static_classes'],
            dynamic_classes=st['dynamic_classes'],
            last_updated=str(st['last_updated']),
            metadata=st,
        )

    def get_all_statuses(self) -> DatasetsStatusResponse:
        datasets = {}
        total = 0
        for lang in [SignLanguageEnum.ASL, SignLanguageEnum.ISL, SignLanguageEnum.BSL, SignLanguageEnum.CUSTOM]:
            status = self.get_dataset_status(lang)
            total += status.sample_count
            datasets[lang.value] = status
        return DatasetsStatusResponse(datasets=datasets, total_samples=total)


_GLOBAL_DATASET_MANAGER: Optional[DatasetManager] = None


def get_dataset_manager() -> DatasetManager:
    global _GLOBAL_DATASET_MANAGER
    if _GLOBAL_DATASET_MANAGER is None:
        _GLOBAL_DATASET_MANAGER = DatasetManager()
    return _GLOBAL_DATASET_MANAGER
