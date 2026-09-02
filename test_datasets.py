"""
tests/test_datasets.py
----------------------
Unit and integration tests for ASL, ISL, BSL dataset adapters.
"""

import json
from pathlib import Path
import pytest
import numpy as np

from src.config import SEQUENCE_LENGTH, TOTAL_FRAME_FEATURES
from src.ingestion.asl_adapter import ASLDatasetAdapter
from src.ingestion.bsl_adapter import BSLDatasetAdapter
from src.ingestion.isl_adapter import ISLDatasetAdapter
from src.ingestion.synthetic_data import generate_synthetic_dataset
from src.models.schemas import SignLanguageEnum


@pytest.fixture
def temp_data_dir(tmp_path):
    return tmp_path / "test_data"


def test_asl_adapter_static_and_dynamic(temp_data_dir):
    adapter = ASLDatasetAdapter(target_dir=temp_data_dir / "asl")

    feats_126 = [0.5] * TOTAL_FRAME_FEATURES
    ok, sample_id = adapter.save_static_sample(feats_126, "Hello")
    assert ok is True
    assert sample_id is not None

    ok_dup, msg = adapter.save_static_sample(feats_126, "Hello")
    assert ok_dup is False
    assert "Duplicate" in msg

    frames = [[0.4 + i * 0.01] * TOTAL_FRAME_FEATURES for i in range(20)]
    ok_dyn, dyn_id = adapter.save_dynamic_sample(frames, "Thank You")
    assert ok_dyn is True
    assert dyn_id is not None

    status = adapter.get_status()
    assert status["sample_count"] == 2
    assert status["static_sample_count"] == 1
    assert status["dynamic_sample_count"] == 1
    assert status["distinct_labels_count"] == 2


def test_isl_adapter_include_json(temp_data_dir):
    adapter = ISLDatasetAdapter(target_dir=temp_data_dir / "isl")

    json_path = temp_data_dir / "include_test.json"
    data = [
        {"word": "Namaste", "frames": [[0.5] * TOTAL_FRAME_FEATURES for _ in range(15)]},
        {"word": "Water", "frames": [0.6] * TOTAL_FRAME_FEATURES}
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    res = adapter.ingest_include_metadata_json(json_path)
    assert res["success"] is True
    assert res["imported"] == 2


def test_bsl_adapter_bsl1k_json(temp_data_dir):
    adapter = BSLDatasetAdapter(target_dir=temp_data_dir / "bsl")

    json_path = temp_data_dir / "bsl1k_test.json"
    data = [{"sign": "Good Morning", "landmarks": [[0.5] * TOTAL_FRAME_FEATURES for _ in range(30)]}]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    res = adapter.ingest_bsl1k_annotations_json(json_path)
    assert res["success"] is True
    assert res["imported"] == 1


def test_malformed_and_missing_labels(temp_data_dir):
    adapter = ASLDatasetAdapter(target_dir=temp_data_dir / "asl")

    ok, msg = adapter.save_static_sample([0.1] * TOTAL_FRAME_FEATURES, "")
    assert ok is False
    assert "Missing" in msg

    ok, msg = adapter.save_static_sample([0.1] * 50, "Invalid")
    assert ok is False
    assert "Invalid landmarks" in msg


def test_synthetic_data_generator():
    res = generate_synthetic_dataset(SignLanguageEnum.ASL, samples_per_class=2, seed=42, overwrite=True)
    assert res["success"] is True
    assert res["static_imported"] > 0
    assert res["dynamic_imported"] > 0