"""
tests/test_custom_signs.py
--------------------------
Automated Python tests for personalized custom signs ("Teach My Sign").
"""

import pytest
from src.custom_signs.manager import CustomSignManager
from src.models.database import init_db
from src.models.schemas import (
    CustomSignCreateRequest,
    CustomSignSampleInput,
    CustomSignUpdateRequest,
    SampleTypeEnum,
)


@pytest.fixture(autouse=True)
def setup_database():
    init_db()


def test_create_custom_sign_validation():
    manager = CustomSignManager(min_samples=3)

    with pytest.raises(Exception):
        manager.create_sign(CustomSignCreateRequest(user_id="user1", label="", samples=[]))

    with pytest.raises(ValueError, match="At least 3 sample recordings are required"):
        manager.create_sign(
            CustomSignCreateRequest(
                user_id="user1",
                label="Need Coffee",
                samples=[CustomSignSampleInput(sample_type=SampleTypeEnum.STATIC, features=[0.5] * 126)],
            )
        )


def test_create_and_retrieve_custom_sign():
    manager = CustomSignManager(min_samples=3)
    samples = [
        CustomSignSampleInput(
            sample_type=SampleTypeEnum.DYNAMIC,
            frames=[[0.5 + f * 0.01] * 126 for f in range(30)],
        )
        for _ in range(3)
    ]

    req = CustomSignCreateRequest(
        user_id="test_user_42",
        label="Emergency Brake",
        description="Two fists clenched gesture",
        samples=samples,
    )

    created = manager.create_sign(req)
    assert created.id is not None
    assert created.label == "Emergency Brake"
    assert created.sample_count == 3
    assert created.language == "CUSTOM"

    retrieved = manager.get_sign(created.id)
    assert retrieved is not None
    assert retrieved.label == "Emergency Brake"
    assert len(retrieved.samples) == 3

    all_signs = manager.list_signs(user_id="test_user_42")
    assert any(s.id == created.id for s in all_signs)

    updated = manager.update_sign(created.id, CustomSignUpdateRequest(label="Emergency Stop"))
    assert updated.label == "Emergency Stop"

    more_samples = [CustomSignSampleInput(sample_type=SampleTypeEnum.STATIC, features=[0.6] * 126)]
    with_more = manager.add_samples(created.id, more_samples)
    assert with_more.sample_count == 4

    deleted = manager.delete_sign(created.id)
    assert deleted is True