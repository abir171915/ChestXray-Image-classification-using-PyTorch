"""Tests for src/data.py — the patient-level split logic.

This is the most important thing in the project to have tests for: the
patient-leakage bug was silent (no error, no crash, just quietly inflated
validation numbers) and only found by manually inspecting filenames. These
tests exist so that kind of bug can't reappear unnoticed.

Uses a small synthetic dataset (empty files with realistic names) under
`tmp_path` rather than the real chest_xray data, so the tests are fast and
don't depend on the dataset being downloaded.
"""

from pathlib import Path

import pytest

from src.data import assert_no_patient_leakage, build_patient_split, patient_id


@pytest.fixture
def fake_train_dir(tmp_path):
    train_dir = tmp_path / "train"
    (train_dir / "NORMAL").mkdir(parents=True)
    (train_dir / "PNEUMONIA").mkdir(parents=True)

    # NORMAL: 10 distinct "patients" (IM-xxxx), one image each
    for i in range(10):
        (train_dir / "NORMAL" / f"IM-{1000 + i}-0001.jpeg").touch()

    # PNEUMONIA: 10 patients, several with 2 images (the leakage-prone case)
    for i in range(10):
        (train_dir / "PNEUMONIA" / f"person{i}_bacteria_1.jpeg").touch()
        if i % 2 == 0:
            (train_dir / "PNEUMONIA" / f"person{i}_virus_2.jpeg").touch()

    return train_dir


def test_patient_id_groups_multiple_images_from_same_pneumonia_patient():
    assert patient_id("person1003_bacteria_2934.jpeg", "PNEUMONIA") == "person1003"
    assert patient_id("person1003_virus_1685.jpeg", "PNEUMONIA") == "person1003"


def test_patient_id_normal_pattern():
    assert patient_id("IM-0115-0001.jpeg", "NORMAL") == "IM-0115"


def test_patient_id_unrecognized_filename_falls_back_to_full_name():
    assert patient_id("weird_filename.jpeg", "PNEUMONIA") == "weird_filename.jpeg"


def test_build_patient_split_produces_no_leakage(tmp_path, fake_train_dir):
    new_train_dir = tmp_path / "new_train"
    new_val_dir = tmp_path / "new_val"

    build_patient_split(fake_train_dir, new_train_dir, new_val_dir, val_fraction=0.3)

    assert_no_patient_leakage(new_train_dir, new_val_dir)  # should not raise


def test_build_patient_split_keeps_all_images(tmp_path, fake_train_dir):
    new_train_dir = tmp_path / "new_train"
    new_val_dir = tmp_path / "new_val"

    build_patient_split(fake_train_dir, new_train_dir, new_val_dir, val_fraction=0.3)

    for cls in ("NORMAL", "PNEUMONIA"):
        original_count = len(list((fake_train_dir / cls).glob("*.jpeg")))
        split_count = len(list((new_train_dir / cls).glob("*.jpeg"))) + len(
            list((new_val_dir / cls).glob("*.jpeg"))
        )
        assert split_count == original_count


def test_build_patient_split_is_deterministic(tmp_path, fake_train_dir):
    new_train_dir = tmp_path / "new_train"
    new_val_dir = tmp_path / "new_val"

    build_patient_split(fake_train_dir, new_train_dir, new_val_dir, val_fraction=0.3, seed=42)
    first_run_val_files = sorted(p.name for p in (new_val_dir / "PNEUMONIA").glob("*.jpeg"))

    build_patient_split(
        fake_train_dir, new_train_dir, new_val_dir, val_fraction=0.3, seed=42, force=True
    )
    second_run_val_files = sorted(p.name for p in (new_val_dir / "PNEUMONIA").glob("*.jpeg"))

    assert first_run_val_files == second_run_val_files


def test_build_patient_split_skips_rebuild_unless_forced(tmp_path, fake_train_dir, capsys):
    new_train_dir = tmp_path / "new_train"
    new_val_dir = tmp_path / "new_val"

    build_patient_split(fake_train_dir, new_train_dir, new_val_dir)
    build_patient_split(fake_train_dir, new_train_dir, new_val_dir)  # should skip, not error

    assert "skipping split" in capsys.readouterr().out


def test_assert_no_patient_leakage_raises_when_a_patient_is_split_across_both(tmp_path):
    new_train_dir = tmp_path / "leaky_train"
    new_val_dir = tmp_path / "leaky_val"
    (new_train_dir / "PNEUMONIA").mkdir(parents=True)
    (new_val_dir / "PNEUMONIA").mkdir(parents=True)
    (new_train_dir / "NORMAL").mkdir(parents=True)
    (new_val_dir / "NORMAL").mkdir(parents=True)

    # Same patient's images placed on both sides on purpose
    (new_train_dir / "PNEUMONIA" / "person1_bacteria_1.jpeg").touch()
    (new_val_dir / "PNEUMONIA" / "person1_virus_2.jpeg").touch()

    with pytest.raises(ValueError):
        assert_no_patient_leakage(new_train_dir, new_val_dir)
