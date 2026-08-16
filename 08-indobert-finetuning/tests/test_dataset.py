# tests/test_dataset.py

import os
import tempfile

import pytest

from src.dataset import LABEL2ID, build_dataset_dict, compute_class_weights, load_raw_df


def _write_tsv(tmpdir, name, rows):
    path = os.path.join(tmpdir, name)
    with open(path, "w") as f:
        for text, label in rows:
            f.write(f"{text}\t{label}\n")
    return path


@pytest.fixture
def fake_data_dir(tmp_path):
    rows = [
        ("bagus sekali", "positive"),
        ("jelek", "negative"),
        ("biasa saja", "neutral"),
        ("sangat buruk", "negative"),
        ("hebat", "positive"),
    ]
    _write_tsv(tmp_path, "train_preprocess.tsv", rows)
    _write_tsv(tmp_path, "valid_preprocess.tsv", rows[:2])
    _write_tsv(tmp_path, "test_preprocess.tsv", rows[:2])
    return str(tmp_path)


def test_load_raw_df_maps_labels(fake_data_dir):
    df = load_raw_df(os.path.join(fake_data_dir, "train_preprocess.tsv"))
    assert set(df["label"].unique()) <= set(LABEL2ID.values())
    assert df["label"].isna().sum() == 0


def test_build_dataset_dict_has_splits(fake_data_dir):
    ds = build_dataset_dict(fake_data_dir)
    assert set(ds.keys()) == {"train", "validation", "test"}
    assert len(ds["train"]) == 5
    assert len(ds["validation"]) == 2


def test_compute_class_weights_normalized(fake_data_dir):
    ds = build_dataset_dict(fake_data_dir)
    weights = compute_class_weights(ds["train"])
    assert len(weights) == len(LABEL2ID)
    assert all(w > 0 for w in weights)
    # minority class gets higher weight
    assert weights[LABEL2ID["neutral"]] > weights[LABEL2ID["positive"]]