"""PRISM loading. The public dataset has no uniq_id; it is assigned here."""

from __future__ import annotations

import os
from collections import defaultdict

import pandas as pd
from datasets import DatasetDict, load_dataset, load_from_disk

from epipersona.paths import data_root

HF_DATASET = "MichaelR207/prism_personalized_0125"
DISK_NAME = "prism_personalized_0125"
DROP_COLUMNS = [
    "chosen_score",
    "rejected_score",
    "dataset",
    "query_reasoning",
    "personalizable_query",
    "response_reasoning",
    "personalizable_responses",
    "reasoning_gemini/gemini-1.5-pro",
    "prediction_flip_gemini/gemini-1.5-pro",
    "reasoning_flip_gemini/gemini-1.5-pro",
    "reasoning_meta-llama/Llama-3.3-70B-Instruct",
    "prediction_flip_meta-llama/Llama-3.3-70B-Instruct",
    "reasoning_flip_meta-llama/Llama-3.3-70B-Instruct",
    "prediction_gemini/gemini-1.5-pro",
    "prediction_meta-llama/Llama-3.3-70B-Instruct",
    "prediction_azure/gpt-4o-mini-240718",
    "reasoning_azure/gpt-4o-mini-240718",
    "prediction_flip_azure/gpt-4o-mini-240718",
    "reasoning_flip_azure/gpt-4o-mini-240718",
    "prediction_Qwen/Qwen2.5-72B-Instruct",
    "reasoning_Qwen/Qwen2.5-72B-Instruct",
    "prediction_flip_Qwen/Qwen2.5-72B-Instruct",
    "reasoning_flip_Qwen/Qwen2.5-72B-Instruct",
    "prediction_meta-llama/Llama-3.1-70B-Instruct",
    "reasoning_meta-llama/Llama-3.1-70B-Instruct",
    "prediction_flip_meta-llama/Llama-3.1-70B-Instruct",
    "reasoning_flip_meta-llama/Llama-3.1-70B-Instruct",
    "agreement",
]


def prism_disk_dir():
    return data_root() / DISK_NAME


def assign_uniq_id(hf_split: str, conversation_id, user_id, row_index: int) -> str:
    return f"{hf_split}.{conversation_id}.{user_id}.{row_index}"


def _with_uniq_ids(dataset_dict: DatasetDict) -> DatasetDict:
    mapped = {}
    for split_name, split in dataset_dict.items():

        def _add(row, index, split_name=split_name):
            row["uniq_id"] = assign_uniq_id(
                split_name, row["conversation_id"], row["user_id"], index
            )
            return row

        mapped[split_name] = split.map(_add, with_indices=True)
    return DatasetDict(mapped)


def download_prism() -> str:
    """Download the Hugging Face dataset and store it with uniq_id columns."""
    destination = prism_disk_dir()
    if (destination / "dataset_dict.json").is_file():
        print(f"PRISM cache already exists: {destination}")
        return str(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    dataset = load_dataset(HF_DATASET, token=token)
    expected = {"train", "validation", "test"}
    if set(dataset.keys()) != expected:
        raise RuntimeError(f"Unexpected PRISM splits: {list(dataset.keys())}")
    dataset = _with_uniq_ids(dataset)
    dataset.save_to_disk(str(destination))
    counts = {name: len(dataset[name]) for name in ("train", "validation", "test")}
    print(f"Saved PRISM to {destination} ({counts}, total {sum(counts.values())})")
    return str(destination)


class PrismDataset:
    """All Hugging Face partitions concatenated; the `split` column is the label split."""

    def __init__(self):
        path = prism_disk_dir()
        if not (path / "dataset_dict.json").is_file():
            raise FileNotFoundError(
                f"PRISM is not prepared at {path}. Run scripts/prepare_prism.sh first."
            )
        dataset = load_from_disk(str(path))
        present = [column for column in DROP_COLUMNS if column in dataset["train"].column_names]
        if present:
            dataset = dataset.remove_columns(present)
        self.dataset = dataset
        frames = [dataset[name].to_pandas() for name in ("train", "validation", "test")]
        self.frame = pd.concat(frames, ignore_index=True)
        if "uniq_id" not in self.frame.columns:
            raise RuntimeError("Prepared PRISM is missing uniq_id")

    def get_all_user_data(self) -> dict:
        grouped = {}
        for user_id, group in self.frame.groupby("user_id", sort=True):
            grouped[user_id] = (
                group[group["split"] == "train"].to_dict(orient="records"),
                group[group["split"] == "val"].to_dict(orient="records"),
                group[group["split"] == "test"].to_dict(orient="records"),
            )
        return grouped

    def get_user_ids(self) -> list:
        return list(pd.unique(self.frame["user_id"]))

    def rows_for_user(self, user_id: str) -> list:
        group = self.frame[self.frame["user_id"] == user_id]
        ordered = []
        for label in ("train", "val", "test"):
            ordered.extend(group[group["split"] == label].to_dict(orient="records"))
        return ordered

    def get_key2pair(self) -> dict:
        return {row["uniq_id"]: row for row in self.frame.to_dict(orient="records")}
