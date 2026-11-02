"""History embeddings and pairwise response batches for the reward model."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from epipersona.data.prism import PrismDataset
from epipersona.io_utils import load_json, save_json
from epipersona.paths import cache_root, chat_template_path
from epipersona.prompts import SYSTEM_MESSAGE, stable_preference_prompt

CHAT_TEMPLATES = {
    "Meta-Llama-3.1-8B-Instruct": "tool_chat_template_llama3.1_json.jinja",
    "Llama-3.2-3B-Instruct": "tool_chat_template_llama3.2_json.jinja",
}
LLAMA_MODELS = set(CHAT_TEMPLATES)

KNOWN_HISTORY = {
    "train": {"train"},
    "val": {"train"},
    "test": {"train", "val"},
}


def prepare_tokenizer(tokenizer, model_name: str):
    filename = CHAT_TEMPLATES.get(model_name)
    if filename is None:
        raise ValueError(
            f"No chat template is registered for {model_name}."
        )
    if model_name in LLAMA_MODELS:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    template = chat_template_path(filename).read_text(encoding="utf-8")
    tokenizer.chat_template = template.replace("    ", "").replace("\n", "")
    return tokenizer


def kept_ids_path() -> Path:
    return Path(__file__).resolve().parent / "resources" / "prism_row_ids.txt"


def load_kept_ids() -> set[str]:
    """Row ids used for training and evaluation, one uniq_id per line."""
    path = kept_ids_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing row-id list: {path}")
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != 16705 or len(set(ids)) != 16705:
        raise RuntimeError(f"Expected 16705 unique row ids in {path}, found {len(set(ids))}")
    return set(ids)


def generate_embeddings(prompts, model, tokenizer, max_length=1600, device="cuda", batch_size=16):
    model.to(device).eval()
    chunks = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="embeddings", leave=False):
        batch = prompts[start:start + batch_size]
        texts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in batch
        ]
        tokens = tokenizer(
            texts, padding="longest", truncation=True, max_length=max_length, return_tensors="pt"
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden * mask).sum(1) / mask.sum(1)
        chunks.append(pooled.cpu())
    embeddings = torch.cat(chunks, dim=0)
    print(f"embeddings {tuple(embeddings.shape)}")
    return embeddings


class RewardDataset(Dataset):
    def __init__(
        self,
        split: str,
        model_name: str,
        model_path: str,
        belief_name: str,
        tokenizer,
        model,
        window_size: int = 30,
        max_turn: int = 1,
        resp_max_length: int = 1024,
        stat_max_length: int = 1600,
        mask_mode: str = "only_test",
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        self.split = split
        self.model_name = model_name
        self.belief_name = belief_name
        self.window_size = window_size
        self.max_turn = max_turn
        self.resp_max_length = resp_max_length
        self.stat_max_length = stat_max_length
        self.tokenizer = tokenizer
        self.model = model
        self.model_path = model_path
        self.mask_mode = mask_mode
        cache = self._load_or_build()
        self.resp1_input_ids = cache["resp1_input_ids"]
        self.resp1_attn_masks = cache["resp1_attn_masks"]
        self.resp2_input_ids = cache["resp2_input_ids"]
        self.resp2_attn_masks = cache["resp2_attn_masks"]
        self.allocated_embeddings = cache["allocated_embedding"]
        self.allocated_masks = cache["allocated_embedding_attn_mask"]
        self.user_ids = cache["user_id"]
        self.uniq_ids = cache["uniq_id"]

    def __len__(self):
        return len(self.uniq_ids)

    def __getitem__(self, index):
        return {
            "input_ids": self.resp1_input_ids[index],
            "attention_mask": self.resp1_attn_masks[index],
            "resp2_input_ids": self.resp2_input_ids[index],
            "resp2_attention_mask": self.resp2_attn_masks[index],
            "allocated_embedding_stat": self.allocated_embeddings[index],
            "allocated_embedding_stat_mask": self.allocated_masks[index],
            "labels": np.array([1], dtype=np.float32),
        }

    def _prefix(self) -> str:
        stable_suffix = ".pooling"
        return (
            f"prism.{self.split}.{self.model_name}.{self.belief_name}{stable_suffix}.v2"
        )

    def _paths(self):
        processed = cache_root() / "processed_data"
        processed.mkdir(parents=True, exist_ok=True)
        json_path = processed / f"prism.{self.split}.{self.belief_name}.v2.mt{self.max_turn}.json"
        prefix = processed / self._prefix()
        embedding_path = Path_join(prefix, f".embedding.mt{self.max_turn}.pt")
        alloc = (
            f"{prefix}.mm{self.mask_mode}.cs0.rank0.n{self.window_size}.mt{self.max_turn}"
        )
        return {
            "json": json_path,
            "embedding": embedding_path,
            "alloc_emb": Path_join_str(alloc, ".allocated_emb.pt"),
            "alloc_mask": Path_join_str(alloc, ".allocated_mask.pt"),
            "alloc_keep": Path_join_str(alloc, ".allocated_keep.pt"),
        }

    def _load_or_build(self) -> dict:
        paths = self._paths()
        if paths["json"].is_file() and paths["embedding"].is_file():
            payload = load_json(paths["json"])
            payload["embedding_stat_prompt"] = torch.load(paths["embedding"], map_location="cpu")
        else:
            payload = self._build_prompts() if not paths["json"].is_file() else load_json(paths["json"])
            if not paths["json"].is_file():
                save_json({k: v for k, v in payload.items() if k != "embedding_stat_prompt"}, paths["json"])
            if paths["embedding"].is_file():
                payload["embedding_stat_prompt"] = torch.load(paths["embedding"], map_location="cpu")
            else:
                payload["embedding_stat_prompt"] = generate_embeddings(
                    payload["stable_generate_prompt"],
                    model=self.model,
                    tokenizer=self.tokenizer,
                    max_length=self.stat_max_length,
                )
                torch.save(payload["embedding_stat_prompt"], paths["embedding"])
        payload = self._allocate(payload, paths)
        return self._tokenize_responses(payload)

    def _build_prompts(self) -> dict:
        prism = PrismDataset()
        kept = load_kept_ids()
        prompts, user_ids, splits, uniq_ids = [], [], [], []
        missing = 0
        for _user_id, (train_rows, val_rows, test_rows) in prism.get_all_user_data().items():
            for item in train_rows + val_rows + test_rows:
                uniq_id = str(item["uniq_id"])
                if uniq_id not in kept:
                    missing += 1
                    continue
                context = item["context"][-self.max_turn:]
                prompts.append(
                    SYSTEM_MESSAGE
                    + [{
                        "role": "user",
                        "content": stable_preference_prompt(
                            context, item["chosen"], item["rejected"]
                        ),
                    }]
                )
                user_ids.append(str(item["user_id"]))
                splits.append(str(item["split"]))
                uniq_ids.append(uniq_id)
        if missing:
            raise RuntimeError(
                f"{missing} PRISM rows are not listed in {kept_ids_path()}."
            )
        if len(uniq_ids) != 16705:
            raise RuntimeError(f"Expected 16705 PRISM rows before split filtering, found {len(uniq_ids)}")
        return {
            "stable_generate_prompt": prompts,
            "user_id": user_ids,
            "split_inside": splits,
            "uniq_id": uniq_ids,
        }

    def _allocate(self, payload: dict, paths: dict) -> dict:
        if paths["alloc_emb"].is_file() and paths["alloc_mask"].is_file() and paths["alloc_keep"].is_file():
            keep = torch.load(paths["alloc_keep"], map_location="cpu")
            payload = _filter_payload(payload, keep)
            payload["allocated_embedding"] = torch.load(paths["alloc_emb"], map_location="cpu")
            payload["allocated_embedding_attn_mask"] = torch.load(paths["alloc_mask"], map_location="cpu")
            return payload

        target = "val" if self.split == "validation" else self.split
        history_splits = KNOWN_HISTORY[target]
        embeddings = payload["embedding_stat_prompt"]
        user_ids = payload["user_id"]
        inner_splits = payload["split_inside"]
        history_of = defaultdict(set)
        for index, (user_id, inner) in enumerate(zip(user_ids, inner_splits)):
            if inner in history_splits:
                history_of[user_id].add(index)

        keep = []
        selected = []
        for index, (user_id, inner) in enumerate(zip(user_ids, inner_splits)):
            if inner != target:
                continue
            others = sorted(history_of[user_id] - {index})
            if others:
                selected.append(embeddings[others])
                keep.append(index)
        payload = _filter_payload(payload, keep)
        padded, masks = _pad(selected, self.window_size)
        torch.save(padded, paths["alloc_emb"])
        torch.save(masks, paths["alloc_mask"])
        torch.save(keep, paths["alloc_keep"])
        payload["allocated_embedding"] = padded
        payload["allocated_embedding_attn_mask"] = masks
        return payload

    def _tokenize_responses(self, payload: dict) -> dict:
        pairs = PrismDataset().get_key2pair()
        chosen_batch, rejected_batch = [], []
        for uniq_id in payload["uniq_id"]:
            pair = pairs[uniq_id]
            context = pair["context"]
            context = context.tolist() if hasattr(context, "tolist") else list(context)
            context = context[-self.max_turn:]
            chosen_batch.append(context + [pair["chosen"]])
            rejected_batch.append(context + [pair["rejected"]])
        self.tokenizer.padding_side = "right"
        chosen_ids, chosen_mask = _batch_tokenize(
            chosen_batch, self.tokenizer, self.resp_max_length
        )
        rejected_ids, rejected_mask = _batch_tokenize(
            rejected_batch, self.tokenizer, self.resp_max_length
        )
        payload.update({
            "resp1_input_ids": chosen_ids,
            "resp1_attn_masks": chosen_mask,
            "resp2_input_ids": rejected_ids,
            "resp2_attn_masks": rejected_mask,
        })
        return payload


def _filter_payload(payload: dict, keep: list) -> dict:
    filtered = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            filtered[key] = value[keep]
        elif isinstance(value, list):
            filtered[key] = [value[index] for index in keep]
        else:
            filtered[key] = value
    return filtered


def _pad(sequences, length: int):
    if not sequences:
        raise RuntimeError("No samples have usable history for this split.")
    hidden = sequences[0].size(1)
    padded = torch.zeros(len(sequences), length, hidden, dtype=sequences[0].dtype)
    masks = torch.zeros(len(sequences), length, dtype=torch.long)
    for index, sequence in enumerate(sequences):
        used = min(sequence.size(0), length)
        padded[index, :used] = sequence[:used]
        masks[index, :used] = 1
    return padded, masks


def _batch_tokenize(messages, tokenizer, max_length: int, batch_size: int = 256):
    ids, masks = [], []
    for start in range(0, len(messages), batch_size):
        batch = messages[start:start + batch_size]
        texts = [
            tokenizer.apply_chat_template(item, tokenize=False, add_generation_prompt=True)
            for item in batch
        ]
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        ids.append(encoded["input_ids"])
        masks.append(encoded["attention_mask"])
    return torch.cat(ids), torch.cat(masks)


def Path_join(prefix, suffix: str):
    from pathlib import Path
    return Path(str(prefix) + suffix)


def Path_join_str(prefix: str, suffix: str):
    from pathlib import Path
    return Path(prefix + suffix)
