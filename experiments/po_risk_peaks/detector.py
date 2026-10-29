"""DistilBERT detector, block shift, and the two adaptations.

Lexical block: token embeddings.
Semantic block: the CLS vector of the last transformer layer.
The CPU box this run uses does not fit a fine-tuned RoBERTa, so the
detector is ``distilbert-base-uncased``. The residual and the blocks are
the same objects on a 12-layer model.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class TextSet(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=int)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def load_backbone(model_name: str = "distilbert-base-uncased"):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.eval()
    return model, tokenizer


def predict_proba(model, tokenizer, texts, batch_size: int = 16, max_length: int = 128) -> np.ndarray:
    model.eval()
    chunks = []
    texts = list(texts)
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            logits = model(**encoded).logits
            chunks.append(torch.softmax(logits, dim=-1).cpu().numpy())
    if not chunks:
        return np.zeros((0, 2), dtype=float)
    return np.concatenate(chunks, axis=0)


def train_detector(model, tokenizer, texts, labels, epochs: int = 2, batch_size: int = 8, lr: float = 2e-5, max_length: int = 128, seed: int = 0) -> dict[str, float]:
    torch.manual_seed(seed)
    loader = DataLoader(TextSet(texts, labels, tokenizer, max_length), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()
    last_loss = 0.0
    steps = 0
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            labels_t = batch.pop("labels")
            loss = model(**batch, labels=labels_t).loss
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach())
            steps += 1
    model.eval()
    return {"train_loss": last_loss, "steps": float(steps)}


def _masked_token_distance(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> float:
    # left/right: (batch, seq, dim). Padding positions are dropped.
    dist = 1.0 - F.cosine_similarity(left, right, dim=-1)
    weight = mask.float()
    total = float(weight.sum().clamp(min=1.0))
    return float((dist * weight).sum() / total)


def block_distances(model, tokenizer, left_texts, right_texts, batch_size: int = 8, max_length: int = 128) -> dict[str, float]:
    """Mean cosine distance of the embedding block and of the final CLS."""
    model.eval()
    lexical = []
    semantic = []
    n = 0
    for start in range(0, len(left_texts), batch_size):
        a = list(left_texts[start : start + batch_size])
        b = list(right_texts[start : start + batch_size])
        enc_a = tokenizer(a, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc_b = tokenizer(b, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        # Pairwise distance needs a shared length.
        length = max(enc_a["input_ids"].shape[1], enc_b["input_ids"].shape[1])
        enc_a = tokenizer(a, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        enc_b = tokenizer(b, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        with torch.no_grad():
            hid_a = model(**enc_a, output_hidden_states=True).hidden_states
            hid_b = model(**enc_b, output_hidden_states=True).hidden_states
        mask = (enc_a["attention_mask"] * enc_b["attention_mask"]).float()
        lexical.append(_masked_token_distance(hid_a[0], hid_b[0], mask) * len(a))
        semantic.append((1.0 - F.cosine_similarity(hid_a[-1][:, 0], hid_b[-1][:, 0], dim=-1)).sum().item())
        n += len(a)
    return {"lexical": float(sum(lexical) / max(n, 1)), "semantic": float(sum(semantic) / max(n, 1))}


def adapt_upper_layers(model, tokenizer, clean_texts, clean_labels, adv_texts, adv_labels, epochs: int = 1, batch_size: int = 4, lr: float = 1e-5, max_length: int = 128) -> None:
    """Freeze the embedding block and fit the upper layers on clean/attacked pairs.

    The extra term pulls the two CLS vectors together, so a lexical edit that
    preserved the label stops moving the semantic block.
    """
    for param in model.distilbert.embeddings.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()
    n = len(clean_texts)
    for _ in range(epochs):
        order = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            clean = [clean_texts[int(i)] for i in idx]
            adv = [adv_texts[int(i)] for i in idx]
            y_clean = torch.tensor([int(clean_labels[int(i)]) for i in idx], dtype=torch.long)
            y_adv = torch.tensor([int(adv_labels[int(i)]) for i in idx], dtype=torch.long)
            enc_c = tokenizer(clean, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            enc_a = tokenizer(adv, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            optimizer.zero_grad(set_to_none=True)
            out_c = model(**enc_c, output_hidden_states=True, labels=y_clean)
            out_a = model(**enc_a, output_hidden_states=True, labels=y_adv)
            cosine = F.cosine_similarity(out_c.hidden_states[-1][:, 0], out_a.hidden_states[-1][:, 0], dim=-1)
            loss = out_c.loss + out_a.loss + 0.5 * (1.0 - cosine).mean()
            loss.backward()
            optimizer.step()
    model.eval()
