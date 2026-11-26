"""Pretrained LLM registry — same models as ``bias_finder_meta_features`` / notebook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EmbeddingModelConfig:
    name: str
    pooling: str
    prefix_format: Optional[str] = None
    dim: int = 256
    max_length: int = 256
    batch_size: int = 32
    trust_remote_code: bool = False
    normalize: bool = True
    default_instruction: Optional[str] = None


MODEL_CONFIGS: dict[str, EmbeddingModelConfig] = {
    "qwen3-0.6b": EmbeddingModelConfig(
        name="Qwen/Qwen3-Embedding-0.6B",
        pooling="mean",
        dim=1024,
        max_length=512,
        batch_size=32,
        prefix_format="Instruct: {task}\nQuery: {text}",
        default_instruction="Given a web search query, retrieve relevant passages",
        trust_remote_code=True,
    ),
    "harrier-0.6b": EmbeddingModelConfig(
        name="microsoft/harrier-oss-v1-0.6b",
        pooling="last",
        dim=1024,
        max_length=512,
        batch_size=32,
        trust_remote_code=True,
    ),
    "MiniLM_L6": EmbeddingModelConfig(
        name="sentence-transformers/all-MiniLM-L6-v2",
        pooling="CLS",
        dim=384,
        max_length=512,
        batch_size=32,
        trust_remote_code=False,
    ),
    "MongoDB_Leaf": EmbeddingModelConfig(
        name="MongoDB/mdbr-leaf-ir",
        pooling="mean",
        dim=768,
        max_length=512,
        batch_size=32,
        trust_remote_code=False,
    ),
    "Emb_gemma": EmbeddingModelConfig(
        name="google/embeddinggemma-300m",
        pooling="mean",
        dim=768,
        max_length=512,
        batch_size=32,
        trust_remote_code=False,
    ),
}


def get_embedding(
    texts: list[str], model_name: str = "MiniLM_L6", batch_size: int | None = None
) -> np.ndarray:
    """Notebook-aligned mean/CLS/last pooling (subset of ``utils.get_embedding``)."""
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    bs = batch_size or cfg.batch_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
    model = AutoModel.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg.prefix_format:
        task = cfg.default_instruction or ""
        processed = [cfg.prefix_format.format(task=task, text=t) for t in texts]
    else:
        processed = texts

    chunks = []
    with torch.no_grad():
        for i in tqdm(range(0, len(processed), bs), desc=f"embed {model_name}"):
            batch = processed[i : i + bs]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            ).to(device)
            hidden = model(**inputs, output_hidden_states=True).hidden_states[-1]
            mask = inputs["attention_mask"]
            if cfg.pooling == "mean":
                m = mask.unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
            elif cfg.pooling == "CLS":
                pooled = hidden[:, 0]
            elif cfg.pooling == "last":
                last = mask.sum(1) - 1
                pooled = hidden[torch.arange(hidden.size(0), device=device), last]
            else:
                raise ValueError(cfg.pooling)
            if cfg.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.float().cpu().numpy())
    return np.vstack(chunks)