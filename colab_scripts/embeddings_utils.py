"""Shared embedding helpers ported from the Colab notebook Collections_of_Embeddings_6dec.

Cell mapping: EmbeddingModelConfig and MODEL_CONFIGS from cell 8, get_embedding from
cell 9, get_embedding_wrapper from cell 31 (the latest version).

Dependencies: torch, transformers, numpy, pandas, tqdm.
Set HF_TOKEN in the environment if a Hugging Face model needs authentication.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = Path(os.environ.get('SAPERM_WORK_DIR', REPO_ROOT / 'colab_work'))
DATASET_DIR = WORK_DIR / 'dataset'
EMBEDDING_DIR = WORK_DIR / 'Embeddings'
TRANSLATION_DIR = WORK_DIR / 'translation_version' / 'translation_version'


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


MODEL_CONFIGS = {
    # 1. Qwen3-Embedding-0.6B
    'qwen3-0.6b': EmbeddingModelConfig(
        name="Qwen/Qwen3-Embedding-0.6B",
        pooling='mean',
        dim=1024,
        max_length=32768,
        batch_size=64,
        prefix_format='Instruct: {task}\nQuery: {text}',
        default_instruction='Given a web search query, retrieve relevant passages',
        trust_remote_code=True),
    # 2. Harrier-OSS-v1-0.6B
    "harrier-0.6b": EmbeddingModelConfig(
        name="microsoft/harrier-oss-v1-0.6b",
        pooling="last",
        dim=1024,
        max_length=32768,
        batch_size=64,
        trust_remote_code=True,
    ),
    # 3. MiniLM Small Language Model
    'MiniLM_L6': EmbeddingModelConfig(
        name='sentence-transformers/all-MiniLM-L6-v2',
        pooling='CLS',
        dim=384,
        max_length=512,
        batch_size=32,
        trust_remote_code=False
    ),
    'MongoDB_Leaf': EmbeddingModelConfig(
        name='MongoDB/mdbr-leaf-ir',
        pooling='mean',
        dim=768,
        max_length=512,
        batch_size=64,
        trust_remote_code=False
    ),
    'Emb_gemma': EmbeddingModelConfig(
        name='google/embeddinggemma-300m',
        pooling='mean',
        dim=768,
        max_length=2048,
        batch_size=64,
        trust_remote_code=False
    ),
}


def get_embedding(texts, model_name='harrier-0.6b', MODEL_CONFIGS=MODEL_CONFIGS):
    config = MODEL_CONFIGS[model_name]
    max_length = config.max_length
    batch_size = config.batch_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_id = config.name
    instruction = config.default_instruction
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    processed_texts = []
    if config.prefix_format:
        default_instruction = 'Given a web search query, retrieve relevant passages'
        task = instruction if instruction is not None else default_instruction
        processed_texts = [config.prefix_format.format(task=task, text=text) for text in texts]
    else:
        processed_texts = texts
    all_embeddings = []
    for i in tqdm(range(0, len(processed_texts), batch_size), desc=f'Embedding {model_name}'):
        batch_texts = processed_texts[i:(i + batch_size)]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
        pooling = config.pooling
        if pooling == 'mean':
            attention_mask = inputs['attention_mask']
            masked_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_emb = torch.sum(hidden_states * masked_expanded, dim=1)
            mean_emb = sum_emb / torch.clamp(masked_expanded.sum(dim=1), min=1e-9)
            batch_embeds = mean_emb
        elif pooling == 'CLS':
            batch_embeds = hidden_states[:, 0, :]
        elif pooling == 'last':
            attention_mask = inputs['attention_mask']
            last_token_indices = attention_mask.sum(dim=1) - 1
            batch_embeds = hidden_states[torch.arange(hidden_states.size(0),
                                                      device=device), last_token_indices]
        else:
            raise ValueError(f"Invalid Pooling: {pooling}")
        all_embeddings.append(batch_embeds.to(torch.float32).cpu().numpy())
    return np.concatenate(all_embeddings)


def get_embedding_wrapper(df_csv, text_col, MODEL_CONFIGS=MODEL_CONFIGS, back_trans=True,
                          delimiter=None, header=None):
    if delimiter is None:
        df = pd.read_csv(df_csv)
    else:
        df = pd.read_csv(df_csv, delimiter=delimiter, on_bad_lines='skip', header=header)
    dataset_name = os.path.splitext(str(df_csv))[0]
    for model_name in list(MODEL_CONFIGS.keys()):
        embedding0 = get_embedding(df[text_col].astype(str).tolist(), model_name=model_name)
        if back_trans:
            np.save(f"{dataset_name}_{model_name}_embedding.npy", embedding0)
        else:
            np.save(f"{dataset_name}_{model_name}_embedding_backtrans.npy", embedding0)
