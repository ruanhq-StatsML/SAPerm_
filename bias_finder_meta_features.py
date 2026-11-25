#embedding with the models for the enhancement of the power:
from utils import *
import os 
os.chdir('/Users/heqiaoruan/Downloads/writeup_software 6/SAPerm/SAPerm_software')
import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd 
from benchmark_methods_distributionshift_testing import *

#streamlined embedding generation for the most recent versions:
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List

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
    #1. Qwen3-Embedding-0.6B
    'qwen3-0.6b': EmbeddingModelConfig(
        name = "Qwen/Qwen3-Embedding-0.6B",
        pooling = 'mean',
        dim = 1024,
        max_length = 32768,
        batch_size = 64,
        prefix_format = 'Instruct: {task}\nQuery: {text}',
        default_instruction = 'Given a web search query, retrieve relevant passages',
        trust_remote_code = True),
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
        name = 'sentence-transformers/all-MiniLM-L6-v2',
        pooling = 'CLS',
        dim = 384,
        max_length = 512,
        batch_size = 32,
        trust_remote_code = False
    ),
    'MongoDB_Leaf': EmbeddingModelConfig(
        name = 'MongoDB/mdbr-leaf-ir',
        pooling = 'mean',
        dim = 768,
        max_length = 512,
        batch_size = 64,
        trust_remote_code = False
    ),
    'Emb_gemma': EmbeddingModelConfig(
    name = 'google/embeddinggemma-300m',
    pooling = 'mean',
    dim = 768,
    max_length = 2048,
    batch_size = 64,
    trust_remote_code = False
    )
}
get_embedding_wrapper('dataset/GEN-sarc-notsarc.csv', text_col = 'text', MODEL_CONFIGS = MODEL_CONFIGS)
get_embedding_wrapper('dataset/RQ-sarc-notsarc.csv', text_col = 'text', MODEL_CONFIGS = MODEL_CONFIGS)
get_embedding_wrapper('dataset/HYP-sarc-notsarc.csv', text_col = 'text', MODEL_CONFIGS = MODEL_CONFIGS)
#Testing the train, test

def concatenate_meta_features()



