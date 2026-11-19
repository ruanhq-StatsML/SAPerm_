"""Optional heavy extractors distilled from the embedding notebooks.

Text  — MiniLM / Qwen-style mean pooling (Collections_of_Embeddings)
Audio — WavLM sliding-window mean (StreamlinedAudioEmbedding_extraction)
Video — ViViT mean pool over frames (VideoEmbeddingExtraction / MSR-VTT)

These imports are lazy so the interaction math stays usable without GPU deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class EmbeddingModelConfig:
    name: str
    pooling: str = "mean"
    prefix_format: Optional[str] = None
    dim: int = 384
    max_length: int = 512
    batch_size: int = 32
    trust_remote_code: bool = False
    normalize: bool = True
    default_instruction: Optional[str] = None


DEFAULT_TEXT_CONFIGS = {
    "MiniLM_L6": EmbeddingModelConfig(
        name="sentence-transformers/all-MiniLM-L6-v2",
        pooling="cls",
        dim=384,
        max_length=512,
        batch_size=32,
    ),
    "qwen3-0.6b": EmbeddingModelConfig(
        name="Qwen/Qwen3-Embedding-0.6B",
        pooling="mean",
        dim=1024,
        max_length=8192,
        batch_size=16,
        prefix_format="Instruct: {task}\nQuery: {text}",
        default_instruction="Given a web search query, retrieve relevant passages",
        trust_remote_code=True,
    ),
}

WAVLM_MODEL = "microsoft/wavlm-base-plus"
VIVIT_MODEL = "google/vivit-b-16x2-kinetics400"
TARGET_SR = 16000


def _device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pool_hidden(last_hidden, attention_mask, pooling: str):
    import torch

    if pooling.lower() == "cls":
        return last_hidden[:, 0]
    if pooling.lower() == "last":
        lengths = attention_mask.sum(dim=1) - 1
        batch = torch.arange(last_hidden.size(0), device=last_hidden.device)
        return last_hidden[batch, lengths]
    # mean pool
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


def extract_text_embeddings(
    texts: Sequence[str],
    model_key: str = "MiniLM_L6",
    config: Optional[EmbeddingModelConfig] = None,
) -> np.ndarray:
    """Encode captions / ASR transcripts with a HuggingFace encoder."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    cfg = config or DEFAULT_TEXT_CONFIGS[model_key]
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
    model = AutoModel.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.prefix_format:
        task = cfg.default_instruction or "retrieve relevant passages"
        processed = [cfg.prefix_format.format(task=task, text=t) for t in texts]
    else:
        processed = [str(t) for t in texts]
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(processed), cfg.batch_size):
            batch = processed[i : i + cfg.batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=cfg.max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**inputs)
            pooled = _pool_hidden(out.last_hidden_state, inputs["attention_mask"], cfg.pooling)
            if cfg.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled.cpu().numpy())
    return np.vstack(chunks)


def extract_audio_embeddings(
    waveforms: Sequence[np.ndarray],
    sample_rates: Sequence[int],
    *,
    model_name: str = WAVLM_MODEL,
    window_sec: float = 10.0,
    hop_sec: float = 2.0,
) -> np.ndarray:
    """WavLM sliding-window embeddings, mean-pooled per clip."""
    import librosa
    import torch
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    device = _device()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    window_samples = int(window_sec * TARGET_SR)
    hop_samples = int(hop_sec * TARGET_SR)
    all_embeds = []
    for audio, sr in zip(waveforms, sample_rates):
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        if len(audio) < window_samples:
            audio = np.pad(audio, (0, window_samples - len(audio)))
        starts = list(range(0, max(1, len(audio) - window_samples + 1), hop_samples))
        window_embeds = []
        for start in starts:
            segment = audio[start : start + window_samples]
            inputs = feature_extractor(
                segment, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
            window_embeds.append(emb)
        all_embeds.append(np.mean(np.stack(window_embeds, axis=0), axis=0))
    return np.stack(all_embeds, axis=0)


def extract_video_embeddings(
    video_paths: Sequence[str],
    *,
    model_name: str = VIVIT_MODEL,
    num_frames: int = 32,
) -> np.ndarray:
    """ViViT CLS/mean embedding for each video path (decord + transformers)."""
    import torch
    from decord import VideoReader, cpu
    from transformers import VivitImageProcessor, VivitModel

    device = _device()
    image_processor = VivitImageProcessor.from_pretrained(model_name)
    model = VivitModel.from_pretrained(model_name).to(device)
    model.eval()
    embeds = []
    for path in video_paths:
        vr = VideoReader(path, ctx=cpu(0))
        indices = np.linspace(0, max(len(vr) - 1, 0), num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        pixel_values = image_processor(list(frames), return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            out = model(pixel_values)
            emb = out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        embeds.append(emb)
    return np.stack(embeds, axis=0)
