#!/usr/bin/env python3
"""Multimodal meta: image (OpenCV) + audio (WAV). Text meta = extract_features."""

from __future__ import annotations

import io
import wave
from typing import Any

import cv2
import numpy as np
from PIL import Image

from extract_features import FEATURE_NAMES as TEXT_NAMES
from extract_features import extract_meta

IMAGE_NAMES = [
    "img_brightness",
    "img_contrast",
    "img_edge_density",
    "img_sat_mean",
    "img_hue_std",
    "img_warm_ratio",
    "img_laplacian_var",
]
AUDIO_NAMES = [
    "aud_rms_energy",
    "aud_zcr",
    "aud_spectral_centroid",
    "aud_lowband_ratio",
    "aud_silence_ratio",
    "aud_peak_to_rms",
]


def _bytes(cell: Any) -> bytes:
    return cell["bytes"] if isinstance(cell, dict) else cell


def extract_image_meta(cell: Any) -> dict[str, float]:
    bgr = cv2.cvtColor(
        np.asarray(Image.open(io.BytesIO(_bytes(cell))).convert("RGB")), cv2.COLOR_RGB2BGR
    )
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[:, :, 0].astype(float), hsv[:, :, 1].astype(float)
    return {
        "img_brightness": float(gray.mean() / 255.0),
        "img_contrast": float(gray.std() / 255.0),
        "img_edge_density": float(cv2.Canny(gray, 80, 160).mean() / 255.0),
        "img_sat_mean": float(sat.mean() / 255.0),
        "img_hue_std": float(hue.std() / 180.0),
        "img_warm_ratio": float(((hue < 25) | (hue > 160)).mean()),
        "img_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0),
    }


def extract_audio_meta(cell: Any) -> dict[str, float]:
    with wave.open(io.BytesIO(_bytes(cell)), "rb") as w:
        sr, nch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError(f"16-bit WAV required, got sampwidth={sw}")
    y = np.frombuffer(frames, dtype=np.int16).astype(np.float64)
    if nch > 1:
        y = y.reshape(-1, nch).mean(1)
    y /= 32768.0
    if len(y) < 8:
        return {n: 0.0 for n in AUDIO_NAMES}
    rms = float(np.sqrt(np.mean(y * y)) + 1e-12)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    mass = float(spec.sum()) + 1e-12
    return {
        "aud_rms_energy": rms,
        "aud_zcr": float(np.mean(np.abs(np.diff(np.sign(y)))) / 2.0),
        "aud_spectral_centroid": float((freqs * spec).sum() / mass) / (sr / 2.0),
        "aud_lowband_ratio": float(spec[freqs <= 500].sum() / mass),
        "aud_silence_ratio": float(np.mean(np.abs(y) < 0.02)),
        "aud_peak_to_rms": min(float(np.max(np.abs(y)) / rms), 20.0) / 20.0,
    }


def build_mm_matrix(
    texts: list[str],
    images: list[Any],
    audios: list[Any] | None = None,
) -> tuple[np.ndarray, list[str]]:
    names = list(TEXT_NAMES) + IMAGE_NAMES + (AUDIO_NAMES if audios is not None else [])
    rows = []
    for i, (t, im) in enumerate(zip(texts, images)):
        d = {**extract_meta(t), **extract_image_meta(im)}
        if audios is not None:
            d.update(extract_audio_meta(audios[i]))
        rows.append([float(d[n]) for n in names])
    return np.asarray(rows, dtype=float), names


def schema_mm(with_audio: bool = True) -> dict:
    pt, pi = len(TEXT_NAMES), len(IMAGE_NAMES)
    pa = len(AUDIO_NAMES) if with_audio else 0
    blocks = {
        "text_meta": {"indices": list(range(pt)), "package": "spacy+vader"},
        "image_meta": {"indices": list(range(pt, pt + pi)), "package": "opencv"},
    }
    modality = {"text": blocks["text_meta"]["indices"], "image": blocks["image_meta"]["indices"]}
    if with_audio:
        blocks["audio_meta"] = {
            "indices": list(range(pt + pi, pt + pi + pa)),
            "package": "wave+numpy",
        }
        modality["audio"] = blocks["audio_meta"]["indices"]
    return {
        "layout": "X=[M_text, M_image, M_audio]" if with_audio else "X=[M_text, M_image]",
        "modality": modality,
        "blocks": blocks,
        "p_total": pt + pi + pa,
    }
