#!/usr/bin/env python3
"""Multimodal meta-features: text (spaCy+VADER) + image (OpenCV).

Main thesis demo on ModaSense/emotion: X=[M_text, M_image] with named
modality blocks for CFPerm localization.
"""

from __future__ import annotations

import io
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
    "img_warm_ratio",  # red-yellow share in hue
    "img_laplacian_var",
]

BLOCKS_MM = {
    "text_meta": list(TEXT_NAMES),
    "image_meta": list(IMAGE_NAMES),
}


def _image_from_cell(cell: Any) -> np.ndarray:
    if isinstance(cell, dict) and "bytes" in cell:
        raw = cell["bytes"]
    elif isinstance(cell, (bytes, bytearray)):
        raw = cell
    else:
        raise TypeError(type(cell))
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


def extract_image_meta(cell: Any) -> dict[str, float]:
    bgr = _image_from_cell(cell)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(gray, 80, 160)
    hue = hsv[:, :, 0].astype(float)
    sat = hsv[:, :, 1].astype(float)
    # warm: hue near red/yellow (OpenCV hue 0-180)
    warm = ((hue < 25) | (hue > 160)).mean()
    return {
        "img_brightness": float(gray.mean() / 255.0),
        "img_contrast": float(gray.std() / 255.0),
        "img_edge_density": float(edges.mean() / 255.0),
        "img_sat_mean": float(sat.mean() / 255.0),
        "img_hue_std": float(hue.std() / 180.0),
        "img_warm_ratio": float(warm),
        "img_laplacian_var": float(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0),
    }


def extract_row_mm(text: str, image_cell: Any) -> dict[str, float]:
    out = extract_meta(text)
    out.update(extract_image_meta(image_cell))
    return out


def build_mm_matrix(texts: list[str], images: list[Any]) -> tuple[np.ndarray, list[str]]:
    names = TEXT_NAMES + IMAGE_NAMES
    rows = []
    for t, im in zip(texts, images):
        d = extract_row_mm(t, im)
        rows.append([float(d[n]) for n in names])
    return np.asarray(rows, dtype=float), names


def schema_mm() -> dict:
    p_text = len(TEXT_NAMES)
    p_img = len(IMAGE_NAMES)
    return {
        "layout": "X=[M_text, M_image]",
        "modality": {
            "text": list(range(0, p_text)),
            "image": list(range(p_text, p_text + p_img)),
        },
        "blocks": {
            "text_meta": {"indices": list(range(0, p_text)), "package": "spacy+vader"},
            "image_meta": {
                "indices": list(range(p_text, p_text + p_img)),
                "package": "opencv",
            },
        },
        "p_total": p_text + p_img,
    }
