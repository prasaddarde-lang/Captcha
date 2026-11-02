"""Image feature extraction shared by training and runtime inference.

Features describe the *layout and appearance* of a cropped element (edge orientation
histograms, colour layout, grid-line profiles, aspect ratio). They are designed to tell
challenge *types* apart, not to read characters.
"""
from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path

import cv2
import numpy as np

FEATURE_VERSION = "f1"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_SIDE = 4000


class ImageInputError(ValueError):
    pass


def load_image(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise ImageInputError(f"image not found: {p.name}")
    if p.stat().st_size > MAX_IMAGE_BYTES:
        raise ImageInputError("image larger than 10 MB")
    data = np.fromfile(str(p), dtype=np.uint8)  # handles non-ASCII Windows paths
    return _decode(data, p.name)


def load_image_base64(value: str) -> np.ndarray:
    """Decode a base64 PNG/JPEG sent inline by a remote robot (optionally as a ``data:image/...;base64,`` URL)."""
    text = value.strip()
    if text.startswith("data:"):
        header, _, text = text.partition(",")
        if not re.fullmatch(r"data:image/(png|jpe?g);base64", header, re.IGNORECASE):
            raise ImageInputError("image_base64 data URL must be image/png or image/jpeg")
    try:
        raw = base64.b64decode("".join(text.split()), validate=True)
    except (binascii.Error, ValueError):
        raise ImageInputError("image_base64 is not valid base64") from None
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageInputError("image larger than 10 MB")
    if not (raw.startswith(b"\x89PNG\r\n\x1a\n") or raw.startswith(b"\xff\xd8\xff")):
        raise ImageInputError("image_base64 must be a PNG or JPEG image")
    return _decode(np.frombuffer(raw, dtype=np.uint8), "image_base64")


def _decode(data: np.ndarray, name: str) -> np.ndarray:
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ImageInputError(f"not a decodable image: {name}")
    if max(img.shape[:2]) > MAX_SIDE:
        raise ImageInputError("image dimensions too large")
    return img


def hog(gray: np.ndarray, cell: int = 8, bins: int = 9) -> np.ndarray:
    """Histogram of oriented gradients (unsigned, bilinear orientation binning,
    2x2-cell blocks with L2-Hys normalisation). numpy implementation because
    OpenCV 5 no longer ships cv2.HOGDescriptor in the main module."""
    g = gray.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    pos = (ang % 180.0) / (180.0 / bins) - 0.5
    b0 = np.floor(pos).astype(np.int64)
    frac = pos - b0
    b0 %= bins
    b1 = (b0 + 1) % bins
    h, w = g.shape
    acc = np.zeros((h, w, bins), np.float32)
    rows, cols = np.indices((h, w))
    np.add.at(acc, (rows, cols, b0), mag * (1 - frac))
    np.add.at(acc, (rows, cols, b1), mag * frac)
    cy, cx = h // cell, w // cell
    cells = acc[:cy * cell, :cx * cell].reshape(cy, cell, cx, cell, bins).sum(axis=(1, 3))
    blocks = np.concatenate([cells[:-1, :-1], cells[:-1, 1:], cells[1:, :-1], cells[1:, 1:]], axis=2)
    blocks = blocks / np.sqrt((blocks ** 2).sum(axis=2, keepdims=True) + 1e-6)
    blocks = np.minimum(blocks, 0.2)
    blocks = blocks / np.sqrt((blocks ** 2).sum(axis=2, keepdims=True) + 1e-6)
    return blocks.ravel()


class FeatureExtractor:
    def __init__(self, width: int = 96, height: int = 48):
        self.width, self.height = width, height

    @property
    def spec(self) -> dict:
        return {"version": FEATURE_VERSION, "width": self.width, "height": self.height}

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        small = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hog_v = hog(gray)

        # coarse colour layout (Lab, 8x4 cells)
        lab = cv2.cvtColor(cv2.resize(img, (8, 4), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB)
        layout = lab.astype(np.float32).ravel() / 255.0

        # global colour statistics
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hue = cv2.calcHist([hsv], [0], None, [16], [0, 180]).ravel()
        sat = cv2.calcHist([hsv], [1], None, [8], [0, 256]).ravel()
        hist = np.concatenate([hue, sat]) / float(self.width * self.height)

        # edge projection profiles: tile grids and widget borders produce strong lines
        edges = cv2.Canny(gray, 60, 160).astype(np.float32) / 255.0
        rows = cv2.resize(edges.mean(axis=1, keepdims=True), (1, 24), interpolation=cv2.INTER_AREA).ravel()
        cols = cv2.resize(edges.mean(axis=0, keepdims=True), (12, 1), interpolation=cv2.INTER_AREA).ravel()

        stats = np.array([
            np.log(w / h), np.log(w) / 8.0, np.log(h) / 8.0,
            gray.mean() / 255.0, gray.std() / 128.0, edges.mean(),
        ], dtype=np.float32)
        return np.concatenate([hog_v, layout, hist, rows, cols, stats]).astype(np.float32)

    @property
    def dim(self) -> int:
        return int(self(np.zeros((self.height, self.width, 3), np.uint8)).shape[0])
