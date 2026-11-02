"""Synthetic, authorised image dataset for CAPTCHA *type* classification.

    python training/generate_dataset.py --count 10000            # base dataset (skips if present)
    python training/generate_dataset.py --count 10000 --force    # regenerate

Every image is drawn here with OpenCV: Hershey fonts, varied backgrounds, contrasts,
resolutions, rotation, warping and noise. Provider categories are generic *lookalike*
widgets (layout and colour only, no logos or trademarks). Nothing is downloaded or
harvested. The manifest records the category label only; rendered characters and
arithmetic answers are deliberately NOT stored, so the dataset cannot be used to train
a solver.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.common import (apply_thread_limits, check_disk, get_paths, load_config, now_iso,  # noqa: E402
                             read_json, sha256_bytes, write_json)

apply_thread_limits()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

GENERATOR_VERSION = "1.0.0"
AA = cv2.LINE_AA
FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX,
         cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
         cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
UI_FONTS = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_COMPLEX_SMALL]
ALNUM = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789"
WORDS = ("account order invoice status report search customer number details settings profile "
         "password email address payment shipping product review summary update notes total date "
         "reference team project support contact help home services billing open close save").split()
NOUNS = ["bicycles", "buses", "crosswalks", "traffic lights", "boats", "bridges", "stairs", "motorcycles",
         "hydrants", "palm trees", "cars", "chimneys", "mountains", "tractors"]


# ----------------------------------------------------------------------------- drawing helpers
def col(g, lo=0, hi=255):
    return tuple(int(v) for v in g.integers(lo, hi + 1, 3))


def palette(g):
    mode = g.choice(["dark_on_light", "dark_on_light", "light_on_dark", "low_contrast"])
    if mode == "dark_on_light":
        return mode, col(g, 185, 255), col(g, 0, 90)
    if mode == "light_on_dark":
        return mode, col(g, 0, 70), col(g, 170, 255)
    base = int(g.integers(110, 170))
    bg = tuple(int(np.clip(base + d, 0, 255)) for d in g.integers(-10, 10, 3))
    fg = tuple(int(np.clip(base + s * int(g.integers(50, 75)), 0, 255)) for s in [g.choice([-1, 1])] * 3)
    return mode, bg, fg


def background(g, w, h, base):
    style = str(g.choice(["solid", "gradient", "noise", "stripes", "dots", "blobs", "paper"]))
    img = np.empty((h, w, 3), np.uint8)
    img[:] = base
    if style == "gradient":
        other = np.clip(np.array(base) + g.integers(-60, 60, 3), 0, 255)
        t = np.linspace(0, 1, w if g.random() < 0.5 else h, dtype=np.float32)
        ramp = (np.array(base)[None, :] * (1 - t[:, None]) + other[None, :] * t[:, None]).astype(np.uint8)
        img[:] = ramp[None, :, :] if ramp.shape[0] == w else ramp[:, None, :]
    elif style == "noise":
        img = np.clip(img.astype(np.int16) + g.normal(0, g.uniform(8, 28), img.shape), 0, 255).astype(np.uint8)
    elif style == "stripes":
        step = int(g.integers(4, 14))
        c = tuple(int(np.clip(v + g.integers(-35, 35), 0, 255)) for v in base)
        for x in range(-h, w, step):
            cv2.line(img, (x, 0), (x + h, h), c, 1, AA)
    elif style == "dots":
        for _ in range(int(g.integers(30, 250))):
            c = tuple(int(np.clip(v + g.integers(-50, 50), 0, 255)) for v in base)
            cv2.circle(img, (int(g.integers(0, w)), int(g.integers(0, h))), int(g.integers(1, 4)), c, -1, AA)
    elif style == "blobs":
        over = img.copy()
        for _ in range(int(g.integers(2, 6))):
            cv2.ellipse(over, (int(g.integers(0, w)), int(g.integers(0, h))),
                        (int(g.integers(10, w // 2 + 11)), int(g.integers(5, h // 2 + 6))),
                        float(g.uniform(0, 180)), 0, 360, col(g), -1, AA)
        img = cv2.addWeighted(img, 0.65, over, 0.35, 0)
    elif style == "paper":
        n = g.normal(0, 40, (h // 4 + 1, w // 4 + 1, 3)).astype(np.float32)
        n = cv2.resize(cv2.GaussianBlur(n, (0, 0), 1.5), (w, h))
        img = np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return img, style


def fit_scale(text, font, thick, max_w, max_h):
    (tw, th), _ = cv2.getTextSize(text, font, 1.0, thick)
    return max(0.25, min(max_w / max(tw, 1), max_h / max(th, 1)))


def put(img, text, x, y, font, scale, color, thick=1):
    cv2.putText(img, text, (int(x), int(y)), font, scale, color, thick, AA)


def put_fit(img, text, box, font, color, thick=1, align="left"):
    x, y, w, h = box
    s = fit_scale(text, font, thick, w, h)
    (tw, th), _ = cv2.getTextSize(text, font, s, thick)
    tx = x + (w - tw) // 2 if align == "center" else x
    put(img, text, tx, y + (h + th) // 2, font, s, color, thick)


def draw_chars(img, text, g, color, font, rot, jitter, overlap, x0, y_mid, char_h):
    """Render characters one by one with individual rotation / jitter."""
    x = x0
    thick = int(g.integers(1, 4))
    for ch in text:
        if ch == " ":
            x += int(char_h * 0.45)
            continue
        s = fit_scale(ch, font, thick, char_h * 0.9, char_h * g.uniform(0.75, 1.0))
        (cw, chh), base = cv2.getTextSize(ch, font, s, thick)
        size = int(max(cw, chh) * 2 + 8)
        mask = np.zeros((size, size), np.uint8)
        put(mask, ch, (size - cw) // 2, (size + chh) // 2, font, s, 255, thick)
        m = cv2.getRotationMatrix2D((size / 2, size / 2), float(g.uniform(-rot, rot)), 1.0)
        mask = cv2.warpAffine(mask, m, (size, size))
        cy = int(y_mid - size / 2 + g.uniform(-jitter, jitter))
        cx = int(x - (size - cw) / 2)
        y1, y2 = max(cy, 0), min(cy + size, img.shape[0])
        x1, x2 = max(cx, 0), min(cx + size, img.shape[1])
        if y2 > y1 and x2 > x1:
            a = (mask[y1 - cy:y2 - cy, x1 - cx:x2 - cx].astype(np.float32) / 255.0)[..., None]
            img[y1:y2, x1:x2] = (img[y1:y2, x1:x2] * (1 - a) + np.array(color) * a).astype(np.uint8)
        x += int(cw * g.uniform(1.0 - overlap, 1.15))
    return x


def sine_warp(img, g, amp):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px, py = g.uniform(20, 60), g.uniform(15, 50)
    mx = xx + amp * np.sin(2 * np.pi * yy / py + g.uniform(0, 6.28))
    my = yy + amp * 0.6 * np.sin(2 * np.pi * xx / px + g.uniform(0, 6.28))
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def perspective(img, g, k):
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + g.uniform(-k, k, src.shape) * [w, h]).astype(np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)


def noise_lines(img, g, n, color=None, thick=(1, 2), arcs=False):
    h, w = img.shape[:2]
    for _ in range(n):
        c = color or col(g)
        t = int(g.integers(thick[0], thick[1] + 1))
        if arcs and g.random() < 0.5:
            cv2.ellipse(img, (int(g.integers(0, w)), int(g.integers(0, h))), (int(g.integers(10, w)), int(g.integers(5, h))),
                        float(g.uniform(0, 180)), float(g.uniform(0, 180)), float(g.uniform(180, 360)), c, t, AA)
        else:
            cv2.line(img, (int(g.integers(0, w)), int(g.integers(0, h))), (int(g.integers(0, w)), int(g.integers(0, h))), c, t, AA)


def speckle(img, g, n):
    h, w = img.shape[:2]
    for _ in range(n):
        cv2.circle(img, (int(g.integers(0, w)), int(g.integers(0, h))), int(g.integers(0, 2)), col(g), -1)


def photo(g, w, h):
    """Abstract 'photo': gradient sky/ground + random shapes, blurred."""
    img, _ = background(g, w, h, col(g, 40, 230))
    if g.random() < 0.6:
        horizon = int(g.integers(h // 3, h))
        img[horizon:] = (img[horizon:] * 0.5 + np.array(col(g, 30, 200)) * 0.5).astype(np.uint8)
    for _ in range(int(g.integers(2, 7))):
        c = col(g)
        kind = g.integers(0, 4)
        cx, cy = int(g.integers(0, w)), int(g.integers(0, h))
        r = int(g.integers(4, max(6, min(w, h) // 2)))
        if kind == 0:
            cv2.circle(img, (cx, cy), r, c, -1, AA)
        elif kind == 1:
            cv2.rectangle(img, (cx - r, cy - r // 2), (cx + r, cy + r // 2), c, -1)
        elif kind == 2:
            pts = np.array([[cx, cy - r], [cx - r, cy + r], [cx + r, cy + r]], np.int32)
            cv2.fillPoly(img, [pts], c, AA)
        else:
            pts = g.integers(0, [w, h], (int(g.integers(3, 7)), 2)).astype(np.int32)
            cv2.polylines(img, [pts], False, c, int(g.integers(1, 4)), AA)
    return cv2.GaussianBlur(img, (3, 3), 0) if g.random() < 0.7 else img


def button(img, g, box, text, fill, fg):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), fill, -1)
    put_fit(img, text, (x + 6, y + 3, w - 12, h - 6), int(g.choice(UI_FONTS)), fg, 1, "center")


def input_box(img, g, box, placeholder, dark=False):
    x, y, w, h = box
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40) if dark else (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), col(g, 120, 200), 1)
    if placeholder:
        put_fit(img, placeholder, (x + 6, y + 4, int(w * 0.7), h - 8), cv2.FONT_HERSHEY_SIMPLEX, (150, 150, 150))


def checkbox(img, g, x, y, s, state, dark=False):
    cv2.rectangle(img, (x, y), (x + s, y + s), (30, 30, 30) if dark else (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x + s, y + s), col(g, 120, 190), 2 if g.random() < 0.5 else 1)
    if state == "checked":
        cv2.polylines(img, [np.array([[x + s // 5, y + s // 2], [x + s // 2.3, y + int(s * 0.78)],
                                      [x + int(s * 0.85), y + s // 5]], np.int32)], False, (40, 160, 60), 3, AA)
    elif state == "spinner":
        cv2.ellipse(img, (x + s // 2, y + s // 2), (s // 2 - 2, s // 2 - 2), 0, float(g.uniform(0, 360)),
                    float(g.uniform(200, 320)), (200, 120, 40), 2, AA)


def widget_frame(g, w, h, dark):
    bg = col(g, 28, 50) if dark else tuple(int(v) for v in np.clip(np.array(col(g, 238, 252)), 0, 255))
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    border = col(g, 70, 90) if dark else col(g, 200, 225)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), border, 1)
    fg = col(g, 210, 240) if dark else col(g, 20, 60)
    return img, fg


def page_pad(img, g, prob=0.35):
    """Sometimes include a little surrounding page so crops are not perfectly tight."""
    if g.random() > prob:
        return img
    p = [int(g.integers(2, 20)) for _ in range(4)]
    base = col(g, 200, 255) if g.random() < 0.7 else col(g, 20, 70)
    return cv2.copyMakeBorder(img, p[0], p[1], p[2], p[3], cv2.BORDER_CONSTANT, value=base)


# ---------------------------------------------------------------------------- category renderers
def r_text(g):
    n = int(g.integers(4, 9))
    text = "".join(g.choice(list(ALNUM), n))
    ch = int(g.integers(26, 46))
    w, h = int(n * ch * g.uniform(0.75, 1.0) + 30), int(ch * g.uniform(1.6, 2.2))
    mode, bg, fg = palette(g)
    img, bstyle = background(g, w, h, bg)
    font = int(g.choice(FONTS)) | (cv2.FONT_ITALIC if g.random() < 0.3 else 0)
    draw_chars(img, text, g, fg, font, rot=8, jitter=3, overlap=0.05, x0=12, y_mid=h // 2, char_h=ch)
    nl = int(g.integers(0, 3))
    noise_lines(img, g, nl, fg if g.random() < 0.5 else None)
    framed = g.random() < 0.3
    if framed:
        img = framed_challenge(img, g, ["Enter the text shown", "Type the characters", "Enter the code"])
    return img, {"mode": mode, "background": bstyle, "font": font, "length": n, "noise_lines": nl, "framed": framed}


def r_text_distorted(g):
    n = int(g.integers(5, 9))
    text = "".join(g.choice(list(ALNUM), n))
    ch = int(g.integers(28, 48))
    w, h = int(n * ch * g.uniform(0.7, 0.95) + 30), int(ch * g.uniform(1.7, 2.4))
    mode, bg, fg = palette(g)
    img, bstyle = background(g, w, h, bg)
    font = int(g.choice(FONTS)) | (cv2.FONT_ITALIC if g.random() < 0.4 else 0)
    draw_chars(img, text, g, fg, font, rot=35, jitter=h * 0.15, overlap=0.3, x0=10, y_mid=h // 2, char_h=ch)
    amp = float(g.uniform(2.5, 8))
    img = sine_warp(img, g, amp)
    nl = int(g.integers(2, 8))
    noise_lines(img, g, nl, fg if g.random() < 0.6 else None, (1, 3), arcs=True)
    speckle(img, g, int(g.integers(40, 300)))
    if g.random() < 0.4:
        img = perspective(img, g, 0.06)
    if g.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, {"mode": mode, "background": bstyle, "font": font, "length": n, "warp_amp": round(amp, 2), "noise_lines": nl}


def r_math(g):
    a, b = int(g.integers(1, 30)), int(g.integers(1, 20))
    op = str(g.choice(["+", "-", "x", "*"]))
    fmt = int(g.integers(0, 4))
    text = [f"{a} {op} {b} = ?", f"{a}{op}{b}=", f"What is {a} {op} {b}?", f"{a} {op} {b} ="][fmt]
    ch = int(g.integers(24, 40))
    w, h = int(len(text) * ch * 0.62 + 30), int(ch * g.uniform(1.6, 2.2))
    mode, bg, fg = palette(g)
    img, bstyle = background(g, w, h, bg)
    font = int(g.choice(FONTS[:6]))
    rot = float(g.uniform(0, 20))
    draw_chars(img, text, g, fg, font, rot=rot, jitter=3 + rot / 5, overlap=0.08, x0=10, y_mid=h // 2, char_h=ch)
    if rot > 10:
        img = sine_warp(img, g, float(g.uniform(1, 4)))
    nl = int(g.integers(0, 5))
    noise_lines(img, g, nl, None, (1, 2), arcs=True)
    framed = g.random() < 0.3
    if framed:
        img = framed_challenge(img, g, ["Solve the sum", "Answer the question", "Security question"])
    return img, {"mode": mode, "background": bstyle, "font": font, "format": fmt, "rotation": round(rot, 1), "framed": framed}


def framed_challenge(chal, g, prompts):
    ch_h, ch_w = chal.shape[:2]
    dark = g.random() < 0.2
    w, h = ch_w + 40, ch_h + 110
    img, fg = widget_frame(g, w, h, dark)
    put_fit(img, str(g.choice(prompts)), (20, 8, w - 40, 22), int(g.choice(UI_FONTS)), fg)
    img[36:36 + ch_h, 20:20 + ch_w] = chal
    input_box(img, g, (20, 46 + ch_h, w - 40, 26), "" if g.random() < 0.5 else "Answer", dark)
    if g.random() < 0.5:
        cv2.circle(img, (w - 14, 20), 7, fg, 1, AA)  # refresh icon
    return img


def r_audio(g):
    w, h = int(g.integers(260, 420)), int(g.integers(120, 220))
    dark = g.random() < 0.3
    img, fg = widget_frame(g, w, h, dark)
    accent = col(g, 20, 220)
    prompts = ["Press PLAY and type what you hear", "Listen and enter the numbers you hear",
               "Audio challenge", "Type the words you hear", "Play the audio, then type the digits"]
    put_fit(img, str(g.choice(prompts)), (12, 8, w - 24, int(h * 0.14)), int(g.choice(UI_FONTS)), fg)
    cy = int(h * 0.42)
    r = int(g.integers(12, 22))
    cx = 16 + r
    cv2.circle(img, (cx, cy), r, accent, -1 if g.random() < 0.6 else 2, AA)
    tri = np.array([[cx - r // 3, cy - r // 2], [cx - r // 3, cy + r // 2], [cx + r // 2, cy]], np.int32)
    cv2.fillPoly(img, [tri], (255, 255, 255) if g.random() < 0.7 else fg, AA)
    left, right = cx + r + 12, w - 16
    style = str(g.choice(["wave", "progress", "both"]))
    if style in ("wave", "both"):
        n = int(g.integers(18, 50))
        xs = np.linspace(left, right, n)
        amp = r * g.uniform(0.6, 1.2)
        for x, v in zip(xs, g.uniform(0.1, 1.0, n)):
            cv2.line(img, (int(x), int(cy - v * amp)), (int(x), int(cy + v * amp)), accent, 2, AA)
    if style in ("progress", "both"):
        py = cy + (int(r * 1.4) if style == "both" else 0)
        cv2.line(img, (left, py), (right, py), col(g, 150, 200), 3, AA)
        k = int(g.integers(left, right))
        cv2.line(img, (left, py), (k, py), accent, 3, AA)
        cv2.circle(img, (k, py), 6, accent, -1, AA)
    if g.random() < 0.5:  # headphones icon
        hx, hy = w - 30, 16
        cv2.ellipse(img, (hx, hy + 10), (10, 10), 0, 180, 360, fg, 2, AA)
        cv2.rectangle(img, (hx - 12, hy + 8), (hx - 7, hy + 16), fg, -1)
        cv2.rectangle(img, (hx + 7, hy + 8), (hx + 12, hy + 16), fg, -1)
    input_box(img, g, (12, int(h * 0.68), int(w * 0.62), int(h * 0.16)), "Enter what you hear" if g.random() < 0.5 else "", dark)
    if g.random() < 0.8:
        button(img, g, (int(w * 0.68), int(h * 0.68), int(w * 0.28), int(h * 0.16)),
               str(g.choice(["VERIFY", "Submit", "Next"])), accent, (255, 255, 255))
    return page_pad(img, g), {"dark": dark, "style": style}


def r_image_classification(g):
    k = int(g.integers(2, 7))
    tile = int(g.integers(55, 110))
    rows = 1 if k <= 4 or g.random() < 0.4 else 2
    per_row = int(np.ceil(k / rows))
    gap = int(g.integers(6, 16))
    head = int(g.integers(34, 60))
    w = per_row * (tile + gap) + gap + 20
    h = head + rows * (tile + gap) + gap + (int(g.integers(30, 50)) if g.random() < 0.6 else 0)
    dark = g.random() < 0.2
    img, fg = widget_frame(g, w, h, dark)
    banner = g.random() < 0.5
    if banner:
        cv2.rectangle(img, (0, 0), (w, head - 4), col(g, 20, 200), -1)
        fg_head = (255, 255, 255)
    else:
        fg_head = fg
    prompts = ["Select the image that is upright", "Pick the odd one out", "Which picture shows an animal?",
               "Click the image that matches", f"Choose the image with {g.choice(NOUNS)}",
               "Select the correctly rotated image"]
    put_fit(img, str(g.choice(prompts)), (10, 6, w - 20, head - 14), int(g.choice(UI_FONTS)), fg_head)
    chosen = int(g.integers(-1, k))
    for i in range(k):
        r_, c_ = divmod(i, per_row)
        x, y = 10 + gap + c_ * (tile + gap), head + gap + r_ * (tile + gap)
        img[y:y + tile, x:x + tile] = photo(g, tile, tile)
        if i == chosen:
            cv2.rectangle(img, (x - 3, y - 3), (x + tile + 2, y + tile + 2), col(g, 0, 255), 3)
    if h > head + rows * (tile + gap) + gap:
        by = head + rows * (tile + gap) + gap + 4
        if g.random() < 0.5:  # rotate arrows
            for cx in (w // 2 - 30, w // 2 + 30):
                cv2.ellipse(img, (cx, by + 12), (10, 10), 0, 30, 300, fg, 2, AA)
        else:
            button(img, g, (w - 110, by, 96, h - by - 6), str(g.choice(["Submit", "VERIFY", "Next"])), col(g, 20, 200), (255, 255, 255))
    return page_pad(img, g), {"images": k, "rows": rows, "banner": banner, "dark": dark}


def r_grid(g):
    n = int(g.choice([3, 3, 4]))
    tile = int(g.integers(40, 95))
    gap = int(g.integers(2, 6))
    head = int(g.integers(50, 95))
    foot = int(g.integers(40, 60))
    w = n * (tile + gap) + gap + 8
    h = head + n * (tile + gap) + gap + foot
    img = np.empty((h, w, 3), np.uint8)
    img[:] = col(g, 235, 255)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), col(g, 190, 220), 1)
    hc = col(g, 20, 210)
    cv2.rectangle(img, (4, 4), (w - 5, head - 4), hc, -1)
    f = int(g.choice(UI_FONTS))
    put_fit(img, str(g.choice(["Select all squares with", "Select all images with", "Click each tile containing"])),
            (12, 10, w - 24, (head - 14) // 3), f, (255, 255, 255))
    put_fit(img, str(g.choice(NOUNS)), (12, 10 + (head - 14) // 3 + 4, int(w * 0.6), (head - 14) // 2), f, (255, 255, 255), 2)
    single = g.random() < 0.5
    gx, gy = 4 + gap, head + gap
    gsize = n * (tile + gap) - gap
    if single:
        img[gy:gy + gsize, gx:gx + gsize] = photo(g, gsize, gsize)
        for i in range(1, n):
            p = i * (tile + gap) - gap // 2 - 1
            cv2.line(img, (gx + p, gy), (gx + p, gy + gsize), (255, 255, 255), gap)
            cv2.line(img, (gx, gy + p), (gx + gsize, gy + p), (255, 255, 255), gap)
    else:
        for r_ in range(n):
            for c_ in range(n):
                x, y = gx + c_ * (tile + gap), gy + r_ * (tile + gap)
                img[y:y + tile, x:x + tile] = photo(g, tile, tile)
    sel = 0
    for r_ in range(n):
        for c_ in range(n):
            if g.random() < 0.2:
                sel += 1
                x, y = gx + c_ * (tile + gap), gy + r_ * (tile + gap)
                cv2.circle(img, (x + 10, y + 10), 8, hc, -1, AA)
                cv2.polylines(img, [np.array([[x + 6, y + 10], [x + 9, y + 13], [x + 14, y + 7]], np.int32)], False, (255, 255, 255), 2, AA)
    fy = h - foot
    cv2.line(img, (0, fy), (w, fy), col(g, 200, 225), 1)
    for i in range(int(g.integers(0, 4))):
        cv2.circle(img, (18 + i * 26, fy + foot // 2), 8, col(g, 90, 140), 2, AA)
    button(img, g, (w - 100, fy + 8, 90, foot - 16), str(g.choice(["VERIFY", "SKIP", "NEXT"])), hc, (255, 255, 255))
    return page_pad(img, g, 0.25), {"n": n, "single_image": single, "selected": sel}


def r_recaptcha(g):
    s = g.uniform(0.75, 1.4)
    w, h = int(304 * s * g.uniform(0.9, 1.1)), int(78 * s * g.uniform(0.9, 1.1))
    dark = g.random() < 0.25
    img, fg = widget_frame(g, w, h, dark)
    if g.random() < 0.6:  # soft shadow
        cv2.line(img, (1, h - 2), (w - 2, h - 2), col(g, 170, 200), 2)
    cb = int(h * 0.36)
    state = str(g.choice(["empty", "empty", "empty", "spinner", "checked"]))
    checkbox(img, g, int(h * 0.17), (h - cb) // 2, cb, state, dark)
    put_fit(img, str(g.choice(["I'm not a robot", "I am not a robot"])), (int(h * 0.17) + cb + 12, int(h * 0.3), int(w * 0.45), int(h * 0.4)),
            int(g.choice(UI_FONTS)), fg)
    lx, ly, lr = int(w * 0.84), int(h * 0.4), int(h * 0.17)
    c1 = col(g, 150, 230) if g.random() < 0.5 else (200, 120, 60)
    cv2.ellipse(img, (lx, ly), (lr, lr), 0, 200, 340, c1, max(2, lr // 4), AA)
    cv2.ellipse(img, (lx, ly), (lr, lr), 0, 20, 160, col(g, 120, 200), max(2, lr // 4), AA)
    cv2.ellipse(img, (lx, ly), (lr, lr), 0, 340, 380, (160, 160, 160), max(2, lr // 4), AA)
    put_fit(img, "Privacy - Terms", (int(w * 0.74), int(h * 0.72), int(w * 0.22), int(h * 0.14)), cv2.FONT_HERSHEY_PLAIN, (130, 130, 130))
    return page_pad(img, g), {"dark": dark, "state": state, "scale": round(float(s), 2)}


def r_hcaptcha(g):
    s = g.uniform(0.75, 1.4)
    w, h = int(300 * s * g.uniform(0.9, 1.1)), int(74 * s * g.uniform(0.9, 1.1))
    dark = g.random() < 0.3
    img, fg = widget_frame(g, w, h, dark)
    cb = int(h * 0.38)
    state = str(g.choice(["empty", "empty", "empty", "spinner", "checked"]))
    checkbox(img, g, int(h * 0.2), (h - cb) // 2, cb, state, dark)
    put_fit(img, str(g.choice(["I am human", "I'm human"])), (int(h * 0.2) + cb + 14, int(h * 0.3), int(w * 0.38), int(h * 0.38)),
            int(g.choice(UI_FONTS)), fg)
    lx, ly, d = int(w * 0.85), int(h * 0.38), int(h * 0.12)
    base_h = int(g.integers(80, 100))
    for i in range(3):  # stacked rounded diamonds in teal/blue tones
        c = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[base_h + i * 6, 180, 200 - i * 30]]]), cv2.COLOR_HSV2BGR)[0, 0])
        pts = np.array([[lx, ly - d + i * d // 2], [lx + d, ly + i * d // 2], [lx, ly + d + i * d // 2], [lx - d, ly + i * d // 2]], np.int32)
        cv2.fillPoly(img, [pts], c, AA)
    put_fit(img, "Privacy - Terms", (int(w * 0.74), int(h * 0.74), int(w * 0.22), int(h * 0.13)), cv2.FONT_HERSHEY_PLAIN, (130, 130, 130))
    return page_pad(img, g), {"dark": dark, "state": state, "scale": round(float(s), 2)}


def r_turnstile(g):
    s = g.uniform(0.75, 1.4)
    w, h = int(300 * s * g.uniform(0.9, 1.1)), int(65 * s * g.uniform(0.9, 1.1))
    dark = g.random() < 0.35
    img, fg = widget_frame(g, w, h, dark)
    state = str(g.choice(["verify", "verify", "verifying", "success"]))
    cx, cy, r = int(h * 0.42), h // 2, int(h * 0.2)
    if state == "verify":
        cv2.rectangle(img, (cx - r, cy - r), (cx + r, cy + r), col(g, 120, 180), 2)
        label = "Verify you are human"
    elif state == "verifying":
        cv2.ellipse(img, (cx, cy), (r, r), 0, float(g.uniform(0, 360)), float(g.uniform(220, 320)), (60, 140, 230), 3, AA)
        label = "Verifying..."
    else:
        cv2.circle(img, (cx, cy), r, (70, 170, 60), -1, AA)
        cv2.polylines(img, [np.array([[cx - r // 2, cy], [cx - r // 8, cy + r // 2], [cx + r // 2, cy - r // 3]], np.int32)],
                      False, (255, 255, 255), 2, AA)
        label = "Success!"
    put_fit(img, label, (cx + r + 12, int(h * 0.3), int(w * 0.45), int(h * 0.4)), int(g.choice(UI_FONTS)), fg)
    lx, ly = int(w * 0.84), int(h * 0.38)  # generic cloud shape in warm tones
    warm = tuple(int(v) for v in cv2.cvtColor(np.uint8([[[int(g.integers(5, 22)), 200, 240]]]), cv2.COLOR_HSV2BGR)[0, 0])
    u = max(3, int(h * 0.09))
    for dx, dy, rr in ((-u, u // 2, u), (0, 0, int(u * 1.3)), (int(u * 1.2), u // 2, u)):
        cv2.circle(img, (lx + dx, ly + dy), rr, warm, -1, AA)
    put_fit(img, "Privacy - Terms", (int(w * 0.72), int(h * 0.72), int(w * 0.24), int(h * 0.15)), cv2.FONT_HERSHEY_PLAIN, (130, 130, 130))
    return page_pad(img, g), {"dark": dark, "state": state, "scale": round(float(s), 2)}


def r_none(g):
    kind = str(g.choice(["login_form", "paragraph", "button", "photo", "logo", "table", "chart", "navbar",
                         "labeled_input", "gallery", "checkbox_option", "settings_row", "search_bar", "card"]))
    f = int(g.choice(UI_FONTS))
    dark = g.random() < 0.2
    words = lambda k: " ".join(g.choice(WORDS, k))  # noqa: E731
    if kind == "photo":
        w, h = int(g.integers(120, 400)), int(g.integers(80, 300))
        return photo(g, w, h), {"kind": kind}
    if kind == "gallery":  # images without any challenge prompt - hard negative
        k, t, gap = int(g.integers(2, 6)), int(g.integers(60, 110)), int(g.integers(4, 14))
        img, fg = widget_frame(g, k * (t + gap) + gap, t + 2 * gap + int(g.integers(0, 30)), dark)
        for i in range(k):
            img[gap:gap + t, gap + i * (t + gap):gap + i * (t + gap) + t] = photo(g, t, t)
        return img, {"kind": kind}
    if kind == "table":  # grid lines with text - hard negative for tile grids
        rws, cls = int(g.integers(3, 8)), int(g.integers(2, 5))
        cw, rh = int(g.integers(60, 120)), int(g.integers(20, 34))
        img, fg = widget_frame(g, cls * cw + 2, rws * rh + 2, dark)
        for r_ in range(rws):
            if r_ == 0:
                cv2.rectangle(img, (1, 1), (cls * cw, rh), col(g, 180, 230), -1)
            for c_ in range(cls):
                cv2.rectangle(img, (c_ * cw, r_ * rh), ((c_ + 1) * cw, (r_ + 1) * rh), col(g, 150, 200), 1)
                put_fit(img, words(1) if r_ == 0 else str(int(g.integers(1, 99999))), (c_ * cw + 5, r_ * rh + 4, cw - 10, rh - 8), f, fg)
        return img, {"kind": kind}
    if kind == "chart":
        w, h = int(g.integers(180, 360)), int(g.integers(120, 240))
        img, fg = widget_frame(g, w, h, dark)
        n = int(g.integers(4, 12))
        bw = (w - 40) // n
        for i in range(n):
            bh = int(g.integers(10, h - 40))
            cv2.rectangle(img, (30 + i * bw, h - 20 - bh), (30 + i * bw + bw - 4, h - 20), col(g, 30, 220), -1)
        cv2.line(img, (28, 10), (28, h - 20), fg, 1)
        cv2.line(img, (28, h - 20), (w - 8, h - 20), fg, 1)
        return img, {"kind": kind}
    if kind == "checkbox_option":  # "Remember me"/"I agree" rows - hard negative for checkbox widgets
        s = g.uniform(0.75, 1.4)
        w, h = int(g.integers(220, 340) * s), int(g.integers(50, 80) * s)
        img, fg = widget_frame(g, w, h, dark)
        cb = int(h * 0.36)
        checkbox(img, g, int(h * 0.2), (h - cb) // 2, cb, str(g.choice(["empty", "checked"])), dark)
        put_fit(img, str(g.choice(["Remember me", "I agree to the terms", "Keep me signed in", "Subscribe to updates",
                                   "Send me a copy", "Accept cookies"])),
                (int(h * 0.2) + cb + 12, int(h * 0.3), int(w * 0.6), int(h * 0.4)), f, fg)
        return page_pad(img, g), {"kind": kind}
    if kind == "settings_row":
        w, h = int(g.integers(240, 380)), int(g.integers(44, 70))
        img, fg = widget_frame(g, w, h, dark)
        put_fit(img, words(2).capitalize(), (12, int(h * 0.3), int(w * 0.5), int(h * 0.4)), f, fg)
        tx, ty = w - 60, h // 2
        on = g.random() < 0.5
        cv2.rectangle(img, (tx, ty - 10), (tx + 40, ty + 10), (80, 170, 60) if on else (170, 170, 170), -1)
        cv2.circle(img, (tx + (30 if on else 10), ty), 8, (255, 255, 255), -1, AA)
        return img, {"kind": kind}
    if kind == "button":
        w, h = int(g.integers(90, 240)), int(g.integers(30, 60))
        img = np.empty((h, w, 3), np.uint8)
        img[:] = col(g, 200, 255)
        button(img, g, (2, 2, w - 4, h - 4), str(g.choice(["Submit", "Continue", "Sign in", "Next", "Save", "Download"])),
               col(g, 20, 200), (255, 255, 255))
        return img, {"kind": kind}
    if kind == "logo":
        w, h = int(g.integers(120, 300)), int(g.integers(50, 100))
        mode, bg, fg = palette(g)
        img, _ = background(g, w, h, bg)
        cv2.circle(img, (h // 2, h // 2), h // 3, col(g), -1, AA)
        put_fit(img, str(g.choice(WORDS)).upper(), (h, h // 4, w - h - 10, h // 2), int(g.choice(FONTS[:6])), fg, 2)
        return img, {"kind": kind}
    if kind == "navbar":
        w, h = int(g.integers(300, 520)), int(g.integers(36, 56))
        img = np.empty((h, w, 3), np.uint8)
        img[:] = col(g, 20, 240)
        fg = (255, 255, 255) if np.mean(img[0, 0]) < 128 else (30, 30, 30)
        x = 10
        for _ in range(int(g.integers(3, 6))):
            word = str(g.choice(WORDS)).capitalize()
            put(img, word, x, h // 2 + 6, f, 0.55, fg, 1)
            x += 18 + cv2.getTextSize(word, f, 0.55, 1)[0][0]
        return img, {"kind": kind}
    # forms and text blocks
    w, h = int(g.integers(220, 420)), int(g.integers(60, 260))
    img, fg = widget_frame(g, w, h, dark)
    if kind == "paragraph":
        y = 20
        while y < h - 8:
            put(img, words(int(g.integers(3, 8))), 10, y, f, float(g.uniform(0.4, 0.6)), fg, 1)
            y += int(g.integers(16, 24))
    elif kind in ("login_form", "labeled_input", "search_bar", "card"):
        if kind == "card":
            img[8:8 + h // 2, 8:w - 8] = photo(g, w - 16, h // 2)
            put_fit(img, words(3).capitalize(), (10, h // 2 + 16, w - 20, 22), f, fg)
        elif kind == "search_bar":
            input_box(img, g, (10, h // 2 - 14, int(w * 0.7), 28), "Search " + str(g.choice(WORDS)), dark)
            button(img, g, (int(w * 0.72) + 10, h // 2 - 14, int(w * 0.24), 28), "Search", col(g, 20, 200), (255, 255, 255))
        else:
            fields = 1 if kind == "labeled_input" else int(g.integers(2, 4))
            y = 10
            for _ in range(fields):
                if y + 50 > h:
                    break
                put_fit(img, str(g.choice(["Enter your name", "Email address", "Password", "Order number",
                                           "Enter your reference", "Customer ID"])), (10, y, int(w * 0.6), 16), f, fg)
                input_box(img, g, (10, y + 20, w - 20, 24), "", dark)
                y += 52
            if y + 30 < h:
                button(img, g, (10, y, 100, 28), str(g.choice(["Sign in", "Submit", "Continue"])), col(g, 20, 200), (255, 255, 255))
    return page_pad(img, g), {"kind": kind}


RENDERERS = {"text": r_text, "text_distorted": r_text_distorted, "math": r_math, "audio": r_audio,
             "image_classification": r_image_classification, "grid": r_grid, "recaptcha": r_recaptcha,
             "hcaptcha": r_hcaptcha, "turnstile": r_turnstile, "none": r_none}


def finalize(img, g, category):
    """Global variation applied to every category so no single nuisance factor identifies a class."""
    s = float(g.uniform(0.6, 1.5))
    h, w = img.shape[:2]
    img = cv2.resize(img, (max(24, int(w * s)), max(16, int(h * s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if g.random() < 0.5:
        img = cv2.convertScaleAbs(img, alpha=float(g.uniform(0.85, 1.15)), beta=float(g.uniform(-20, 20)))
    if g.random() < 0.3:
        widget = category in ("recaptcha", "hcaptcha", "turnstile", "grid", "audio", "image_classification", "none")
        a = float(g.uniform(-2, 2) if widget else g.uniform(-4, 4))
        hh, ww = img.shape[:2]
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D((ww / 2, hh / 2), a, 1.0), (ww, hh), borderMode=cv2.BORDER_REPLICATE)
    if g.random() < 0.35:
        img = np.clip(img.astype(np.int16) + g.normal(0, g.uniform(2, 9), img.shape), 0, 255).astype(np.uint8)
    if g.random() < 0.15:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    h, w = img.shape[:2]
    if w > 640 or h > 640:
        f = 640 / max(w, h)
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return img, round(s, 3)


def dhash(img) -> str:
    gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def render_sample(category: str, seed_seq: list[int], quality: int):
    """Render one sample; returns (jpeg bytes, width, height, params)."""
    g = np.random.default_rng(seed_seq)
    img, params = RENDERERS[category](g)
    img, scale = finalize(img, g, category)
    q = int(np.clip(quality + g.integers(-15, 8), 60, 95))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    params.update(final_scale=scale, jpeg_quality=q)
    return buf.tobytes(), img.shape[1], img.shape[0], params, dhash(img)


# --------------------------------------------------------------------------- dataset building
def _counts(total: int, categories: list[str]) -> dict[str, int]:
    base, extra = divmod(total, len(categories))
    return {c: base + (1 if i < extra else 0) for i, c in enumerate(categories)}


def _split_sizes(n: int, split: dict) -> tuple[int, int, int]:
    val, test = int(round(n * split["validation"])), int(round(n * split["test"]))
    return n - val - test, val, test


def _make_entry(cfg, sample_id, category, split, rel, w, h, params, sha, dh, source, seed_seq):
    ds = cfg["dataset"]
    return {"sample_id": sample_id, "category": category, "split": split, "file": rel, "width": w, "height": h,
            "source": source, "license": ds["license"], "authorization_ref": ds["authorization_ref"],
            "label": category, "content_label": None, "created_at": now_iso(), "sha256": sha, "dhash": dh,
            "generator": {"version": GENERATOR_VERSION, "seed": seed_seq, "params": params}}


def _write_sample(paths, cfg, category, split, sample_id, seed_seq, seen_sha, seen_dh, source):
    quality = int(cfg["dataset"].get("jpeg_quality", 88))
    for attempt in range(25):
        data, w, h, params, dh = render_sample(category, seed_seq + [attempt], quality)
        sha = sha256_bytes(data)
        if sha in seen_sha or dh in seen_dh:
            continue  # duplicate or perceptual duplicate - draw again
        seen_sha.add(sha)
        seen_dh.add(dh)
        rel = f"{split}/{sample_id}.jpg"  # flat layout keeps Windows paths under 260 chars
        out = paths.datasets / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        params["attempt"] = attempt
        return _make_entry(cfg, sample_id, category, split, rel, w, h, params, sha, dh, source, seed_seq)
    raise RuntimeError(f"could not produce a unique sample for {sample_id}")


def generate(count: int, cfg: dict | None = None, seed: int | None = None, force: bool = False,
             progress: bool = True) -> dict:
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    if paths.manifest.exists() and not force:
        m = read_json(paths.manifest)
        if m.get("base_count") == count:
            if progress:
                print(f"dataset already present ({len(m['samples'])} samples) - use --force to regenerate")
            return m
    check_disk(paths.datasets, cfg)
    seed = int(cfg["seed"] if seed is None else seed)
    cats = list(cfg["dataset"]["categories"])
    for split in ("train", "validation", "test"):
        d = paths.datasets / split
        if d.exists():
            for f in d.rglob("*.jpg"):
                f.unlink()
    seen_sha: set[str] = set()
    seen_dh: set[str] = set()
    samples = []
    t0 = time.time()
    source = cfg["dataset"]["source"] + f" v{GENERATOR_VERSION}"
    for ci, (cat, n) in enumerate(_counts(count, cats).items()):
        order = np.random.default_rng([seed, ci, 999]).permutation(n)
        n_train, n_val, _ = _split_sizes(n, cfg["dataset"]["split"])
        for rank, idx in enumerate(order):
            split = "train" if rank < n_train else ("validation" if rank < n_train + n_val else "test")
            samples.append(_write_sample(paths, cfg, cat, split, f"{cat}_{int(idx):05d}", [seed, ci, int(idx)],
                                         seen_sha, seen_dh, source))
        if progress:
            print(f"  {cat:<22} {n:>5} samples  ({time.time() - t0:5.1f}s)", flush=True)
    samples.sort(key=lambda s: s["sample_id"])
    manifest = {"dataset": "captcha-type-classification", "generator_version": GENERATOR_VERSION,
                "created_at": now_iso(), "seed": seed, "base_count": count, "categories": cats,
                "license": cfg["dataset"]["license"], "authorization_ref": cfg["dataset"]["authorization_ref"],
                "notes": "Synthetic only. Labels are challenge categories; challenge text/answers are never recorded.",
                "samples": samples}
    write_json(paths.manifest, manifest)
    if progress:
        print(f"wrote {len(samples)} samples + manifest in {time.time() - t0:.1f}s -> {paths.datasets}")
    return manifest


def generate_extra(category: str, count: int, iteration: int, cfg: dict | None = None) -> list[dict]:
    """Extra training samples for one category (continuous improvement). Test/validation never change."""
    cfg = cfg or load_config()
    paths = get_paths(cfg)
    check_disk(paths.datasets, cfg)
    manifest = read_json(paths.manifest)
    if manifest is None:
        raise FileNotFoundError("base dataset missing - run generate_dataset.py first")
    cats = manifest["categories"]
    ci = cats.index(category)
    seen_sha = {s["sha256"] for s in manifest["samples"]}
    seen_dh = {s["dhash"] for s in manifest["samples"]}
    source = f"synthetic-augmentation:iteration {iteration} ({cfg['dataset']['source']} v{GENERATOR_VERSION})"
    new = [_write_sample(paths, cfg, category, "train", f"{category}_it{iteration:03d}_{i:04d}",
                         [int(manifest["seed"]), ci, 100_000 + iteration, i], seen_sha, seen_dh, source)
           for i in range(count)]
    manifest["samples"].extend(new)
    manifest.setdefault("augmentation", []).append({"iteration": iteration, "category": category,
                                                    "count": count, "created_at": now_iso()})
    write_json(paths.manifest, manifest)
    return new


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Generate the synthetic CAPTCHA-type dataset")
    ap.add_argument("--count", type=int, default=None, help="total base samples (default from config)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="regenerate even if a dataset exists")
    args = ap.parse_args(argv)
    cfg = load_config()
    generate(args.count or int(cfg["dataset"]["total_samples"]), cfg, args.seed, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
