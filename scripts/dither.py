# /// script
# requires-python = ">=3.9"
# dependencies = ["pillow", "numpy"]
# ///
import datetime as dt
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONTS = ROOT / "assets" / "fonts"
OUT = ROOT / "assets" / "dither"

NAME = "ANTHONY KROEGER"
LOGIN = "anthnykr"
W, H = 415, 90
THEMES = {
    "light": {"ink": (31, 35, 40), "accent": (232, 93, 4)},
    "dark": {"ink": (230, 237, 243), "accent": (255, 122, 26)},
}


def bayer(n):
    m = np.array([[0]])
    while m.shape[0] < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]])
    return (m + 0.5) / m.size


THRESHOLD = np.tile(bayer(8), (H // 8 + 1, W // 8 + 1))[:H, :W]
YY = np.arange(H, dtype=float)[:, None] * np.ones((1, W))
XX = np.ones((H, 1)) * np.arange(W, dtype=float)[None, :]


def fit_size(text, font_file, max_w, max_h, ss):
    size = 4
    while True:
        l, t, r, b = ImageFont.truetype(str(FONTS / font_file), (size + 1) * ss).getbbox(text)
        if r - l > max_w * ss or b - t > max_h * ss:
            return size
        size += 1


def text_cover(text, font_file, max_w, max_h, pos=None, ss=4):
    font = ImageFont.truetype(str(FONTS / font_file), fit_size(text, font_file, max_w, max_h, ss) * ss)
    img = Image.new("L", (W * ss, H * ss), 0)
    draw = ImageDraw.Draw(img)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    x, y = ((W * ss - (r - l)) // 2, (H * ss - (b - t)) // 2) if pos is None else (pos[0] * ss, pos[1] * ss)
    draw.text((x - l, y - t), text, font=font, fill=255)
    return np.asarray(img.resize((W, H), Image.BOX), dtype=float) / 255


def pixel_text(text, pos, size=8):
    img = Image.new("L", (W, H), 0)
    draw = ImageDraw.Draw(img)
    draw.fontmode = "1"
    font = ImageFont.truetype(str(FONTS / "Silkscreen-Regular.ttf"), size)
    l, t, _, _ = draw.textbbox((0, 0), text, font=font)
    draw.text((pos[0] - l, pos[1] - t), text, font=font, fill=255)
    return np.asarray(img) > 127


def blur(a, radius):
    img = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(radius)), dtype=float) / 255


def shifted(a, dy, dx):
    out = np.zeros_like(a)
    out[dy:, dx:] = a[: H - dy, : W - dx]
    return out


def dilate(mask, n):
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.filter(ImageFilter.MaxFilter(2 * n + 1))) > 127


def compose(layers):
    index = np.zeros((H, W), dtype=np.uint8)
    for mask, field, color in layers:
        index[mask] = np.where(field[mask] > THRESHOLD[mask], color, 0)
    return index


def to_image(index, theme, cell):
    palette = [0, 0, 0, *THEMES[theme]["ink"], *THEMES[theme]["accent"]]
    img = Image.fromarray(index, "P")
    img.putpalette(palette + [0] * (768 - len(palette)))
    img.info["transparency"] = 0
    return img.resize((W * cell, H * cell), Image.NEAREST)


INK, ACCENT = 1, 2
FRAMES = 110
GROW = 18
FADE_OUT = 10


def contributions():
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    query = "query($login:String!){user(login:$login){contributionsCollection{contributionCalendar{totalContributions weeks{contributionDays{date contributionCount}}}}}}"
    request = urllib.request.Request(
        "https://api.github.com/graphql",
        data=json.dumps({"query": query, "variables": {"login": LOGIN}}).encode(),
        headers={"Authorization": f"bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        calendar = json.load(response)["data"]["user"]["contributionsCollection"]["contributionCalendar"]
    counts = [day["contributionCount"] for week in calendar["weeks"] for day in week["contributionDays"]]
    return calendar["totalContributions"], np.array(counts, dtype=float)


def ridge(counts, window, base, height):
    padded = np.pad(counts, (window // 2, window - 1 - window // 2), mode="edge")
    smooth = np.convolve(padded, np.ones(window) / window, mode="valid")
    smooth = np.interp(np.linspace(0, len(smooth) - 1, W), np.arange(len(smooth)), smooth)
    return H - base - height * np.sqrt(smooth / max(smooth.max(), 1e-9))


def scene():
    total, counts = contributions()
    near_top = ridge(counts, 7, 4, H * 0.42)[None, :]
    far_top = ridge(counts, 30, 16, H * 0.5)[None, :]
    best_week = max(np.convolve(counts, np.ones(7), mode="valid").max(), 1)
    name = text_cover(NAME, "ArchivoBlack-Regular.ttf", W * 0.55, 18, pos=(10, 8)) > 0.5
    name_bottom = np.where(name.any(axis=1))[0][-1]
    updated = pixel_text(f"UPDATED {dt.date.today():%Y-%m-%d}", (10, name_bottom + 16))
    rows, cols = np.where(updated)
    cursor = (YY >= rows.min()) & (YY <= rows.max()) & (XX >= cols.max() + 3) & (XX <= cols.max() + 7)
    text = name | updated | pixel_text(f"{total:,} CONTRIBUTIONS IN THE LAST YEAR", (10, name_bottom + 6))
    return {
        "near_top": near_top,
        "far_top": far_top,
        "near": 0.85 - 0.8 * np.clip((YY - near_top) / (H * 0.5), 0, 1),
        "far": 0.4 - 0.3 * np.clip((YY - far_top) / (H * 0.5), 0, 1),
        "sun_radius": 5 + 7 * min(counts[-7:].sum() / best_week, 1),
        "sun_dist": np.hypot(XX - W * 0.84, YY - H * 0.32),
        "text": text,
        "cursor": cursor,
        "knockout": dilate(text | cursor, 2),
    }


def frame(s, i):
    ease = lambda x: x * x * (3 - 2 * x)
    grow = ease(min(i / (GROW - 1), 1))
    amount = 1 - ease(max(i - (FRAMES - FADE_OUT), 0) / (FADE_OUT - 1))
    sweep = np.clip((grow * (W + 24) - XX) / 24, 0, 1)

    shine_x = -40 + (i - 30) / 50 * (W + 80)
    near = s["near"] + (0.35 * np.exp(-(((XX - shine_x) / 12) ** 2)) if 30 <= i <= 80 else 0)

    radius, dist = s["sun_radius"], s["sun_dist"]
    rings = (0.5 + 0.5 * np.sin((dist - radius) / 2 - 2 * np.pi * i / 22)) * 0.35 * np.exp(-(((dist - radius) / 24) ** 2))
    sun = np.where(dist <= radius, 0.95, np.maximum(0.4 * np.exp(-(((dist - radius) / 9) ** 2)), rings) * grow)
    sun[sun < 0.03] = 0

    text = s["text"] | (s["cursor"] & (i < GROW or (i // 8) % 2 == 0))
    layers = [
        (sun > 0, sun * amount, ACCENT),
        ((YY >= s["far_top"]) & (sweep > 0), s["far"] * sweep * amount, INK),
        ((YY >= s["near_top"]) & (sweep > 0), near * sweep * amount, ACCENT),
        (s["knockout"], np.where(text, min(1.0, i / 8) * amount, 0.0), INK),
    ]
    return compose(layers)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = scene()
    indices = [frame(s, i) for i in range(FRAMES)]
    durations = [1600 if i == FRAMES - FADE_OUT - 1 else 70 for i in range(FRAMES)]
    for theme in THEMES:
        images = [to_image(index, theme, 4) for index in indices]
        images[0].save(
            OUT / f"activity-{theme}.gif",
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0,
            disposal=2,
            transparency=0,
        )


if __name__ == "__main__":
    main()
