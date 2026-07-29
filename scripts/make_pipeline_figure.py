#!/usr/bin/env python
"""Turn docs/figures/pipeline.src.svg into a maximally portable pipeline.svg (+ PNG).

Why this exists
---------------
The source figure is written with a ``<style>`` block and ``feTurbulence``
filters, which is pleasant to author but does not survive contact with the real
world:

* GitHub sanitises ``<style>`` out of SVGs embedded via markdown ``![]()``.
* Most markdown previewers (VS Code, PyCharm, Obsidian, Quarto) and every SVG
  rasteriser (cairosvg, librsvg, ImageMagick) ignore SVG filter primitives, so
  every ``feTurbulence`` rainfall thumbnail renders as an empty box.

So this script rewrites the figure into the lowest common denominator:

1. resolves every CSS class into inline presentation attributes;
2. replaces each filtered rectangle with an explicit grid of coloured cells,
   generated from deterministic fractal value noise -- which also makes the
   resolution contrast (28 km / 10 km / 5 km) literal rather than implied;
3. drops the ``<style>`` and ``<filter>`` definitions entirely.

Run after editing the source:

    python scripts/make_pipeline_figure.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs/figures/pipeline.src.svg"
OUT = ROOT / "docs/figures/pipeline.svg"
PNG = ROOT / "docs/figures/pipeline.png"

PRESENTATION = {
    "fill", "stroke", "stroke-width", "stroke-dasharray", "stroke-linecap",
    "font-size", "font-weight", "font-family", "marker-end", "opacity",
}

# filter id -> (n cells per side, rgb ramp low, rgb ramp high, gamma)
TEXTURES = {
    "fCoarse": (6,  (247, 250, 253), (16, 78, 140), 1.1),   # ERA5 0.25 deg
    "fMid":    (12, (253, 249, 250), (150, 45, 90), 1.1),   # IMERG 0.1 deg
    "fFine":   (26, (250, 252, 251), (10, 74, 128), 1.2),   # CHIRPS/analysis 0.05 deg
    "fNoise":  (26, (238, 239, 242), (120, 124, 133), 1.0),  # pure Gaussian noise
    "fTerrain": (18, (246, 248, 244), (110, 88, 58), 1.0),  # orography
}


def value_noise(n: int, octaves: int, seed: int) -> np.ndarray:
    """Deterministic fractal value noise on an n x n grid, normalised to [0, 1]."""
    rng = np.random.default_rng(seed)
    out = np.zeros((n, n))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        res = max(2, int(np.ceil(n / (2 ** (octaves - 1 - o)))))
        coarse = rng.random((res, res))
        yi = np.linspace(0, res - 1, n)
        xi = np.linspace(0, res - 1, n)
        y0 = np.clip(np.floor(yi).astype(int), 0, res - 2)
        x0 = np.clip(np.floor(xi).astype(int), 0, res - 2)
        fy = (yi - y0)[:, None]
        fx = (xi - x0)[None, :]
        fy = fy * fy * (3 - 2 * fy)          # smoothstep
        fx = fx * fx * (3 - 2 * fx)
        c00 = coarse[np.ix_(y0, x0)]
        c01 = coarse[np.ix_(y0, x0 + 1)]
        c10 = coarse[np.ix_(y0 + 1, x0)]
        c11 = coarse[np.ix_(y0 + 1, x0 + 1)]
        out += amp * ((c00 * (1 - fx) + c01 * fx) * (1 - fy)
                      + (c10 * (1 - fx) + c11 * fx) * fy)
        total += amp
        amp *= 0.55
    out /= total
    return (out - out.min()) / (np.ptp(out) + 1e-9)


def texture_cells(fid: str, x: float, y: float, w: float, h: float, seed: int) -> str:
    n, lo, hi, gamma = TEXTURES[fid]
    if fid == "fNoise":
        z = np.random.default_rng(seed).random((n, n))       # white, not fractal
    else:
        z = value_noise(n, 4 if n > 12 else 2, seed) ** gamma
    cw, ch = w / n, h / n
    parts = []
    for i in range(n):
        for j in range(n):
            t = z[i, j]
            r = int(lo[0] + (hi[0] - lo[0]) * t)
            g = int(lo[1] + (hi[1] - lo[1]) * t)
            b = int(lo[2] + (hi[2] - lo[2]) * t)
            parts.append(
                f'<rect x="{x + j*cw:.2f}" y="{y + i*ch:.2f}" '
                f'width="{cw + 0.35:.2f}" height="{ch + 0.35:.2f}" fill="#{r:02x}{g:02x}{b:02x}"/>'
            )
    return "<g>" + "".join(parts) + "</g>"


def parse_style(svg: str) -> dict[str, dict[str, str]]:
    block = re.search(r"<style>(.*?)</style>", svg, re.S)
    classes: dict[str, dict[str, str]] = {}
    if not block:
        return classes
    for name, body in re.findall(r"\.([\w-]+)\s*\{([^}]*)\}", block.group(1)):
        decls = {}
        for decl in body.split(";"):
            if ":" not in decl:
                continue
            k, v = decl.split(":", 1)
            k, v = k.strip(), v.strip()
            if k in PRESENTATION:
                decls[k] = v
        classes[name] = decls
    return classes


def inline_classes(svg: str, classes: dict[str, dict[str, str]]) -> str:
    def repl(m: re.Match) -> str:
        tag = m.group(0)
        names = m.group(1).split()
        merged: dict[str, str] = {}
        for n in names:
            merged.update(classes.get(n, {}))
        # never clobber an attribute the element already sets explicitly
        for k in list(merged):
            if re.search(rf'\b{re.escape(k)}\s*=\s*"', tag):
                merged.pop(k)
        attrs = " ".join(f'{k}="{v}"' for k, v in merged.items())
        tag = re.sub(r'\s*class="[^"]*"', "", tag)
        close = "/>" if tag.rstrip().endswith("/>") else ">"
        body = tag.rstrip()[: -len(close)].rstrip()
        return body + (" " + attrs if attrs else "") + close

    return re.sub(r'<(?:rect|text|circle|path|g|tspan)\b[^>]*class="([^"]*)"[^>]*>', repl, svg)


def main() -> None:
    svg = SRC.read_text()
    classes = parse_style(svg)
    svg = inline_classes(svg, classes)

    seed = [0]

    def repl_filter(m: re.Match) -> str:
        tag, fid = m.group(0), m.group(2)
        def num(a: str) -> float:
            return float(re.search(rf'\b{a}="([-\d.]+)"', tag).group(1))
        seed[0] += 1
        return texture_cells(fid, num("x"), num("y"), num("width"), num("height"), seed[0] * 17)

    svg = re.sub(r'<rect\b([^>]*)filter="url\(#(\w+)\)"([^>]*)/>', repl_filter, svg)

    svg = re.sub(r"<style>.*?</style>", "", svg, flags=re.S)
    svg = re.sub(r"<filter id=\"\w+\".*?</filter>", "", svg, flags=re.S)
    svg = re.sub(r"\n\s*\n+", "\n", svg)
    svg = svg.replace("<defs>", '<title>BDhighresDA pipeline</title>\n  <defs>', 1)
    svg = svg.replace("<svg ", '<svg role="img" ', 1)

    OUT.write_text(svg)
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size/1024:.0f} KB)")

    try:
        import cairosvg

        cairosvg.svg2png(url=str(OUT), write_to=str(PNG), output_width=2220)
        print(f"wrote {PNG.relative_to(ROOT)}")
    except Exception as exc:
        print(f"[png] cairosvg unavailable ({exc.__class__.__name__}); "
              f"rasterise with:  rsvg-convert -w 2220 {OUT.name} -o {PNG.name}")


if __name__ == "__main__":
    main()
