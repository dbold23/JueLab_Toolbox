#!/usr/bin/env python3
"""
Draw the figures used by the "How it works" manual (docs/how_it_works) and
the README.

    python scripts/make_docs_figures.py                  # writes docs/images/*.svg and gallery.png
    python scripts/make_docs_figures.py --diagrams-only  # the SVG diagrams only (fast)

The four diagrams are plain SVG written by hand here so they render in the
app, on GitHub and in a browser without any font or library dependency. The
gallery is a matplotlib figure of three real wells from the example plate
and three generated curves, each run through the real analysis so the
annotated verdicts are the ones the software actually gives.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "docs" / "images"

# Palette shared with tecan_growth.plots
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GOOD, WARN, BAD = "#0ca30c", "#fab219", "#d03b3b"
INK, MUTED, RULE, PAPER = "#1f2328", "#6b7280", "#d0d4da", "#ffffff"
FONT = "font-family='-apple-system, Segoe UI, Helvetica, Arial, sans-serif'"


def _svg(width: int, height: int, body: str) -> str:
    return (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {width} {height}' width='{width}' height='{height}' "
            f"{FONT} font-size='14' fill='{INK}'>\n<rect width='{width}' height='{height}' fill='{PAPER}'/>\n{body}\n</svg>\n")


def _text(x, y, s, size=14, anchor="middle", weight="normal", fill=INK, style=""):
    return (f"<text x='{x}' y='{y}' text-anchor='{anchor}' font-size='{size}' font-weight='{weight}' "
            f"fill='{fill}' {style}>{s}</text>")


def _box(x, y, w, h, title, sub, color):
    r = f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='10' fill='{color}' fill-opacity='0.12' stroke='{color}' stroke-width='2'/>"
    lines = [_text(x + w / 2, y + 28, title, 15, weight="600")]
    for i, line in enumerate(sub):
        lines.append(_text(x + w / 2, y + 50 + i * 17, line, 12, fill=MUTED))
    return r + "".join(lines)


def _arrow(x1, y, x2, color=MUTED):
    return (f"<line x1='{x1}' y1='{y}' x2='{x2 - 8}' y2='{y}' stroke='{color}' stroke-width='2'/>"
            f"<polygon points='{x2 - 10},{y - 6} {x2},{y} {x2 - 10},{y + 6}' fill='{color}'/>")


def pipeline_svg() -> str:
    steps = [
        ("Plate reader export", ["OD600 of 96 wells", "every 15 min"], BLUE),
        ("Blank &amp; average", ["subtract media blank", "mean of replicates"], AQUA),
        ("Growth window", ["find where primary", "growth ends"], YELLOW),
        ("Fit Gompertz", ["A, μ, λ by", "least squares"], ORANGE),
        ("Quality gates", ["R², errors, signal,", "residual pattern"], VIOLET),
        ("Verdict &amp; export", ["GOOD / BAD with", "the reason"], GREEN),
    ]
    w, h, gap, x0, y0 = 150, 92, 22, 16, 22
    body = []
    for i, (title, sub, color) in enumerate(steps):
        x = x0 + i * (w + gap)
        body.append(_box(x, y0, w, h, title, sub, color))
        if i < len(steps) - 1:
            body.append(_arrow(x + w + 2, y0 + h / 2, x + w + gap))
    body.append(_text(x0 + 3 * (w + gap) - gap / 2, y0 + h + 30,
                      "You can override the window, exclude a well or change a verdict at any step; every override is recorded.",
                      12, fill=MUTED))
    return _svg(x0 * 2 + 6 * w + 5 * gap, y0 + h + 44, "\n".join(body))


def gompertz(t, a, mu, lam):
    return a * np.exp(-np.exp((mu * math.e / a) * (lam - t) + 1))


def gompertz_parameters_svg() -> str:
    W, H = 760, 340
    left, right, top, bottom = 70, 30, 30, 60
    a, mu, lam = 1.2, 0.35, 3.0
    t = np.linspace(0, 16, 300)
    y = gompertz(t, a, mu, lam)
    xmax, ymax = 16, 1.45
    sx = lambda v: left + v / xmax * (W - left - right)
    sy = lambda v: H - bottom - v / ymax * (H - top - bottom)
    path = " ".join(f"{'M' if i == 0 else 'L'}{sx(tt):.1f},{sy(yy):.1f}" for i, (tt, yy) in enumerate(zip(t, y)))
    # inflection point: y = A/e at t = lam + A/(mu e)
    ti = lam + a / (mu * math.e)
    yi = a / math.e
    # tangent through (ti, yi) with slope mu, from baseline to A
    t_lo, t_hi = lam, lam + a / mu
    body = [
        f"<line x1='{left}' y1='{sy(0)}' x2='{W - right}' y2='{sy(0)}' stroke='{INK}' stroke-width='1.5'/>",
        f"<line x1='{left}' y1='{sy(0)}' x2='{left}' y2='{top}' stroke='{INK}' stroke-width='1.5'/>",
        _text((left + W - right) / 2, H - 18, "time (h)", 13),
        _text(20, (top + H - bottom) / 2, "OD600", 13, style=f"transform='rotate(-90 20 {(top + H - bottom) / 2})'"),
        # A: ceiling
        f"<line x1='{left}' y1='{sy(a)}' x2='{W - right}' y2='{sy(a)}' stroke='{AQUA}' stroke-width='2' stroke-dasharray='6 5'/>",
        _text(W - right - 6, sy(a) - 8, "A: maximum OD (ceiling)", 13, anchor="end", fill=AQUA, weight="600"),
        # tangent
        f"<line x1='{sx(t_lo)}' y1='{sy(0)}' x2='{sx(t_hi)}' y2='{sy(a)}' stroke='{ORANGE}' stroke-width='2.5'/>",
        _text(sx(t_hi) + 8, sy(a * 0.90), "slope = μ, the maximum", 13, anchor="start", fill=ORANGE, weight="600"),
        _text(sx(t_hi) + 8, sy(a * 0.90) + 17, "growth rate (OD/h)", 13, anchor="start", fill=ORANGE, weight="600"),
        # inflection marker
        f"<circle cx='{sx(ti)}' cy='{sy(yi)}' r='5' fill='{ORANGE}'/>",
        _text(sx(ti) + 10, sy(yi) + 22, "inflection point, OD = A/e", 12, anchor="start", fill=MUTED),
        # lambda
        f"<line x1='{sx(lam)}' y1='{sy(0)}' x2='{sx(lam)}' y2='{sy(0.55 * a)}' stroke='{VIOLET}' stroke-width='2' stroke-dasharray='4 4'/>",
        f"<line x1='{sx(0)}' y1='{sy(0) + 22}' x2='{sx(lam)}' y2='{sy(0) + 22}' stroke='{VIOLET}' stroke-width='2'/>",
        _text(sx(lam / 2), sy(0) + 38, "λ: lag time (h)", 13, fill=VIOLET, weight="600"),
        # curve
        f"<path d='{path}' fill='none' stroke='{BLUE}' stroke-width='3'/>",
        _text(W - right - 6, sy(0.32), "y(t) = A·exp(−exp((μe/A)(λ − t) + 1))", 13, anchor="end", fill=BLUE),
        _text(W - right - 6, sy(0.18), "doubling time  t_d = ln2 · A / (μ e)", 13, anchor="end", fill=MUTED),
    ]
    return _svg(W, H, "\n".join(body))


def growth_window_svg() -> str:
    W, H = 820, 330
    left, right, top, bottom = 60, 24, 34, 60
    t = np.linspace(0, 30, 400)
    y = gompertz(t, 1.0, 0.25, 4.0)
    decline = np.where(t > 17, np.exp(-(t - 17) / 12), 1.0)
    y = y * decline
    rng = np.random.default_rng(3)
    y = y + rng.normal(0, 0.006, len(t))
    xmax, ymax = 30, 1.2
    sx = lambda v: left + v / xmax * (W - left - right)
    sy = lambda v: H - bottom - v / ymax * (H - top - bottom)
    phases = [(0, 4.5, "lag", MUTED), (4.5, 11, "exponential", ORANGE), (11, 17, "stationary", AQUA), (17, 30, "decline", RED)]
    peak_t, buffer_t = 16.2, 17.0
    body = [
        f"<line x1='{left}' y1='{sy(0)}' x2='{W - right}' y2='{sy(0)}' stroke='{INK}' stroke-width='1.5'/>",
        f"<line x1='{left}' y1='{sy(0)}' x2='{left}' y2='{top}' stroke='{INK}' stroke-width='1.5'/>",
        _text((left + W - right) / 2, H - 16, "time (h)", 13),
    ]
    for t0, t1, name, color in phases:
        body.append(f"<rect x='{sx(t0)}' y='{top}' width='{sx(t1) - sx(t0)}' height='{sy(0) - top}' fill='{color}' fill-opacity='0.07'/>")
        body.append(_text((sx(t0) + sx(t1)) / 2, top + 16, name, 12, fill=color, weight="600"))
    # fitted region
    body.append(f"<rect x='{sx(0)}' y='{top + 24}' width='{sx(buffer_t) - sx(0)}' height='15' fill='{GREEN}' fill-opacity='0.25'/>")
    body.append(_text((sx(0) + sx(buffer_t)) / 2, top + 36, "points given to the model", 11, fill=GREEN, weight="600"))
    pts = " ".join(f"{sx(tt):.1f},{sy(yy):.1f}" for tt, yy in zip(t, y))
    body.append(f"<polyline points='{pts}' fill='none' stroke='{BLUE}' stroke-width='2'/>")
    body.append(f"<line x1='{sx(peak_t)}' y1='{sy(0)}' x2='{sx(peak_t)}' y2='{sy(0.96)}' stroke='{ORANGE}' stroke-width='2' stroke-dasharray='5 4'/>")
    body.append(_text(sx(peak_t) - 6, sy(0.30), "first peak", 12, anchor="end", fill=ORANGE, weight="600"))
    body.append(f"<line x1='{sx(buffer_t)}' y1='{sy(0)}' x2='{sx(buffer_t)}' y2='{sy(0.96)}' stroke='{GREEN}' stroke-width='2'/>")
    body.append(_text(sx(buffer_t) + 6, sy(0.30), "cut, after 3 buffer points", 12, anchor="start", fill=GREEN, weight="600"))
    body.append(_text(sx(24.5), sy(0.17), "not fitted: the model", 12, fill=RED))
    body.append(_text(sx(24.5), sy(0.09), "has no decline phase", 12, fill=RED))
    body.append(_text((left + W - right) / 2, H - 36,
                      "The peak is found on a smoothed copy of the curve; the model is fitted to the raw points. A curve still rising in its last 15 % is kept whole.",
                      11.5, fill=MUTED))
    return _svg(W, H, "\n".join(body))


def verdict_tree_svg() -> str:
    W = 880
    nodes = []  # (x, y, w, h, text lines, color)

    def node(x, y, w, h, lines, color, radius=10):
        r = f"<rect x='{x}' y='{y}' width='{w}' height='{h}' rx='{radius}' fill='{color}' fill-opacity='0.12' stroke='{color}' stroke-width='2'/>"
        ts = []
        n = len(lines)
        for i, line in enumerate(lines):
            ts.append(_text(x + w / 2, y + h / 2 + (i - (n - 1) / 2) * 16 + 5, line, 12.5, weight="600" if i == 0 else "normal"))
        nodes.append(r + "".join(ts))

    def edge(x1, y1, x2, y2, label="", color=MUTED):
        nodes.append(f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2 - 8}' stroke='{color}' stroke-width='2'/>")
        nodes.append(f"<polygon points='{x2 - 5},{y2 - 9} {x2 + 5},{y2 - 9} {x2},{y2}' fill='{color}'/>")
        if label:
            nodes.append(_text((x1 + x2) / 2 + 12, (y1 + y2) / 2, label, 11.5, anchor="start", fill=color))

    cx = 300
    node(cx - 110, 16, 220, 46, ["Rise above baseline?", "SNR ≥ 3  or  Δ OD ≥ 0.5"], BLUE)
    node(cx - 110, 96, 220, 46, ["A usable window?", "≥ 20 points, peak not at start"], BLUE)
    node(cx - 110, 176, 220, 46, ["Fit converged?", "inside the allowed ranges"], BLUE)
    node(cx - 110, 256, 220, 62, ["Always-on gates", "R² ≥ 0.95, A and μ error ≤ 20 %,", "Δ OD ≥ 0.15, A ≥ 3 × noise"], VIOLET)
    node(cx - 110, 352, 220, 46, ["Excellent fit?", "R² ≥ 0.98"], BLUE)
    node(cx - 110, 432, 220, 62, ["Noise gates", "SNR ≥ 5, Δ OD − 2σ ≥ 0.1,", "monotone ≥ 0.55, autocorr ≤ 0.7"], VIOLET)
    node(cx - 60, 530, 120, 40, ["GOOD"], GOOD)

    node(640, 16, 200, 46, ["NOT FITTED", "too flat to fit"], WARN)
    node(640, 96, 200, 46, ["NOT FITTED", "no growth window"], WARN)
    node(640, 176, 200, 46, ["BAD", "fit failed"], BAD)
    node(640, 264, 200, 46, ["BAD", "with the gate that failed"], BAD)
    node(640, 440, 200, 46, ["BAD", "with the gate that failed"], BAD)

    # vertical spine
    edge(cx, 62, cx, 96, "yes")
    edge(cx, 142, cx, 176, "yes")
    edge(cx, 222, cx, 256, "yes")
    edge(cx, 318, cx, 352, "all pass")
    edge(cx, 398, cx, 432, "no")
    edge(cx, 494, cx, 530, "all pass")
    # bypass: excellent fit skips noise gates
    nodes.append(f"<path d='M{cx + 110},375 C 520,375 520,550 {cx + 60},550' fill='none' stroke='{GOOD}' stroke-width='2' stroke-dasharray='6 4'/>")
    nodes.append(_text(cx + 84, 556, "yes: noise gates skipped", 11.5, anchor="start", fill=GOOD))
    # horizontal exits
    for y, label in ((39, "no"), (119, "no"), (199, "no"), (287, "any fails"), (463, "any fails")):
        nodes.append(f"<line x1='{cx + 110}' y1='{y}' x2='632' y2='{y}' stroke='{BAD if 'fail' in label or y == 199 else WARN}' stroke-width='2'/>")
        nodes.append(f"<polygon points='630,{y - 5} 640,{y} 630,{y + 5}' fill='{BAD if 'fail' in label or y == 199 else WARN}'/>")
        nodes.append(_text(cx + 120, y - 6, label, 11.5, anchor="start", fill=MUTED))
    nodes.append(_text(W / 2, 600, "Thresholds shown are the defaults; the sliders in the sidebar change them and the manual text follows.", 11.5, fill=MUTED))
    return _svg(W, 616, "\n".join(nodes))


def gallery_png(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tecan_growth import load_plate, preprocess_plate
    from tecan_growth.analyze import analyze_curve
    from tecan_growth.settings import AnalysisSettings

    settings = AnalysisSettings.interactive_defaults()
    plate = preprocess_plate(load_plate(ROOT / "data" / "raw" / "Group 3" / "GrowthRate_GROUP3.xlsx"))

    # Pick real wells: best LB curve, a water control, the lowest-signal fitted curve.
    picks = {}
    for row in plate.conditions.itertuples():
        t, y = plate.curve(row.curve_id)
        picks[row.curve_id] = analyze_curve(t, y, row.curve_id, settings, media=row.media, strain=row.strain)
    def rise(c):
        return float(np.max(c.od) - np.mean(c.od[:5]))

    good = max((c for c in picks.values() if c.is_good and c.media == "LB"), key=lambda c: c.fit.r_squared)
    flat = min(picks.values(), key=rise)
    bad = [c for c in picks.values() if not c.is_good and c.legacy.stage == "classified" and c is not flat]
    # A real curve the single sigmoid cannot describe: the residual-pattern gate catches it.
    shape_gates = {"residual_autocorr", "r_squared"}
    wrong_shape = max((c for c in bad if {g.name for g in c.failed_gates()} & shape_gates) or bad, key=rise)

    rng = np.random.default_rng(7)
    t = np.arange(0, 40, 0.25)

    def synth(y):
        return np.clip(y + rng.normal(0, 0.005, len(t)), 0, None)

    incomplete = analyze_curve(t, synth(gompertz(t, 1.6, 0.05, 6.0)), "incomplete", settings)
    early = analyze_curve(t, synth(gompertz(t, 0.9, 0.35, 2.0) * np.where(t > 9, np.exp(-(t - 9) / 10), 1.0)), "early_peak", settings)
    diaux_y = synth(gompertz(t, 0.6, 0.2, 3.0) + gompertz(t, 0.4, 0.12, 16.0))
    diauxic = analyze_curve(t, diaux_y, "diauxic", settings)
    diauxic_global = analyze_curve(t, diaux_y, "diauxic", AnalysisSettings.interactive_defaults(use_first_peak=False))

    panels = [
        ("1  Good: clean sigmoid", good, None),
        ("2  Flat: nothing grew", flat, None),
        ("3  Wrong shape: pattern in the residuals", wrong_shape, None),
        ("4  Incomplete: still rising", incomplete, None),
        ("5  Early peak, then decline", early, None),
        ("6  Diauxic: first peak vs global max", diauxic, diauxic_global),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), dpi=130)
    for ax, (title, cr, alt) in zip(axes.ravel(), panels):
        ax.plot(cr.time_h, cr.od, ".", color="#9aa3ad", ms=3.5, label="data")
        if cr.fit_succeeded:
            ft, fy = cr.fitted_curve()
            ax.plot(ft, fy, color=BLUE, lw=2.2, label="Gompertz fit (first peak)" if alt else "Gompertz fit")
            ax.axvline(cr.truncation.truncation_time, color=BLUE, lw=1, ls=":")
        if alt is not None and alt.fit_succeeded:
            ft, fy = alt.fitted_curve()
            ax.plot(ft, fy, color=ORANGE, lw=2.2, label="Gompertz fit (global max)")
        call = "GOOD" if cr.is_good else ("NOT FITTED" if cr.legacy.stage != "classified" else "BAD")
        color = GOOD if call == "GOOD" else (WARN if call == "NOT FITTED" else BAD)
        fails = cr.failed_gates()
        if cr.legacy.stage == "prefit_rejected":
            why = "rejected before fitting: no rise above noise"
        elif fails:
            why = "; ".join(g.description for g in fails[:2])
        elif cr.fit_succeeded:
            why = f"R² = {cr.fit.r_squared:.3f}, errors {cr.legacy.classification.metrics.get('a_err_pct', float('nan')):.0f} % / {cr.legacy.classification.metrics.get('mu_err_pct', float('nan')):.0f} %"
        else:
            why = cr.legacy.classification.reason
        if cr.legacy.is_incomplete:
            why += " (A extrapolated)"
        if alt is not None:
            alt_call = "GOOD" if alt.is_good else "BAD"
            why = f"first peak: {call}; global max: {alt_call}"
            if alt.failed_gates():
                why += f" ({alt.failed_gates()[0].description})"
        ax.set_title(title, loc="left", fontsize=11, fontweight="600")
        ax.text(0.02, 0.97, call, transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold", color=color)
        ax.text(0.02, 0.86, why, transform=ax.transAxes, va="top", ha="left", fontsize=8.2, color="#374151", wrap=True)
        ax.set_xlabel("time (h)", fontsize=9)
        ax.set_ylabel("OD600", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, color="#e5e7eb", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.legend(fontsize=7.5, loc="lower right", frameon=False)
        ymax = float(np.nanmax(cr.od))
        ax.set_ylim(-0.02 * max(ymax, 0.1), max(ymax, 0.1) * 1.45)
    fig.suptitle("What the verdicts look like (panels 1-3: example plate, Group 3; panels 4-6: generated curves)", fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pipeline.svg").write_text(pipeline_svg(), encoding="utf-8")
    (OUT / "gompertz_parameters.svg").write_text(gompertz_parameters_svg(), encoding="utf-8")
    (OUT / "growth_window.svg").write_text(growth_window_svg(), encoding="utf-8")
    (OUT / "verdict_tree.svg").write_text(verdict_tree_svg(), encoding="utf-8")
    if "--diagrams-only" not in argv:
        gallery_png(OUT / "gallery.png")
    for p in sorted(OUT.glob("*")):
        if p.is_file():
            print(f"wrote {p.relative_to(ROOT)} ({p.stat().st_size // 1024} kB)")


if __name__ == "__main__":
    main()
