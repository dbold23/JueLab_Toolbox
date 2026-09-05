"""
Every figure the lab interface draws, as Plotly figures.

One visual system is used everywhere so that the same thing always looks the
same: a strain keeps its colour on every panel, measured data is grey, the
fitted model is blue, the part of the curve that was cut away is an amber band
over a grey wash, and the four status colours (good / warning / serious /
critical) are reserved for verdicts and always carry a word as well as a hue.

Every function returns a :class:`plotly.graph_objects.Figure` and accepts
``height`` and ``width`` in pixels, so the same builder serves the app, the
exported HTML and the PNG gallery.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from .analyze import CurveResult, PlateResults
from .io import PlateRun
from .media import (CATEGORICAL_COLORS, OTHER_COLOR, TREATMENT_TYPES, annotate,
                    media_color_map, strain_color_map, treatment_label)
from .plate import PLATE_96_WELLS, PlateKey, normalize_well
from .preprocess import ProcessedPlate

#: Every colour used by the figures, keyed by the job it does.
PALETTE: Dict[str, object] = {
    # identity: assigned in this fixed order, never cycled past eight
    "categorical": list(CATEGORICAL_COLORS),
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "magenta": "#e87ba4", "green": "#008300", "violet": "#4a3aa7", "red": "#e34948",
    "other": OTHER_COLOR,
    # magnitude: one hue, light to dark
    "sequential": ["#cde2fb", "#a8cbf5", "#7fb0e9", "#4a86d3", "#255ea8", "#0d366b"],
    # state: reserved, always shown with a label
    "good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b",
    # surfaces and ink
    "surface": "#fcfcfb", "text": "#0b0b0b", "text_secondary": "#52514e", "text_muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7",
    # roles inside a curve figure
    "data": "#898781", "fit": "#2a78d6", "cut": "#fab219", "excluded": "#e1e0d9",
}

#: Colour scale for magnitude (single hue, light to dark).
SEQUENTIAL_SCALE = [[i / (len(PALETTE["sequential"]) - 1), c]  # type: ignore[arg-type]
                    for i, c in enumerate(PALETTE["sequential"])]  # type: ignore[index]

#: Status colour and plain-English word for each verdict.
VERDICT_STYLE: Dict[str, Tuple[str, str]] = {
    "GOOD": (PALETTE["good"], "GOOD"),          # type: ignore[index]
    "BAD": (PALETTE["critical"], "BAD"),        # type: ignore[index]
    "BORDERLINE": (PALETTE["warning"], "BORDERLINE"),  # type: ignore[index]
    "NOT FITTED": (PALETTE["axis"], "not fitted"),     # type: ignore[index]
}

FONT_FAMILY = "Inter, Helvetica Neue, Helvetica, Arial, sans-serif"
_ROWS = "ABCDEFGH"
_COLS = list(range(1, 13))


# =============================================================================
# theme and small helpers
# =============================================================================

#: Axis styling shared by every figure: hairline solid grid, muted rule, no zero line.
AXIS_STYLE = dict(
    showgrid=True, gridcolor=PALETTE["grid"], gridwidth=1, griddash="solid",
    zeroline=False, showline=True, linecolor=PALETTE["axis"], linewidth=1,
    ticks="outside", tickcolor=PALETTE["axis"], ticklen=4,
    tickfont=dict(size=11, color=PALETTE["text_secondary"]),
    title=dict(font=dict(size=12, color=PALETTE["text_secondary"])),
)


def _build_template() -> go.layout.Template:
    """The lab's Plotly template: built once, then handed to every figure."""
    template = copy.deepcopy(pio.templates["plotly_white"])
    template.layout.update(
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        font=dict(family=FONT_FAMILY, size=12, color=PALETTE["text"]),
        title=dict(font=dict(family=FONT_FAMILY, size=15, color=PALETTE["text"]),
                   x=0.0, xanchor="left"),
        colorway=list(CATEGORICAL_COLORS),
        legend=dict(font=dict(size=11, color=PALETTE["text_secondary"]),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0, itemsizing="constant"),
        hoverlabel=dict(bgcolor=PALETTE["surface"], bordercolor=PALETTE["axis"],
                        font=dict(family=FONT_FAMILY, size=12, color=PALETTE["text"]), align="left"),
        margin=dict(l=64, r=26, t=54, b=48),
        hovermode="closest",
        xaxis=AXIS_STYLE,
        yaxis=AXIS_STYLE,
    )
    return template


#: Registered so exported HTML and any other Plotly figure can ask for ``template='tecan'``.
TEMPLATE = _build_template()
pio.templates["tecan"] = TEMPLATE


def apply_theme(fig: go.Figure) -> go.Figure:
    """
    Put a figure on the lab's visual system: off-white surface, hairline solid
    grid, muted axis ink, one type family. Call it on every figure; per-figure
    tweaks go after it.
    """
    fig.update_layout(template="tecan")
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    """``'#2a78d6', 0.1`` -> ``'rgba(42,120,214,0.1)'``."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha:g})"


def _ink_on(hex_color: str) -> str:
    """Readable text colour (white or near-black) for a filled marker."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return "#ffffff" if luminance < 0.62 else PALETTE["text"]  # type: ignore[return-value]


def _num(value: object, digits: int = 3) -> str:
    """Format a number for a tooltip; ``'n/a'`` when it is missing."""
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}g}"


def _marker_size(n_points: int) -> int:
    """Marker diameter that keeps a dense series readable."""
    if n_points <= 60:
        return 9
    if n_points <= 150:
        return 7
    return 5


def _empty(message: str, height: Optional[int] = None, width: Optional[int] = None) -> go.Figure:
    """A blank panel carrying one explanatory sentence instead of a chart."""
    fig = go.Figure()
    apply_theme(fig)
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=13, color=PALETTE["text_secondary"]))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height or 220, width=width)
    return fig


def _size(fig: go.Figure, height: Optional[int], width: Optional[int], default_height: int) -> go.Figure:
    fig.update_layout(height=height or default_height, width=width)
    return fig


def _strain_colors(strains: Sequence[str]) -> Dict[str, str]:
    """Strain colours in a stable (alphabetical) slot order, so panels agree."""
    return strain_color_map(sorted({str(s) for s in strains}))


def _media_colors(medias: Sequence[str]) -> Dict[str, str]:
    """Media colours in the order the media first appear in the key."""
    return media_color_map([str(m) for m in medias])


def _verdict_of(cr: CurveResult) -> str:
    """``'GOOD'``, ``'BAD'`` or ``'BORDERLINE'`` for one analysed curve."""
    ml = cr.row.get("classification")
    if ml in VERDICT_STYLE:
        return str(ml)
    return "GOOD" if cr.is_good else "BAD"


def _short_reason(cr: CurveResult, limit: int = 72) -> str:
    reason = str(cr.legacy.classification.reason or "").replace("BAD: ", "").replace("GOOD: ", "")
    reason = reason.strip() or ("all quality gates passed" if cr.is_good else "no reason recorded")
    return reason if len(reason) <= limit else reason[: limit - 1] + "…"


# =============================================================================
# plate-level figures
# =============================================================================

def plate_grid(
    key: Optional[PlateKey] = None,
    results: Optional[PlateResults] = None,
    run: Optional[PlateRun] = None,
    mode: str = "layout",
    selected: Sequence[str] = (),
    replicate_flags: Optional[Dict[str, str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    """
    The 96-well plate as it sits on the bench: row A at the top, column 1 at
    the left, one circle per well.

    Parameters
    ----------
    key : PlateKey, optional
        Well assignments. Without it every well is drawn empty.
    results : PlateResults, optional
        Needed for ``mode='verdict'``.
    run : PlateRun, optional
        Raw data; used by ``mode='max_od'``.
    mode : {'layout', 'verdict', 'max_od'}
        ``layout`` colours by media and writes the strain in the well
        (blank wells carry a bold **B**, unused wells are hollow);
        ``verdict`` colours by GOOD / BAD / BORDERLINE / not fitted;
        ``max_od`` shades by the highest raw OD the well reached.
    selected : sequence of str
        Wells to ring in dark ink (the app's paint selection).
    replicate_flags : dict, optional
        ``{'B7': 'deviates from its replicates'}`` — draws a warning tick on
        the well and puts the message in its tooltip.

    The single data trace carries one point per well with ``customdata`` set to
    the well id, so a click or box-select in Streamlit returns well names.
    """
    if mode not in ("layout", "verdict", "max_od"):
        raise ValueError("mode must be 'layout', 'verdict' or 'max_od'")
    flags = dict(replicate_flags or {})
    selected_set = {normalize_well(w) for w in selected} - {None}

    info = _well_info(key, results, run)
    xs, ys, fills, lines, line_widths, texts, text_colors, hovers, customdata = [], [], [], [], [], [], [], [], []
    numeric_color: List[float] = []

    media_colors = _media_colors(list(key.media_names) if key is not None else [])
    max_od_values = [i["max_od"] for i in info.values() if i["max_od"] is not None]
    od_hi = max(max_od_values) if max_od_values else 1.0

    for well in PLATE_96_WELLS:
        row, col = well[0], int(well[1:])
        if row not in _ROWS or col not in _COLS:
            continue
        item = info[well]
        fill, text, hover = _well_style(item, mode, media_colors, od_hi)
        xs.append(col)
        ys.append(len(_ROWS) - _ROWS.index(row))
        fills.append(fill)
        numeric_color.append(item["max_od"] if item["max_od"] is not None else float("nan"))
        outline = PALETTE["text"] if well in selected_set else PALETTE["axis"]
        lines.append(outline)
        line_widths.append(3.0 if well in selected_set else 1.2)
        texts.append(text)
        text_colors.append(_ink_on(fill) if fill.startswith("#") else PALETTE["text_secondary"])
        if well in flags:
            hover += f"<br>check: {flags[well]}"
        hovers.append(hover)
        customdata.append(well)

    marker = dict(size=30, line=dict(color=lines, width=line_widths), symbol="circle")
    if mode == "max_od":
        marker.update(color=numeric_color, colorscale=SEQUENTIAL_SCALE, cmin=0.0, cmax=od_hi,
                      colorbar=dict(title=dict(text="max OD", font=dict(size=11, color=PALETTE["text_secondary"])),
                                    thickness=10, len=0.72, outlinewidth=0,
                                    tickfont=dict(size=10, color=PALETTE["text_secondary"])))
    else:
        marker.update(color=fills)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", marker=marker,
        text=texts, textposition="middle center",
        textfont=dict(size=9, color=text_colors, family=FONT_FAMILY),
        hovertext=hovers, hovertemplate="%{hovertext}<extra></extra>",
        customdata=customdata, name="wells", showlegend=False,
    ))
    if flags:
        fx = [int(w[1:]) + 0.31 for w in flags if w in info]
        fy = [len(_ROWS) - _ROWS.index(w[0]) + 0.29 for w in flags if w in info]
        ftext = [f"{w}: {flags[w]}" for w in flags if w in info]
        fig.add_trace(go.Scatter(
            x=fx, y=fy, mode="markers", name="check this well",
            marker=dict(symbol="triangle-up", size=10, color=PALETTE["warning"],
                        line=dict(color=PALETTE["surface"], width=1.5)),
            hovertext=ftext, hovertemplate="%{hovertext}<extra></extra>", showlegend=True,
        ))

    for label, colour in _grid_legend(mode, info, media_colors).items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=label, hoverinfo="skip",
            marker=dict(size=11, color=colour, line=dict(color=PALETTE["axis"], width=1.2)),
        ))

    titles = {"layout": "Plate layout", "verdict": "Verdict per well", "max_od": "Highest OD reached"}
    apply_theme(fig)
    fig.update_layout(
        title=dict(text=titles[mode]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
        margin=dict(l=34, r=26, t=54, b=70),
    )
    fig.update_xaxes(range=[0.4, 12.6], tickvals=_COLS, ticktext=[str(c) for c in _COLS],
                     side="top", showgrid=False, showline=False, ticks="", zeroline=False)
    fig.update_yaxes(range=[0.4, 8.6], tickvals=list(range(8, 0, -1)), ticktext=list(_ROWS),
                     showgrid=False, showline=False, ticks="", zeroline=False, scaleanchor="x", scaleratio=1)
    return _size(fig, height, width, 430)


def _well_info(key: Optional[PlateKey], results: Optional[PlateResults],
               run: Optional[PlateRun]) -> Dict[str, Dict[str, object]]:
    """Gather everything known about each of the 96 wells."""
    info: Dict[str, Dict[str, object]] = {
        w: {"well": w, "media": "", "strain": "", "is_blank": False, "in_use": False,
            "curve_id": None, "verdict": None, "reason": "", "max_od": None}
        for w in PLATE_96_WELLS
    }
    if key is not None:
        for rec in key.table.itertuples():
            if rec.well in info:
                info[rec.well].update(media=rec.media, strain=rec.strain,
                                      is_blank=bool(rec.is_blank), in_use=bool(rec.in_use))
    if results is not None:
        for cid, cr in results.curves.items():
            for w in cr.wells:
                if w in info:
                    info[w].update(curve_id=cid, verdict=_verdict_of(cr), reason=_short_reason(cr))
    if run is not None:
        for w in run.well_ids:
            if w in info:
                column = run.wells[w].to_numpy(dtype=float)
                if np.isfinite(column).any():
                    info[w]["max_od"] = float(np.nanmax(column))
    elif results is not None:
        for cid, cr in results.curves.items():
            value = cr.row.get("max_od")
            for w in cr.wells:
                if w in info and value is not None and np.isfinite(value):
                    info[w]["max_od"] = float(value)
    return info


def _well_style(item: Dict[str, object], mode: str, media_colors: Dict[str, str],
                od_hi: float) -> Tuple[str, str, str]:
    """Fill colour, in-well label and tooltip for one well."""
    well, media, strain = item["well"], str(item["media"]), str(item["strain"])
    head = f"<b>{well}</b>"
    if not item["in_use"]:
        return _rgba(PALETTE["grid"], 0.35), "", f"{head}<br>empty well"  # type: ignore[arg-type]
    label = f"{head}<br>{media} · {strain}"

    if mode == "verdict":
        if item["is_blank"]:
            return _rgba(PALETTE["grid"], 0.5), "<b>B</b>", f"{label}<br>blank (media only)"  # type: ignore[arg-type]
        verdict = item["verdict"]
        if verdict is None:
            colour, word = VERDICT_STYLE["NOT FITTED"]
            return _rgba(colour, 0.45), strain[:5], f"{label}<br>{word}"
        colour, word = VERDICT_STYLE[str(verdict)]
        return colour, strain[:5], f"{label}<br><b>{word}</b> — {item['reason']}"

    if mode == "max_od":
        value = item["max_od"]
        text = "" if value is None else f"{float(value):.2f}"
        shade = 0.0 if value is None else min(float(value) / od_hi, 1.0) if od_hi else 0.0
        colour = PALETTE["sequential"][min(int(shade * 5), 5)]  # type: ignore[index]
        return colour, text, f"max OD {_num(value, 3)}<br>{label}"

    colour = media_colors.get(media, OTHER_COLOR)
    if item["is_blank"]:
        return _rgba(colour, 0.32), "<b>B</b>", f"{label}<br>blank (media only)"
    return colour, strain[:5], label


def _grid_legend(mode: str, info: Dict[str, Dict[str, object]],
                 media_colors: Dict[str, str]) -> Dict[str, str]:
    """Legend entries for the plate grid, in a stable order."""
    if mode == "max_od":
        return {}
    if mode == "verdict":
        present = {str(i["verdict"]) for i in info.values() if i["verdict"] is not None}
        out = {word: colour for key_, (colour, word) in VERDICT_STYLE.items() if key_ in present}
        if any(i["in_use"] and i["verdict"] is None and not i["is_blank"] for i in info.values()):
            colour, word = VERDICT_STYLE["NOT FITTED"]
            out[word] = _rgba(colour, 0.45)
        return out
    used = [str(i["media"]) for i in info.values() if i["in_use"]]
    return {m: media_colors.get(m, OTHER_COLOR) for m in dict.fromkeys(used)}


def delta_od_heatmap(
    run: PlateRun,
    key: Optional[PlateKey] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    """
    How much each well grew, straight from the raw data: highest OD minus the
    average of the first five cycles. Dead wells, edge effects and a
    contaminated blank all show up here before any model is fitted.
    """
    wells = run.wells
    delta: Dict[str, float] = {}
    for w in run.well_ids:
        column = wells[w].to_numpy(dtype=float)
        if not np.isfinite(column).any():
            continue
        baseline = float(np.nanmean(column[:5])) if len(column) >= 5 else float(np.nanmin(column))
        delta[w] = float(np.nanmax(column) - baseline)

    key_rows = {rec.well: rec for rec in key.table.itertuples()} if key is not None else {}
    z, hover, custom = [], [], []
    for row in _ROWS:
        z_row, hover_row, custom_row = [], [], []
        for col in _COLS:
            w = f"{row}{col}"
            value = delta.get(w)
            rec = key_rows.get(w)
            if rec is None or not rec.in_use:
                where = "not in the key" if rec is None else "empty well"
            elif rec.is_blank:
                where = f"{rec.media} · blank"
            else:
                where = f"{rec.media} · {rec.strain}"
            z_row.append(value if value is not None else None)
            hover_row.append(f"<b>ΔOD {_num(value, 3)}</b><br>{w} · {where}")
            custom_row.append(w)
        z.append(z_row)
        hover.append(hover_row)
        custom.append(custom_row)

    fig = go.Figure(go.Heatmap(
        z=z, x=[str(c) for c in _COLS], y=list(_ROWS), text=hover, customdata=custom,
        colorscale=SEQUENTIAL_SCALE, hovertemplate="%{text}<extra></extra>",
        xgap=2, ygap=2, zmin=0,
        colorbar=dict(title=dict(text="ΔOD", font=dict(size=11, color=PALETTE["text_secondary"])),
                      thickness=10, len=0.8, outlinewidth=0,
                      tickfont=dict(size=10, color=PALETTE["text_secondary"])),
        name="growth",
    ))
    blanks = [w for w, rec in key_rows.items() if rec.in_use and rec.is_blank]
    if blanks:
        fig.add_trace(go.Scatter(
            x=[str(int(w[1:])) for w in blanks], y=[w[0] for w in blanks], mode="text",
            text=["<b>B</b>"] * len(blanks), textfont=dict(size=10, color=PALETTE["text"]),
            hovertext=[f"{w} · blank well" for w in blanks],
            hovertemplate="%{hovertext}<extra></extra>", showlegend=False, name="blank wells",
        ))
    apply_theme(fig)
    fig.update_layout(title=dict(text="Growth per well (raw ΔOD)"), showlegend=False,
                      margin=dict(l=34, r=26, t=54, b=40))
    fig.update_xaxes(side="top", showgrid=False, showline=False, ticks="", title_text="")
    fig.update_yaxes(autorange="reversed", showgrid=False, showline=False, ticks="", title_text="")
    return _size(fig, height, width, 400)


def replicate_facets(
    processed: ProcessedPlate,
    media: str,
    blanked: bool = True,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    """
    Do the triplicates of one media agree? One small panel per strain, one thin
    line per well, with the blank level as a dashed grey reference.
    """
    conditions = processed.conditions[processed.conditions.media == media]
    if conditions.empty:
        available = ", ".join(dict.fromkeys(processed.conditions.media))
        raise ValueError(f"No curves for media '{media}'. This plate has: {available}")

    strains = list(dict.fromkeys(conditions.strain))
    colors = _strain_colors(processed.conditions.strain)
    value_col = "od_blanked" if blanked else "od_raw"
    blank_rows = processed.blanks[processed.blanks.media == media]

    fig = make_subplots(rows=1, cols=len(strains), shared_yaxes=True, horizontal_spacing=0.035,
                        subplot_titles=strains)
    for i, strain in enumerate(strains, start=1):
        colour = colors.get(strain, OTHER_COLOR)
        reps = processed.replicates[(processed.replicates.media == media)
                                    & (processed.replicates.strain == strain)]
        for j, (well, grp) in enumerate(reps.groupby("well", sort=True)):
            fig.add_trace(go.Scatter(
                x=grp.time_h, y=grp[value_col], mode="lines", name=strain,
                legendgroup=strain, showlegend=False,
                line=dict(color=colour, width=1.5), opacity=0.85,
                hovertext=[f"well {well} · {strain}"] * len(grp),
                hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<br>%{hovertext}<extra></extra>",
            ), row=1, col=i)
        if not blank_rows.empty:
            y_blank = (np.zeros(len(blank_rows)) if blanked else blank_rows.blank_mean.to_numpy())
            fig.add_trace(go.Scatter(
                x=blank_rows.time_h, y=y_blank, mode="lines",
                name="blank mean" + (" (subtracted)" if blanked else ""),
                legendgroup="blank", showlegend=(i == 1),
                line=dict(color=PALETTE["text_muted"], width=1.5, dash="dash"),
                hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<br>blank (media only)<extra></extra>",
            ), row=1, col=i)

    apply_theme(fig)
    what = "blank-subtracted" if blanked else "raw"
    fig.update_layout(
        title=dict(text=f"{media} — replicate wells ({what})"),
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="right", x=1),
        margin=dict(l=62, r=26, t=84, b=56),
    )
    fig.update_xaxes(title_text="time (h)")
    fig.update_yaxes(title_text="OD600", col=1)
    for note in fig.layout.annotations:
        note.font = dict(size=12, color=PALETTE["text_secondary"], family=FONT_FAMILY)
    return _size(fig, height, width, 340)


def temperature_trace(
    run: PlateRun,
    target_c: Optional[float] = 37.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    """Did the incubator hold? Temperature over the run against the target band."""
    if run.temperature_c is None or not len(np.asarray(run.temperature_c)):
        return _empty("This export has no temperature column.", height, width)

    temp = np.asarray(run.temperature_c, dtype=float)
    time_h = np.asarray(run.time_h, dtype=float)[: len(temp)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_h, y=temp, mode="lines", name="incubator temperature",
        line=dict(color=PALETTE["blue"], width=2),
        hovertemplate="<b>%{y:.1f} °C</b> at %{x:.1f} h<extra></extra>",
    ))
    outside = np.array([], dtype=int)
    if target_c is not None:
        outside = np.flatnonzero(np.abs(temp - target_c) > 1.0)
        fig.add_hrect(y0=target_c - 1, y1=target_c + 1, line_width=0,
                      fillcolor=_rgba(PALETTE["good"], 0.1), layer="below")  # type: ignore[arg-type]
        fig.add_annotation(x=1.0, xref="paper", y=target_c + 1, yanchor="bottom", xanchor="right",
                           text=f"target {target_c:g} ± 1 °C", showarrow=False,
                           font=dict(size=11, color=PALETTE["text_secondary"]))
        if len(outside):
            fig.add_trace(go.Scatter(
                x=time_h[outside], y=temp[outside], mode="markers", name="outside ± 1 °C",
                marker=dict(color=PALETTE["warning"], size=8,
                            line=dict(color=PALETTE["surface"], width=2)),
                hovertemplate="<b>%{y:.1f} °C</b> at %{x:.1f} h<br>outside the target band<extra></extra>",
            ))
    apply_theme(fig)
    fig.update_layout(title=dict(text="Incubator temperature"), showlegend=bool(len(outside)),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="left", x=0))
    fig.update_xaxes(title_text="time (h)")
    fig.update_yaxes(title_text="temperature (°C)")
    return _size(fig, height, width, 280)


# =============================================================================
# per-curve figures
# =============================================================================

def pipeline_strip(
    processed: ProcessedPlate,
    cr: CurveResult,
    cut_override: Optional[Tuple[int, int]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    """
    The whole analysis of one curve in four stacked panels that share a time
    axis, so you can see exactly what the model was shown:

    1. the raw replicate wells and the blank they were compared against,
    2. their blank-subtracted average with a replicate spread ribbon,
    3. the smoothed curve, the growth window kept (amber) and the data cut away,
    4. the points that were fitted and the Gompertz curve through them.
    """
    cid = cr.curve_id
    reps = processed.replicates[processed.replicates.curve_id == cid]
    curve = processed.curves[processed.curves.curve_id == cid]
    blank_rows = processed.blanks[processed.blanks.media == cr.media]
    trunc = cr.truncation

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055,
        row_heights=[0.22, 0.24, 0.24, 0.30],
        subplot_titles=(
            "1 · Raw wells and the blank",
            "2 · Blank-subtracted average of the replicates",
            "3 · Smoothing and the growth window that was kept",
            "4 · Gompertz fit",
        ),
    )
    _panel_raw(fig, reps, blank_rows)
    _panel_blanked(fig, curve)
    _panel_truncation(fig, cr, trunc, cut_override)
    _panel_fit(fig, cr, trunc)

    apply_theme(fig)
    verdict = _verdict_of(cr)
    fig.update_layout(
        title=dict(text=f"{cid} — {verdict}"),
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="left", x=0, tracegroupgap=6),
        margin=dict(l=66, r=26, t=62, b=96),
        hovermode="closest",
    )
    fig.update_xaxes(title_text="time (h)", row=4, col=1)
    for row, label in enumerate(["raw OD600", "OD600", "OD600", "OD600"], start=1):
        fig.update_yaxes(title_text=label, row=row, col=1)
    for note in fig.layout.annotations[:4]:
        note.update(x=0.0, xanchor="left",
                    font=dict(size=12, color=PALETTE["text_secondary"], family=FONT_FAMILY))
    return _size(fig, height, width, 900)


def _panel_raw(fig: go.Figure, reps: pd.DataFrame, blank_rows: pd.DataFrame) -> None:
    for i, (well, grp) in enumerate(reps.groupby("well", sort=True)):
        fig.add_trace(go.Scatter(
            x=grp.time_h, y=grp.od_raw, mode="lines", name="replicate wells",
            legendgroup="wells", showlegend=(i == 0),
            line=dict(color=PALETTE["data"], width=1.2), opacity=0.8,
            hovertext=[f"well {well}"] * len(grp),
            hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<br>%{hovertext}<extra></extra>",
        ), row=1, col=1)
    if not blank_rows.empty:
        fig.add_trace(go.Scatter(
            x=blank_rows.time_h, y=blank_rows.blank_mean, mode="lines", name="blank mean",
            legendgroup="blank", line=dict(color=PALETTE["text_secondary"], width=1.6, dash="dash"),
            hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<br>blank (media only)<extra></extra>",
        ), row=1, col=1)


def _panel_blanked(fig: go.Figure, curve: pd.DataFrame) -> None:
    if curve.empty:
        return
    time_h = curve.time_h.to_numpy()
    od = curve.od.to_numpy()
    sd = np.nan_to_num(curve.od_sd.to_numpy(), nan=0.0)
    fig.add_trace(go.Scatter(
        x=time_h, y=od - sd, mode="lines", line=dict(width=0), hoverinfo="skip",
        showlegend=False, name="replicate spread (low)", legendgroup="spread",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=time_h, y=od + sd, mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor=_rgba(PALETTE["data"], 0.18), name="± 1 SD of the replicates",  # type: ignore[arg-type]
        legendgroup="spread", hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=time_h, y=od, mode="lines", name="blank-subtracted mean",
        line=dict(color=PALETTE["text_secondary"], width=2),
        hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<br>mean of the replicates<extra></extra>",
    ), row=2, col=1)
    clipped = np.flatnonzero((od <= 0) & ((curve.od_raw_mean.to_numpy() - curve.blank_mean.to_numpy()) < 0))
    if len(clipped):
        fig.add_trace(go.Scatter(
            x=time_h[clipped], y=od[clipped], mode="markers", name="clipped to zero",
            marker=dict(color=PALETTE["warning"], size=7, line=dict(color=PALETTE["surface"], width=1.5)),
            hovertemplate="<b>0.000 OD</b> at %{x:.1f} h<br>negative after blanking, floored at zero<extra></extra>",
        ), row=2, col=1)


def _panel_truncation(fig: go.Figure, cr: CurveResult, trunc, cut_override) -> None:
    time_h, od = cr.time_h, cr.od
    index = np.arange(len(time_h))
    fig.add_trace(go.Scatter(
        x=time_h, y=od, mode="markers", name="measured points",
        marker=dict(color=PALETTE["data"], size=5, line=dict(width=0)), opacity=0.55,
        customdata=index,
        hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<br>cycle %{customdata}<extra></extra>",
    ), row=3, col=1)
    if trunc is not None and trunc.smoothed_od is not None and len(trunc.smoothed_od):
        n = min(len(trunc.smoothed_od), len(time_h))
        fig.add_trace(go.Scatter(
            x=time_h[:n], y=np.asarray(trunc.smoothed_od)[:n], mode="lines",
            name="5-point rolling mean", line=dict(color=PALETTE["text_secondary"], width=2),
            hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<br>smoothed<extra></extra>",
        ), row=3, col=1)

    if trunc is not None and len(time_h):
        start = float(time_h[max(int(trunc.start_index), 0)])
        end = float(trunc.truncation_time)
        fig.add_vrect(x0=start, x1=end, row=3, col=1, layer="below", line_width=0,
                      fillcolor=_rgba(PALETTE["cut"], 0.16))  # type: ignore[arg-type]
        if end < float(time_h[-1]):
            fig.add_vrect(x0=end, x1=float(time_h[-1]), row=3, col=1, layer="below", line_width=0,
                          fillcolor=_rgba(PALETTE["excluded"], 0.7))  # type: ignore[arg-type]
        if start > float(time_h[0]):
            fig.add_vrect(x0=float(time_h[0]), x1=start, row=3, col=1, layer="below", line_width=0,
                          fillcolor=_rgba(PALETTE["excluded"], 0.7))  # type: ignore[arg-type]
        fig.add_annotation(row=3, col=1, x=(start + end) / 2, y=1.0, yref="y domain", yanchor="top",
                           text=f"window used: {start:.1f}–{end:.1f} h", showarrow=False,
                           font=dict(size=11, color=PALETTE["text_secondary"]))
    if cr.legacy.is_incomplete:
        fig.add_annotation(row=3, col=1, x=0.01, xref="x domain", y=0.92, yref="y domain",
                           xanchor="left", yanchor="top", text="INCOMPLETE - not truncated",
                           showarrow=False, font=dict(size=11, color=PALETTE["text"]),
                           bgcolor=_rgba(PALETTE["warning"], 0.28), borderpad=4)  # type: ignore[arg-type]
    if cut_override is not None and len(time_h):
        lo = float(time_h[int(np.clip(cut_override[0], 0, len(time_h) - 1))])
        hi = float(time_h[int(np.clip(cut_override[1], 0, len(time_h) - 1))])
        for x in (lo, hi):
            fig.add_vline(x=x, row=3, col=1, line=dict(color=PALETTE["text"], width=1.5, dash="dot"))
        fig.add_annotation(row=3, col=1, x=hi, y=0.06, yref="y domain", yanchor="bottom", xanchor="left",
                           text=" proposed cut", showarrow=False,
                           font=dict(size=11, color=PALETTE["text"]))


def _panel_fit(fig: go.Figure, cr: CurveResult, trunc) -> None:
    time_h, od = cr.time_h, cr.od
    if trunc is None or not len(getattr(trunc, "time_truncated", [])):
        fig.add_annotation(row=4, col=1, x=0.5, y=0.5, xref="x domain", yref="y domain",
                           text=cr.explanation(), showarrow=False, align="center",
                           font=dict(size=12, color=PALETTE["text_secondary"]))
        return

    start = int(trunc.start_index)
    used = np.arange(start, start + len(trunc.od_truncated))
    dropped = np.setdiff1d(np.arange(len(time_h)), used)
    if len(dropped):
        fig.add_trace(go.Scatter(
            x=time_h[dropped], y=od[dropped], mode="markers", name="not used",
            marker=dict(color=PALETTE["axis"], size=4, line=dict(width=0)), opacity=0.6,
            customdata=dropped,
            hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<br>cycle %{customdata} — outside the window<extra></extra>",
        ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=trunc.time_truncated, y=trunc.od_truncated, mode="markers", name="points used for the fit",
        marker=dict(color=PALETTE["data"], size=_marker_size(len(used)),
                    line=dict(color=PALETTE["surface"], width=1.5)),
        customdata=used,
        hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<br>cycle %{customdata}<extra></extra>",
    ), row=4, col=1)

    if not cr.fit_succeeded:
        fig.add_annotation(row=4, col=1, x=0.5, y=0.92, xref="x domain", yref="y domain",
                           text=cr.explanation(), showarrow=False, align="center",
                           font=dict(size=12, color=PALETTE["text_secondary"]))
        return

    from . import legacy  # local import keeps module import cheap
    gca = legacy.growth_curve_analysis()
    fit = cr.fit
    full_t = np.linspace(float(time_h[0]), float(time_h[-1]), 400)
    full_y = gca.gompertz_model(full_t, fit.a_opt, fit.mu_opt, fit.lambda_opt)
    fig.add_trace(go.Scatter(
        x=full_t, y=full_y, mode="lines", name="model beyond the window",
        line=dict(color=PALETTE["fit"], width=1.5, dash="dash"), opacity=0.75,
        hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<br>model, extrapolated<extra></extra>",
    ), row=4, col=1)
    fit_t, fit_y = cr.fitted_curve(240)
    fig.add_trace(go.Scatter(
        x=fit_t, y=fit_y, mode="lines", name="Gompertz fit",
        line=dict(color=PALETTE["fit"], width=2),
        hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<br>fitted model<extra></extra>",
    ), row=4, col=1)

    fig.add_hline(y=fit.a_opt, row=4, col=1,
                  line=dict(color=PALETTE["text_secondary"], width=1, dash="dash"))
    fig.add_annotation(row=4, col=1, x=1.0, xref="x domain", y=fit.a_opt, xanchor="right", yanchor="bottom",
                       text=f"A = {fit.a_opt:.3g} OD", showarrow=False,
                       font=dict(size=11, color=PALETTE["text_secondary"]))
    fig.add_vline(x=fit.lambda_opt, row=4, col=1,
                  line=dict(color=PALETTE["text_secondary"], width=1, dash="dot"))
    fig.add_annotation(row=4, col=1, x=fit.lambda_opt, y=1.0, yref="y domain", xanchor="left", yanchor="top",
                       text=f" λ = {fit.lambda_opt:.3g} h", showarrow=False,
                       font=dict(size=11, color=PALETTE["text_secondary"]))
    if fit.mu_opt > 0:
        t_end = min(fit.lambda_opt + fit.a_opt / fit.mu_opt, float(time_h[-1]))
        fig.add_trace(go.Scatter(
            x=[fit.lambda_opt, t_end], y=[0.0, fit.mu_opt * (t_end - fit.lambda_opt)],
            mode="lines", name="μ (steepest slope)",
            line=dict(color=PALETTE["fit"], width=1.5, dash="dot"),
            hovertemplate=f"<b>μ = {fit.mu_opt:.3g} OD/h</b><br>steepest slope of the fit<extra></extra>",
        ), row=4, col=1)


def residual_plot(cr: CurveResult, height: Optional[int] = None, width: Optional[int] = None) -> go.Figure:
    """
    What the model missed. Residuals should scatter around zero like noise; a
    smooth arc through them means the S-curve is the wrong shape for this well.
    """
    if not cr.fit_succeeded or cr.truncation is None:
        return _empty("No fit, so there are no residuals to look at.", height, width)

    time_h = np.asarray(cr.truncation.time_truncated, dtype=float)
    residuals = np.asarray(cr.fit.residuals, dtype=float)
    n = min(len(time_h), len(residuals))
    time_h, residuals = time_h[:n], residuals[:n]
    rmse = float(cr.fit.rmse)
    autocorr = cr.row.get("residual_autocorr")
    limit = cr.settings.max_residual_autocorr

    fig = go.Figure()
    fig.add_hrect(y0=-2 * rmse, y1=2 * rmse, line_width=0,
                  fillcolor=_rgba(PALETTE["data"], 0.12), layer="below")  # type: ignore[arg-type]
    fig.add_hline(y=0, line=dict(color=PALETTE["axis"], width=1))
    fig.add_trace(go.Scatter(
        x=time_h, y=residuals, mode="markers", name="residual",
        marker=dict(color=PALETTE["data"], size=_marker_size(n),
                    line=dict(color=PALETTE["surface"], width=1.5)),
        hovertemplate="<b>%{y:+.4f} OD</b> at %{x:.1f} h<br>measured minus model<extra></extra>",
    ))
    apply_theme(fig)
    fig.add_annotation(x=1.0, xref="paper", y=2 * rmse, xanchor="right", yanchor="bottom",
                       text=f"± 2 × RMSE ({2 * rmse:.3f} OD)", showarrow=False,
                       font=dict(size=11, color=PALETTE["text_secondary"]))
    if autocorr is not None and np.isfinite(autocorr):
        passed = float(autocorr) <= limit
        fig.add_annotation(x=0.0, xref="paper", y=1.06, yref="paper", xanchor="left", yanchor="bottom",
                           text=f"lag-1 autocorrelation {float(autocorr):.2f} "
                                f"({'within' if passed else 'above'} the limit of {limit:g})",
                           showarrow=False, font=dict(size=11, color=PALETTE["text"]),
                           bgcolor=_rgba(PALETTE["good"] if passed else PALETTE["warning"], 0.18),  # type: ignore[arg-type]
                           borderpad=3)
    fig.update_layout(title=dict(text=f"{cr.curve_id} — residuals"), showlegend=False,
                      margin=dict(l=66, r=26, t=64, b=48))
    fig.update_xaxes(title_text="time (h)")
    fig.update_yaxes(title_text="measured − model (OD)")
    return _size(fig, height, width, 280)


def curve_sparkline(cr: CurveResult, height: Optional[int] = None, width: Optional[int] = None) -> go.Figure:
    """A thumbnail of one curve for a card list: the data, the fit, and a rule in the verdict colour."""
    verdict = _verdict_of(cr)
    colour, word = VERDICT_STYLE[verdict]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cr.time_h, y=cr.od, mode="lines", name="measured",
        line=dict(color=PALETTE["data"], width=1.5),
        hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<extra></extra>",
    ))
    if cr.fit_succeeded:
        fit_t, fit_y = cr.fitted_curve(160)
        fig.add_trace(go.Scatter(
            x=fit_t, y=fit_y, mode="lines", name="fit",
            line=dict(color=PALETTE["fit"], width=1.5),
            hovertemplate="<b>%{y:.3f} OD</b> at %{x:.1f} h<br>fit<extra></extra>",
        ))
    apply_theme(fig)
    fig.add_shape(type="line", xref="paper", yref="paper", x0=0, x1=1, y0=0, y1=0,
                  line=dict(color=colour, width=3))
    fig.add_annotation(x=0.0, xref="paper", y=1.0, yref="paper", xanchor="left", yanchor="top",
                       text=word, showarrow=False, font=dict(size=10, color=PALETTE["text_secondary"]))
    fig.update_layout(showlegend=False, margin=dict(l=2, r=2, t=2, b=4), title=None)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _size(fig, height, width, 58)


def gate_table(cr: CurveResult) -> pd.DataFrame:
    """
    The scorecard behind the verdict: every check, the number it saw, the
    threshold it compared against, and whether it passed. Feed it straight to
    ``st.dataframe``.
    """
    baseline_std = cr.legacy.classification.metrics.get("baseline_std")
    clipped = baseline_std is not None and np.isfinite(baseline_std) and float(baseline_std) <= 1e-6
    rows = []
    for gate in cr.gates:
        value, note = gate.value, gate.note
        if gate.name in ("snr", "prefit_snr") and (clipped or (value is not None and abs(value) > 1e6)):
            value = None
            note = "SNR not meaningful (baseline clipped to zero)"
        if gate.passed is None or not gate.applies:
            status = "BYPASSED" if "bypass" in gate.note.lower() else "NOT APPLIED"
        else:
            status = "PASS" if gate.passed else "FAIL"
        rows.append({
            "gate": gate.name,
            "description": gate.description,
            "value": None if value is None else round(float(value), 4),
            "threshold": None if gate.threshold is None else round(float(gate.threshold), 4),
            "comparison": gate.comparison,
            "status": status,
            "note": note,
        })
    return pd.DataFrame(rows, columns=["gate", "description", "value", "threshold",
                                       "comparison", "status", "note"])


# =============================================================================
# comparison figures
# =============================================================================

#: Axis labels for the parameters the Compare page plots.
PARAMETER_LABELS: Dict[str, str] = {
    "gompertz_a": "A — highest OD the culture reached (OD600)",
    "gompertz_mu": "μ — fastest growth rate (OD/h)",
    "gompertz_lambda": "λ — lag before growth (h)",
    "doubling_time_h": "doubling time at μ (h)",
    "fit_r_squared": "R² of the fit",
    "max_od": "highest measured OD600",
    "delta_od": "total growth ΔOD",
}


def _require_treatment(table: pd.DataFrame) -> pd.DataFrame:
    """Annotate the table with treatment types if the caller has not already."""
    if "treatment_type" in table.columns:
        return table
    return annotate(table)


def _call_of(table: pd.DataFrame) -> pd.Series:
    """GOOD/BAD per row, honouring a hand override in ``final_call`` when present."""
    if "final_call" in table.columns:
        return table.final_call.astype(str).str.upper()
    if "is_good" in table.columns:
        return table.is_good.map(lambda v: "GOOD" if bool(v) else "BAD")
    return pd.Series(["GOOD"] * len(table), index=table.index)


def parameter_dotplot(
    table: pd.DataFrame,
    parameter: str = "gompertz_mu",
    include_bad: bool = False,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    """
    One Gompertz parameter across the four kinds of well, one column per
    treatment type and one colour per strain, with standard-error bars.

    ``table`` is a results table; :func:`tecan_growth.media.annotate` is applied
    automatically if it has no ``treatment_type`` column. Curves the pipeline
    called BAD are hidden unless ``include_bad=True``, and then drawn hollow.
    """
    if parameter not in table.columns:
        raise KeyError(f"Column '{parameter}' is not in the results table")
    data = _require_treatment(table).copy()
    data["call"] = _call_of(data)
    data = data[pd.to_numeric(data[parameter], errors="coerce").notna()]
    if not include_bad:
        data = data[data.call == "GOOD"]
    if data.empty:
        return _empty("No curves to compare with the current filter.", height, width)

    err_col = f"{parameter}_err"
    types = [t for t in TREATMENT_TYPES if t in set(data.treatment_type)]
    types += sorted(set(data.treatment_type) - set(types))
    strains = sorted(set(data.strain.astype(str)))
    colors = _strain_colors(strains)
    spread = min(0.62 / max(len(strains), 1), 0.16)

    fig = go.Figure()
    for k, strain in enumerate(strains):
        sub = data[data.strain.astype(str) == strain]
        offset = (k - (len(strains) - 1) / 2) * spread
        xs = [types.index(t) + offset for t in sub.treatment_type]
        colour = colors[strain]
        fills = [colour if c == "GOOD" else "rgba(0,0,0,0)" for c in sub.call]
        errors = (pd.to_numeric(sub[err_col], errors="coerce").to_numpy()
                  if err_col in sub.columns else None)
        fig.add_trace(go.Scatter(
            x=xs, y=pd.to_numeric(sub[parameter], errors="coerce"), mode="markers", name=strain,
            marker=dict(size=11, color=fills, line=dict(color=colour, width=2)),
            error_y=(dict(type="data", array=np.nan_to_num(errors, nan=0.0), visible=True,
                          color=PALETTE["axis"], thickness=1.2, width=4)
                     if errors is not None else None),
            customdata=np.stack([sub.media.astype(str), sub.call, sub.curve_id.astype(str)], axis=-1),
            hovertemplate=("<b>%{y:.4g}</b><br>%{customdata[2]}<br>%{customdata[0]} · "
                           "%{customdata[1]}<extra></extra>"),
        ))
    apply_theme(fig)
    fig.update_layout(
        title=dict(text=PARAMETER_LABELS.get(parameter, parameter)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0),
        margin=dict(l=70, r=26, t=58, b=84),
    )
    fig.update_xaxes(tickvals=list(range(len(types))), ticktext=types,
                     range=[-0.6, len(types) - 0.4], showgrid=False, title_text="what was in the well")
    fig.update_yaxes(title_text=PARAMETER_LABELS.get(parameter, parameter).split(" — ")[-1])
    if include_bad:
        fig.add_annotation(x=1.0, xref="paper", y=1.02, yref="paper", xanchor="right", yanchor="bottom",
                           text="hollow = called BAD, number not trustworthy", showarrow=False,
                           font=dict(size=11, color=PALETTE["text_secondary"]))
    return _size(fig, height, width, 420)


def ratio_strip(
    table: pd.DataFrame,
    include_bad: bool = True,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    """
    Two questions on one strip, per strain: did the pesticide slow growth down
    when food was available (μ with pesticide+LB divided by μ in LB), and did
    the pesticide feed the cells on its own (A in pesticide divided by A in
    water)? A ratio of 1 means no difference.
    """
    data = _require_treatment(table).copy()
    data["call"] = _call_of(data)
    if not include_bad:
        data = data[data.call == "GOOD"]

    rows = _ratio_rows(data)
    if not rows:
        return _empty("Not enough fitted curves to build the ratios "
                      "(each strain needs its LB, water and pesticide wells).", height, width)

    frame = pd.DataFrame(rows)
    counts = frame.groupby("strain").pesticide.nunique()
    frame["label"] = [f"{r.strain} · {r.pesticide}" if counts[r.strain] > 1 else r.strain
                      for r in frame.itertuples()]
    labels = list(dict.fromkeys(frame.label))[::-1]
    colors = _strain_colors(frame.strain)

    panels = [("mu", "μ(pesticide + LB) / μ(LB)"), ("a", "A(pesticide) / A(water)")]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=[title for _, title in panels])
    for col, (kind, _) in enumerate(panels, start=1):
        sub = frame[frame.kind == kind]
        stem_x, stem_y = [], []
        for r in sub.itertuples():
            stem_x += [1.0, r.ratio, None]
            stem_y += [r.label, r.label, None]
        fig.add_trace(go.Scatter(x=stem_x, y=stem_y, mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=PALETTE["grid"], width=2), name="distance from 1"),
                      row=1, col=col)
        for strain in sorted(set(sub.strain)):
            part = sub[sub.strain == strain]
            colour = colors[strain]
            fills = [colour if ok else "rgba(0,0,0,0)" for ok in part.both_good]
            fig.add_trace(go.Scatter(
                x=part.ratio, y=part.label, mode="markers", name=strain, legendgroup=strain,
                showlegend=(col == 1),
                marker=dict(size=12, color=fills, line=dict(color=colour, width=2)),
                customdata=np.stack([part.detail, part.call], axis=-1),
                hovertemplate="<b>%{x:.2f}×</b><br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
            ), row=1, col=col)
        fig.add_vline(x=1.0, row=1, col=col, line=dict(color=PALETTE["text_muted"], width=1))

    apply_theme(fig)
    fig.update_layout(
        title=dict(text="Pesticide effect, as a ratio to the matching control"),
        legend=dict(orientation="h", yanchor="top", y=-0.42, xanchor="left", x=0),
        margin=dict(l=140, r=26, t=76, b=118),
    )
    fig.update_xaxes(title_text="ratio (1 = no change)", rangemode="tozero")
    fig.update_yaxes(categoryorder="array", categoryarray=labels, showgrid=False, title_text="")
    for note in fig.layout.annotations[:2]:
        note.font = dict(size=12, color=PALETTE["text_secondary"], family=FONT_FAMILY)
    return _size(fig, height, width, max(300, 130 + 34 * len(labels)))


def _ratio_rows(data: pd.DataFrame) -> List[Dict[str, object]]:
    """Build one row per (strain, pesticide, ratio kind) with the numbers behind it."""
    rows: List[Dict[str, object]] = []
    for strain, grp in data.groupby(data.strain.astype(str)):
        by_type = {t: g for t, g in grp.groupby("treatment_type")}
        lb = by_type.get("LB")
        water = by_type.get("H2O")
        for kind, numerator_type, denominator, column, what in (
            ("mu", "Pesticide+LB", lb, "gompertz_mu", "μ"),
            ("a", "Pesticide", water, "gompertz_a", "A"),
        ):
            if denominator is None or denominator.empty:
                continue
            base_row = denominator.iloc[0]
            base = pd.to_numeric(pd.Series([base_row[column]]), errors="coerce").iloc[0]
            if not np.isfinite(base) or base <= 0:
                continue
            for r in (by_type.get(numerator_type, pd.DataFrame())).itertuples():
                value = pd.to_numeric(pd.Series([getattr(r, column)]), errors="coerce").iloc[0]
                if not np.isfinite(value):
                    continue
                both_good = (str(r.call) == "GOOD") and (str(base_row["call"]) == "GOOD")
                rows.append({
                    "strain": strain, "pesticide": str(r.pesticide), "kind": kind,
                    "ratio": float(value / base), "both_good": both_good,
                    "detail": f"{what} {value:.3g} ({r.media}) ÷ {base:.3g} ({base_row['media']})",
                    "call": "both curves GOOD" if both_good else "one of the two curves is BAD",
                })
    return rows


def strain_overlay(
    processed: ProcessedPlate,
    results: PlateResults,
    strain: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> go.Figure:
    """
    One strain in every media on the same axes: the blank-subtracted curves as
    thin lines with their fitted models on top, so the shapes can be compared
    directly.
    """
    conditions = processed.conditions[processed.conditions.strain.astype(str) == str(strain)]
    if conditions.empty:
        available = ", ".join(dict.fromkeys(processed.conditions.strain.astype(str)))
        raise ValueError(f"No curves for strain '{strain}'. This plate has: {available}")

    colors = _media_colors(list(dict.fromkeys(processed.conditions.media.astype(str))))
    fig = go.Figure()
    for rec in conditions.itertuples():
        colour = colors.get(str(rec.media), OTHER_COLOR)
        time_h, od = processed.curve(rec.curve_id)
        label = treatment_label(str(rec.media))
        fig.add_trace(go.Scatter(
            x=time_h, y=od, mode="lines", name=label, legendgroup=rec.curve_id,
            line=dict(color=colour, width=1.5), opacity=0.55,
            hovertemplate=f"<b>%{{y:.3f}} OD</b> at %{{x:.1f}} h<br>{label} · measured<extra></extra>",
        ))
        cr = results.curves.get(rec.curve_id)
        if cr is None or not cr.fit_succeeded:
            continue
        fit_t, fit_y = cr.fitted_curve(240)
        verdict = _verdict_of(cr)
        fig.add_trace(go.Scatter(
            x=fit_t, y=fit_y, mode="lines", name=f"{label} fit", legendgroup=rec.curve_id,
            showlegend=False,
            line=dict(color=colour, width=2, dash="solid" if verdict == "GOOD" else "dot"),
            hovertemplate=(f"<b>%{{y:.3f}} OD</b> at %{{x:.1f}} h<br>{label} · fit ({verdict})"
                           "<extra></extra>"),
        ))
    apply_theme(fig)
    fig.update_layout(
        title=dict(text=f"{strain} in every media"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0),
        margin=dict(l=66, r=26, t=58, b=84),
    )
    fig.add_annotation(x=1.0, xref="paper", y=1.02, yref="paper", xanchor="right", yanchor="bottom",
                       text="thin line = measured, thick = fitted model (dotted when BAD)",
                       showarrow=False, font=dict(size=11, color=PALETTE["text_secondary"]))
    fig.update_xaxes(title_text="time (h)")
    fig.update_yaxes(title_text="blank-subtracted OD600")
    return _size(fig, height, width, 420)
