"""Colours and danger classes shared by the GeoTIFF styles and the animation."""

from __future__ import annotations

from dataclasses import dataclass
from xml.sax.saxutils import escape

# Six-step heat ramp, light to dark. Checked with the dataviz palette validator
# (--ordinal): monotone lightness, adjacent steps >= 0.06 apart, light end >= 2:1
# on the light surface. Multi-hue on purpose: fire danger is "semantic heat".
HEAT = ["#d9a400", "#e07a10", "#cc4a1a", "#a32a20", "#701a18", "#3d0c0c"]
# Single-hue sequential ramp (blue 250-700, ordinal-validated) for magnitudes that are not "danger",
# e.g. the day a cell first burned.
BLUES = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"        # single-series charts
SERIES_MUTED = "#9ec5f4"  # same hue, de-emphasised (non-current bars)
WATER = "#b7d3f6"
LAND = "#f0efec"


@dataclass(frozen=True)
class DangerClasses:
    """Upper bounds (exclusive) of each class; the last class is open-ended."""

    bounds: list[float]
    labels: list[str]
    source: str

    def label_for(self, value: float) -> str:
        for bound, label in zip(self.bounds, self.labels):
            if value < bound:
                return label
        return self.labels[-1]


DANGER_CLASSES: dict[str, DangerClasses] = {
    # European Forest Fire Information System (EFFIS) FWI classes.
    "fwi": DangerClasses(
        [5.2, 11.2, 21.3, 38.0, 50.0],
        ["Very low", "Low", "Moderate", "High", "Very high", "Extreme"],
        "EFFIS",
    ),
    # McArthur Forest Fire Danger Index ratings (pre-2022 Australian system).
    "ffdi": DangerClasses(
        [12, 25, 50, 75, 100],
        ["Low–moderate", "High", "Very high", "Severe", "Extreme", "Catastrophic"],
        "McArthur",
    ),
}


def _qml(renderer: str) -> str:
    return (
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
        '<qgis version="3.22" styleCategories="Symbology">\n'
        f"  <pipe>\n{renderer}  </pipe>\n"
        "</qgis>\n"
    )


def qml_classes(classes: DangerClasses, opacity: float = 0.75) -> str:
    """QGIS style: discrete danger classes on band 1."""
    items = []
    lower = None
    for bound, label, color in zip([*classes.bounds, "inf"], classes.labels, HEAT):
        rng = f"< {bound:g}" if lower is None else (f"≥ {lower:g}" if bound == "inf" else f"{lower:g}–{bound:g}")
        items.append(
            f'          <item value="{bound}" label="{escape(f"{label} ({rng})", {chr(34): "&quot;"})}" '
            f'color="{color}" alpha="255"/>\n'
        )
        lower = bound
    top = classes.bounds[-1] * 1.5
    return _qml(
        f'    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="{opacity}" alphaBand="-1" '
        f'classificationMin="0" classificationMax="{top:g}">\n'
        "      <rastershader>\n"
        f'        <colorrampshader colorRampType="DISCRETE" classificationMode="1" clip="0" '
        f'minimumValue="0" maximumValue="{top:g}">\n'
        + "".join(items)
        + "        </colorrampshader>\n      </rastershader>\n    </rasterrenderer>\n"
    )


def qml_gradient(vmin: float, vmax: float, colors: list[str] = HEAT, opacity: float = 0.75,
                 integer: bool = False) -> str:
    """QGIS style: continuous ramp from ``vmin`` to ``vmax`` on band 1."""
    vmax = vmax if vmax > vmin else vmin + 1
    n = len(colors) - 1
    values = [vmin + (vmax - vmin) * i / n for i in range(n + 1)]
    items = "".join(
        f'          <item value="{v:.6g}" label="{round(v) if integer else f"{v:.3g}"}" color="{c}" alpha="255"/>\n'
        for v, c in zip(values, colors)
    )
    return _qml(
        f'    <rasterrenderer type="singlebandpseudocolor" band="1" opacity="{opacity}" alphaBand="-1" '
        f'classificationMin="{vmin:.6g}" classificationMax="{vmax:.6g}">\n'
        "      <rastershader>\n"
        f'        <colorrampshader colorRampType="INTERPOLATED" classificationMode="1" clip="0" '
        f'minimumValue="{vmin:.6g}" maximumValue="{vmax:.6g}">\n'
        + items
        + "        </colorrampshader>\n      </rastershader>\n    </rasterrenderer>\n"
    )
