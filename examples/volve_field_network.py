"""Volve field: the whole wellbore network in one interactive 3D figure.

Streams the public Equinor Volve EDM export (CC BY 4.0 -- attribute Equinor
and the Volve licence partners) with ``EDMReader``, builds the master-data
hierarchy + wellbore forest via ``network_from_edm`` with the definitive
surveys attached, and renders one line per wellbore trajectory (coloured per
Well, hover = wellbore name) plus site/wellhead markers. The figure is saved
as a standalone HTML file next to this script.

Requirements: ``pip install welleng plotly``. Point ``WELLENG_VOLVE_XML`` at
the ~211 MB ``Volve.xml`` export, or place it at ``data/Volve.xml``.
"""
import os
import pathlib
import sys
import warnings

from welleng.exchange.edm_stream import EDMReader
from welleng.hierarchy import Site, Well, Wellbore, network_from_edm

# --------------------------------------------------------------------------- #
# chart chrome + a fixed-order categorical palette (colour follows the Well;
# wells beyond the fixed list fall back to muted grey -- identity is always
# carried by the hover label and the legend, never by colour alone)
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
PALETTE = [
    "#2a78d6", "#008300", "#e87ba4", "#eda100",   # slots 1-8 (light mode)
    "#1baf7a", "#eb6834", "#4a3aa7", "#e34948",
    "#3987e5", "#d55181", "#c98500", "#199e70",   # darker steps of the same
    "#d95926", "#9085e9", "#e66767",              # hues extend the fixed order
]
OVERFLOW = INK_MUTED


def resolve_volve_path() -> str | None:
    """Return the Volve.xml path, or None with a message if unavailable."""
    path = os.environ.get(
        "WELLENG_VOLVE_XML",
        str(pathlib.Path(__file__).resolve().parent.parent / "data" / "Volve.xml"),
    )
    if os.path.isfile(path):
        return path
    print(
        "Volve.xml not found. Download the public Volve dataset (Equinor, "
        "CC BY 4.0) and set WELLENG_VOLVE_XML to its path, or place it at "
        f"{path}."
    )
    return None


def parent_well(node: Wellbore) -> Well | None:
    """Walk the parent chain of a wellbore up to its Well."""
    p = node.parent
    while p is not None and not isinstance(p, Well):
        p = p.parent
    return p


def main() -> int:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print(
            "plotly is required for this example: pip install plotly "
            "(or welleng[easy])."
        )
        return 1

    path = resolve_volve_path()
    if path is None:
        return 0

    print("Indexing the EDM export (one streaming sweep)...")
    reader = EDMReader(path)
    print(
        f"  {len(reader.wells)} wells / {len(reader.wellbores)} wellbores on "
        f"site(s): {', '.join(s['site_name'] for s in reader.sites.values())}"
    )

    print("Building the well network with definitive surveys...")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        net = network_from_edm(reader, surveys=True)
    wellbores = [n for n in net._nodes.values() if isinstance(n, Wellbore)]
    surveyed = [wb for wb in wellbores if wb.survey is not None]
    print(
        f"  {len(surveyed)}/{len(wellbores)} wellbores carry a definitive "
        f"ACTUAL survey ({len(caught)} skipped with warnings)."
    )

    # fixed colour assignment: wells sorted by name, in palette order
    well_names = sorted({
        w.name for w in (parent_well(wb) for wb in surveyed) if w is not None
    })
    colour = {
        name: (PALETTE[i] if i < len(PALETTE) else OVERFLOW)
        for i, name in enumerate(well_names)
    }

    fig = go.Figure()
    seen_wells: set[str] = set()
    for wb in sorted(surveyed, key=lambda n: n.name):
        well = parent_well(wb)
        wname = well.name if well is not None else "unknown"
        # field-frame position: survey offsets are well-local; shift by the
        # well's slot offset from the site origin (all metres)
        slot_ns, slot_ew = (well.slot if well is not None and well.slot
                            else (0.0, 0.0))
        s = wb.survey
        fig.add_trace(go.Scatter3d(
            x=s.e + slot_ew, y=s.n + slot_ns, z=s.tvd,
            mode="lines",
            line=dict(color=colour.get(wname, OVERFLOW), width=3),
            name=wname,
            legendgroup=wname,
            showlegend=wname not in seen_wells,
            text=[wb.name] * len(s.md),
            customdata=s.md,
            hovertemplate=(
                "%{text}<br>MD %{customdata:.0f} m | TVD %{z:.0f} m"
                "<extra>" + wname + "</extra>"
            ),
        ))
        seen_wells.add(wname)

    # site origin + wellhead slot markers (surface reference layer)
    sites = [n for n in net._nodes.values() if isinstance(n, Site)]
    fig.add_trace(go.Scatter3d(
        x=[0.0], y=[0.0], z=[0.0],
        mode="markers+text",
        marker=dict(size=8, color=INK_PRIMARY, symbol="diamond"),
        text=[sites[0].name if sites else "site"],
        textposition="top center",
        textfont=dict(color=INK_MUTED, size=11),
        name="site origin",
        hovertemplate="%{text}<extra>site origin</extra>",
    ))
    slot_wells = [
        w for w in net._nodes.values()
        if isinstance(w, Well) and w.slot is not None and w.name in colour
    ]
    fig.add_trace(go.Scatter3d(
        x=[w.slot[1] for w in slot_wells],
        y=[w.slot[0] for w in slot_wells],
        z=[0.0] * len(slot_wells),
        mode="markers",
        marker=dict(size=4, color=INK_MUTED, symbol="circle"),
        name="well slots",
        text=[w.name for w in slot_wells],
        hovertemplate="%{text}<extra>well slot</extra>",
    ))

    axis = dict(
        showbackground=True, backgroundcolor=SURFACE,
        gridcolor=GRIDLINE, zerolinecolor=GRIDLINE,
        color=INK_MUTED,
    )
    fig.update_layout(
        title=dict(
            text="Volve field — definitive wellbore trajectories",
            font=dict(color=INK_PRIMARY),
        ),
        paper_bgcolor=SURFACE,
        font=dict(
            family='system-ui, -apple-system, "Segoe UI", sans-serif',
            color=INK_MUTED,
        ),
        scene=dict(
            xaxis=dict(title="East (m)", **axis),
            yaxis=dict(title="North (m)", **axis),
            zaxis=dict(title="TVD (m)", autorange="reversed", **axis),
            aspectmode="data",
        ),
        legend=dict(title="Well", font=dict(color=INK_PRIMARY)),
    )

    out = pathlib.Path(__file__).resolve().parent / "volve_field_network.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"Saved interactive figure to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
