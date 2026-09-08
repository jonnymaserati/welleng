"""Plug-and-abandonment well-schematic example.

Builds an OSDU-lean :class:`WellSchematic` from a lean data dict, then renders:

* a **column** schematic (nested casings, cement, plugs, completion) with
  per-section radial exaggeration (shallow sections exaggerated more),
* a **section** view following the trajectory in the VS/TVD plane,
* a **composite** multi-track figure (depth | schematic | litho | seal/flow |
  pore/frac **pressure**),

each exported to DXF + SVG + PDF + PNG.

Run:  ``python examples/schematic_pa_example.py``
"""
from pathlib import Path

from welleng.schematic import (
    DepthResolver,
    PressureProfile,
    SurveyRef,
    WellFigure,
    WellSchematic,
    build_column,
    build_section,
    to_dxf,
    to_pdf,
    to_png,
    to_svg,
)

# --- OSDU-lean input -------------------------------------------------------
DATA = {
    "well": {"name": "P&A EXAMPLE 15/9-A-12"},
    "survey": {
        "md": [0, 400, 700, 1000, 2600],
        "inc": [0, 0, 15, 30, 30],
        "azi": [45, 45, 45, 45, 45],
    },
    # per-section radial exaggeration: monotonic non-increasing with depth
    "hole_sections": [
        {"bit_in": 36, "top_md": 0, "base_md": 80, "radial_scale": 50},
        {"bit_in": 26, "top_md": 80, "base_md": 400, "radial_scale": 50},
        {"bit_in": 17.5, "top_md": 400, "base_md": 1200, "radial_scale": 40},
        {"bit_in": 12.25, "top_md": 1200, "base_md": 2200, "radial_scale": 32},
        {"bit_in": 8.5, "top_md": 2200, "base_md": 2600, "radial_scale": 30},
    ],
    "casings": [
        {
            "name": '30" Cond.',
            "od_in": 30,
            "id_in": 28,
            "top_md": 0,
            "shoe_md": 80,
            "toc_md": 0,
        },
        {
            "name": '20" Surf.',
            "od_in": 20,
            "id_in": 18.7,
            "top_md": 0,
            "shoe_md": 400,
            "toc_md": 0,
        },
        {
            "name": '13-3/8"',
            "od_in": 13.375,
            "id_in": 12.4,
            "top_md": 0,
            "shoe_md": 1200,
            "toc_md": 800,
        },
        {
            "name": '9-5/8"',
            "od_in": 9.625,
            "id_in": 8.68,
            "top_md": 0,
            "shoe_md": 2200,
            "toc_md": 1400,
        },
        {
            "name": '7" Liner',
            "od_in": 7,
            "id_in": 6.18,
            "top_md": 2050,
            "shoe_md": 2600,
            "toc_md": 2050,
        },
    ],
    # What stands in each annulus where cement does not. Identified by the OD
    # of the string forming the INNER wall, and the density is stated, never
    # guessed from the name -- for a barrier argument the density is the number
    # that matters.
    "annulus_fluids": [
        {"name": "Seawater", "inside_od_in": 20, "top_md": 0, "base_md": 400,
         "density_sg": 1.03},
        {"name": "WBM", "inside_od_in": 13.375, "top_md": 0, "base_md": 800,
         "density_sg": 1.35},
        {"name": "Packer fluid (CaCl2)", "inside_od_in": 9.625,
         "top_md": 0, "base_md": 1400, "density_sg": 1.18},
    ],
    # The reservoir was perforated through the 7in liner before abandonment;
    # the plug below is set across it.
    "perforations": [
        {"top_md": 2380, "base_md": 2480, "casing_od_in": 7.0,
         "shots_per_m": 39},
    ],
    "cement_plugs": [
        {"name": "Reservoir plug", "top_md": 2350, "base_md": 2600},
        {"name": "Barrier plug", "top_md": 2100, "base_md": 2260},
        {"name": "Surface plug", "top_md": 0, "base_md": 200},
    ],
    # NO completion: this well is PLUGGED. The tubing, packer and SCSSV have
    # been pulled -- which is why plugs can be set in the bore at all. Leaving
    # them in alongside three cement plugs (as this example previously did)
    # draws a physically impossible well: you cannot set a plug through a
    # completion that is still in the hole. Wellbore now refuses that
    # combination, so the example has to pick one, and for a P&A example the
    # answer is the plugs.
    "completion": [],
    "formations": [
        {"name": "Nordland", "top_md": 0, "litho": "clay", "color": "#c9e6a8"},
        {"name": "Hordaland", "top_md": 900, "litho": "clay/silt", "color": "#a8d18d"},
        {"name": "Shetland", "top_md": 1500, "litho": "marl", "color": "#bcaaa4"},
        {
            "name": "Draupne (seal)",
            "top_md": 2150,
            "litho": "shale",
            "color": "#5e35b1",
            "seal": True,
        },
        {
            "name": "Brent (reservoir)",
            "top_md": 2300,
            "litho": "sand",
            "color": "#ffca28",
            "flow": True,
        },
        {"name": "", "top_md": 2600, "litho": "", "color": "#ffffff"},
    ],
}

# --- pore/frac PRESSURE (bar), converted from EMW (sg) via hydrostatics -----
P_MD = [0, 900, 1500, 2150, 2300, 2600]
PORE_SG = [1.03, 1.05, 1.10, 1.30, 1.62, 1.66]
FRAC_SG = [1.30, 1.45, 1.60, 1.78, 1.85, 1.90]


def build() -> WellSchematic:
    resolver = DepthResolver(SurveyRef(**DATA["survey"]))
    tvd = [float(t) for t in resolver.tvd_at(P_MD)]
    DATA["pressures"] = PressureProfile.from_emw(
        P_MD, tvd, PORE_SG, FRAC_SG, unit="bar"
    ).model_dump()
    return WellSchematic.from_dict(DATA)


def main() -> None:
    out = Path(__file__).parent / "schematic_output"
    out.mkdir(exist_ok=True)
    schematic = build()

    column = build_column(schematic, mode="MD")
    section = build_section(schematic)
    figure = WellFigure(schematic, mode="MD")

    jobs = [
        ("column", column, None),
        ("section", section, None),
        ("figure", figure.build(), None),
    ]
    for name, drawing, _ in jobs:
        to_dxf(drawing, str(out / f"pa_{name}.dxf"))
        to_svg(drawing, str(out / f"pa_{name}.svg"))
        to_pdf(drawing, str(out / f"pa_{name}.pdf"))
        to_png(drawing, str(out / f"pa_{name}.png"))
        print(f"wrote pa_{name}.dxf / .svg / .pdf / .png")

    print(f"\noutput -> {out}")


if __name__ == "__main__":
    main()
