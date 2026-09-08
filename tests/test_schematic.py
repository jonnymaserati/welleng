"""Smoke + behaviour tests for welleng.schematic."""
import json

import pytest

ezdxf = pytest.importorskip("ezdxf")
pytest.importorskip("pydantic")
pytest.importorskip("matplotlib")

from welleng.schematic import (  # noqa: E402
    DepthResolver,
    PressureProfile,
    SurveyRef,
    WellFigure,
    WellSchematic,
    build_column,
    build_section,
    to_dxf,
    to_matplotlib,
    to_pdf,
    to_png,
    to_svg,
)

DATA = {
    "well": {"name": "TEST-1"},
    "survey": {"md": [0, 400, 1000, 2600], "inc": [0, 0, 30, 30],
               "azi": [45, 45, 45, 45]},
    "hole_sections": [
        {"bit_in": 26, "top_md": 0, "base_md": 400, "radial_scale": 50},
        {"bit_in": 17.5, "top_md": 400, "base_md": 1200, "radial_scale": 40},
        {"bit_in": 8.5, "top_md": 1200, "base_md": 2600, "radial_scale": 30},
    ],
    "casings": [
        {
            "name": "20in",
            "od_in": 20,
            "id_in": 18.7,
            "top_md": 0,
            "shoe_md": 400,
            "toc_md": 0,
        },
        {
            "name": "13-3/8",
            "od_in": 13.375,
            "id_in": 12.4,
            "top_md": 0,
            "shoe_md": 1200,
            "toc_md": 800,
        },
        {
            "name": "9-5/8",
            "od_in": 9.625,
            "id_in": 8.68,
            "top_md": 0,
            "shoe_md": 2600,
            "toc_md": 1400,
        },
    ],
    "cement_plugs": [{"name": "Reservoir plug", "top_md": 2350, "base_md": 2600}],
    "completion": [
        {"type": "tubing", "od_in": 4.5, "top_md": 0, "base_md": 2300},
        {"type": "packer", "name": "Pkr", "od_in": 8.68, "md": 2300},
        {"type": "sssv", "name": "SCSSV", "od_in": 4.5, "md": 350},
    ],
    "formations": [
        {"name": "A", "top_md": 0, "color": "#c9e6a8"},
        {"name": "Seal", "top_md": 2150, "color": "#5e35b1", "seal": True},
        {"name": "Res", "top_md": 2300, "color": "#ffca28", "flow": True},
        {"name": "", "top_md": 2600, "color": "#ffffff"},
    ],
}


@pytest.fixture
def schematic():
    resolver = DepthResolver(SurveyRef(**DATA["survey"]))
    tvd = [float(t) for t in resolver.tvd_at([0, 1000, 2600])]
    data = dict(DATA)
    data["pressures"] = PressureProfile.from_emw(
        [0, 1000, 2600], tvd, [1.03, 1.10, 1.66], [1.30, 1.60, 1.90], unit="bar"
    ).model_dump()
    return WellSchematic.from_dict(data)


# --- models ---------------------------------------------------------------
def test_flat_wrap_to_single_bore(schematic):
    assert len(schematic.wellbores) == 1
    assert schematic.primary.id == "main"
    assert schematic.primary.parent_id is None
    assert len(schematic.primary.casings) == 3


def test_json_roundtrip(schematic):
    text = schematic.to_json()
    again = WellSchematic.from_json(text)
    assert again.well.name == "TEST-1"
    assert json.loads(text)["well"]["name"] == "TEST-1"


def test_multilateral_hook():
    """A well can be a tree of wellbores (lateral off a parent)."""
    tree = {
        "well": {"name": "ML"},
        "wellbores": [
            {"id": "main", "survey": DATA["survey"], "casings": DATA["casings"]},
            {"id": "lat1", "parent_id": "main", "kickoff_md": 1500.0,
             "survey": {"md": [1500, 2600], "inc": [30, 60], "azi": [45, 90]}},
        ],
    }
    sch = WellSchematic.from_dict(tree)
    assert len(sch.wellbores) == 2
    assert sch.wellbores[1].parent_id == "main"
    assert sch.wellbores[1].kickoff_md == 1500.0
    assert sch.primary.id == "main"


def test_radial_scale_monotonic_enforced():
    """A deeper section may not exaggerate more than a shallower one."""
    bad = dict(DATA)
    bad["hole_sections"] = [
        {"bit_in": 26, "top_md": 0, "base_md": 400, "radial_scale": 30},
        {"bit_in": 8.5, "top_md": 400, "base_md": 2600, "radial_scale": 50},
    ]
    with pytest.raises(Exception):
        WellSchematic.from_dict(bad)


def test_pressure_from_emw_increases_with_depth():
    pp = PressureProfile.from_emw(
        [0, 2000], [0, 1800], [1.0, 1.5], [1.4, 1.8], unit="bar"
    )
    assert pp.unit == "bar"
    assert pp.pore[1] > pp.pore[0]
    assert pp.frac[-1] > pp.pore[-1]
    # ~ 1.5 * 0.0981 * 1800 ~ 265 bar
    assert 200 < pp.pore[1] < 320


def test_survey_ref_validates_length():
    with pytest.raises(Exception):
        SurveyRef(md=[0, 100], inc=[0], azi=[0, 0])


# --- generators + backends ------------------------------------------------
def _layers(drawing):
    return set(drawing.layers)


def test_build_column(schematic):
    dwg = build_column(schematic, mode="MD")
    assert {"CASING", "CEMENT", "HOLE", "PLUG"} <= _layers(dwg)
    assert len(dwg.entities) > 20
    assert "casing_shoe" in dwg.symbols


@pytest.mark.parametrize("ext,fn", [
    ("dxf", to_dxf), ("svg", to_svg), ("pdf", to_pdf), ("png", to_png),
])
def test_column_backends_write(schematic, tmp_path, ext, fn):
    dwg = build_column(schematic, mode="MD")
    out = tmp_path / f"col.{ext}"
    fn(dwg, str(out))
    assert out.exists() and out.stat().st_size > 0


def test_column_matplotlib_returns_figure(schematic):
    fig = to_matplotlib(build_column(schematic, mode="MD"))
    assert fig is not None
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_dxf_reopens_with_layers(schematic, tmp_path):
    dwg = build_column(schematic, mode="MD")
    out = tmp_path / "col.dxf"
    to_dxf(dwg, str(out))
    doc = ezdxf.readfile(str(out))
    names = {ln.dxf.name for ln in doc.layers}
    assert {"CASING", "CEMENT", "HOLE"} <= names
    # symbols realised as blocks -> INSERT references present
    assert len(list(doc.modelspace().query("INSERT"))) > 0
    assert len(list(doc.modelspace().query("HATCH"))) > 0


def test_section_backends(schematic, tmp_path):
    dwg = build_section(schematic)
    assert "SHOE" in _layers(dwg)
    for ext, fn in (("dxf", to_dxf), ("svg", to_svg), ("png", to_png)):
        out = tmp_path / f"sec.{ext}"
        fn(dwg, str(out))
        assert out.exists() and out.stat().st_size > 0


def test_figure_backends(schematic, tmp_path):
    fig = WellFigure(schematic, mode="MD")
    for ext in ("dxf", "svg", "pdf", "png"):
        out = tmp_path / f"fig.{ext}"
        fig.render(ext, str(out))
        assert out.exists() and out.stat().st_size > 0
    assert fig.render("matplotlib") is not None


def _max_depth_y(drawing):
    """Deepest world-y coordinate among filled entities."""
    ys = []
    for e in drawing.entities:
        pts = getattr(e, "points", None) or getattr(e, "boundary", None)
        if pts:
            ys.extend(p[1] for p in pts)
    return max(ys)


def test_md_vs_tvd_deep_component_differs(schematic):
    col_md = build_column(schematic, mode="MD")
    col_tvd = build_column(schematic, mode="TVD")
    y_md = _max_depth_y(col_md)
    y_tvd = _max_depth_y(col_tvd)
    # inclined well -> TVD of the deepest component is well above its MD
    assert abs(y_md - y_tvd) > 50.0
