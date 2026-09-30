"""A string that has been cut and pulled, or milled, is not there to host.

Every "which string is at this depth" test in the renderers asks
``Tubular.steel_at``, so a cut or milled length is not taken as the bore a
plug fills, the string a packer sets in, or the host a liner hangs from.

Known answer: a well whose inner string is cut above the plug must draw the
plug exactly as the same well with that string removed -- above the cut there
is no steel of it.
"""
import pytest

from welleng.schematic import (
    Casing,
    CementPlug,
    CompletionItem,
    HoleSection,
    MilledInterval,
    SurveyRef,
    Well,
    Wellbore,
    WellSchematic,
    build_column,
    build_section,
)
from welleng.schematic.column import L_COMPLETION, L_PLUG

SURVEY = SurveyRef(md=[0.0, 1500.0], inc=[0.0, 0.0], azi=[0.0, 0.0])
C133 = dict(name="13-3/8in", od_in=13.375, nominal_weight_ppf=68.0,
            top_md=0.0, shoe_md=1000.0, cut_md=300.0)
C958 = dict(name="9-5/8in", od_in=9.625, nominal_weight_ppf=47.0,
            top_md=0.0, shoe_md=1450.0)


def _well(c958=None, plugs=(), completion=()):
    casings = [Casing(**C133)] + ([Casing(**c958)] if c958 else [])
    b = Wellbore(
        id="w", survey=SURVEY,
        hole_sections=[HoleSection(bit_in=17.5, top_md=0.0, base_md=1000.0),
                       HoleSection(bit_in=12.25, top_md=1000.0, base_md=1500.0)],
        casings=casings, cement_plugs=list(plugs), completion=list(completion))
    return WellSchematic(well=Well(name="w"), wellbores=[b])


def _half_width(dwg, layer, lo, hi):
    xs = [abs(x) for e in dwg.entities if getattr(e, "layer", None) == layer
          for x, y in (getattr(e, "points", None) or getattr(e, "boundary", []))
          if lo < y < hi]
    assert xs, f"nothing on {layer} between {lo} and {hi}"
    return max(xs)


PLUG = CementPlug(name="plug", top_md=670.0, base_md=770.0)


@pytest.mark.parametrize("builder", [build_column, build_section],
                         ids=["column", "section"])
def test_a_plug_above_a_cut_fills_the_outer_string(builder):
    cut = _well(dict(C958, cut_md=810.0), plugs=[PLUG])
    absent = _well(None, plugs=[PLUG])
    present = _well(C958, plugs=[PLUG])
    w_cut = _half_width(builder(cut), L_PLUG, 669.0, 771.0)
    assert w_cut == pytest.approx(
        _half_width(builder(absent), L_PLUG, 669.0, 771.0), rel=1e-12)
    # known positive: with the 9-5/8in in place the plug is narrower
    assert _half_width(builder(present), L_PLUG, 669.0, 771.0) < w_cut * 0.8


def test_a_plug_in_a_milled_length_fills_the_outer_string():
    milled = dict(C958, milled=[MilledInterval(top_md=650.0, base_md=800.0,
                                               cement_removed=True)])
    w = _half_width(build_column(_well(milled, plugs=[PLUG])), L_PLUG, 669, 771)
    assert w == pytest.approx(
        _half_width(build_column(_well(None, plugs=[PLUG])), L_PLUG, 669, 771),
        rel=1e-12)


def test_a_packer_sets_in_the_string_that_is_still_there():
    tbg = CompletionItem(type="tubing", name="tbg", od_in=4.5, top_md=0.0,
                         base_md=1200.0)
    pkr = CompletionItem(type="packer", name="pkr", md=700.0)   # OD unrecorded
    cut = _well(dict(C958, cut_md=810.0), completion=[tbg, pkr])
    absent = _well(None, completion=[tbg, pkr])
    assert _half_width(build_column(cut), L_COMPLETION, 690.0, 710.0) == \
        pytest.approx(_half_width(build_column(absent), L_COMPLETION,
                                  690.0, 710.0), rel=1e-12)
