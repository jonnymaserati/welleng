"""OSDU reference-data resolution, and the governance-driven strictness."""
import json
from pathlib import Path

import pytest

from welleng import osdu_ref
from welleng.hierarchy import Datum
from welleng.osdu import to_osdu
from welleng.hierarchy import Well


# --- the vendored data ------------------------------------------------------ #
def test_every_vendored_list_carries_its_provenance():
    """A vocabulary that drifts silently is worse than none: each file has to
    say where it came from, when, and under whose authority."""
    assert osdu_ref.available()
    for name in osdu_ref.available():
        p = osdu_ref.provenance(name)
        assert p["source"].startswith("https://community.opengroup.org/")
        assert p["licence"] == "Apache-2.0"
        assert p["governance"] in (osdu_ref.FIXED, osdu_ref.OPEN, osdu_ref.LOCAL)
        assert len(p["retrieved"]) == 10          # ISO date
        assert osdu_ref.codes(name), f"{name} has no codes"


def test_ppdm_attribution_is_carried_not_dropped():
    """Several of these lists are curated by PPDM, not OSDU. Attribution is
    not ours to drop."""
    assert "PPDM Association" in \
        osdu_ref.provenance("VerticalMeasurementType")["attribution"]


def test_deprecated_codes_are_not_offered():
    """A DEPRECATED record must not be handed out as a current code."""
    for name in osdu_ref.available():
        raw = json.loads(
            (Path(osdu_ref.__file__).parent / "data" / "osdu"
             / f"{name}.json").read_text()
        )
        assert all(not v.upper().startswith("DEPRECATED")
                   for v in raw["codes"].values())


def test_an_unvendored_list_says_how_to_add_it():
    with pytest.raises(osdu_ref.OsduRefError, match="fetch_osdu_reference_data"):
        osdu_ref.codes("NotAList")


# --- resolution -------------------------------------------------------------- #
@pytest.mark.parametrize("value,expected", [
    ("RotaryTable", "RotaryTable"),          # exact code
    ("rotary table", "RotaryTable"),         # display name, folded
    ("ROTARY-TABLE", "RotaryTable"),         # separators folded
    ("RKB", "RotaryTable"),                  # field alias
    ("rt", "RotaryTable"),
    ("kb", "KellyBushing"),
    ("MSL", "MeanSeaLevel"),
    ("seabed", "Seafloor"),
    ("mudline", "MLS"),          # OSDU carries MLS *and* Seafloor
])
def test_datum_references_resolve(value, expected):
    assert osdu_ref.resolve("VerticalMeasurementType", value) == expected


def test_an_ambiguous_term_resolves_to_nothing_rather_than_a_guess():
    """OSDU distinguishes the casing-head, tubing-head and top/bottom flanges;
    a bare "wellhead" does not say which. An unresolved value is recoverable,
    a wrong one is not."""
    assert osdu_ref.resolve("VerticalMeasurementType", "wellhead") is None


@pytest.mark.parametrize("value,expected", [
    ("true", "TrueNorth"), ("grid", "GridNorth"), ("magnetic", "MagneticNorth"),
])
def test_welleng_azimuth_references_map(value, expected):
    assert osdu_ref.resolve("AzimuthReferenceType", value) == expected


def test_resolve_is_none_safe_and_empty_safe():
    assert osdu_ref.resolve("AzimuthReferenceType", None) is None
    assert osdu_ref.resolve("AzimuthReferenceType", "   ") is None


def test_catalogue_grades_are_already_osdu_codes():
    """Both follow API 5CT, so this is validation, not a rename."""
    from welleng.catalog import grades
    for g in grades():
        assert osdu_ref.resolve("TubularComponentGrade", g) == g


# --- governance IS the strictness ------------------------------------------- #
def test_open_list_warns_but_passes_an_extension_through():
    """An OPEN list may be extended by an operator, so an unknown code is
    legitimate -- but the caller has to know it is non-standard."""
    assert osdu_ref.governance("GeoMagneticModel") == osdu_ref.OPEN
    with pytest.warns(UserWarning, match="OPEN"):
        assert osdu_ref.validate("GeoMagneticModel", "WMM2025") == "WMM2025"


def test_local_list_passes_silently(recwarn):
    assert osdu_ref.governance("TubularComponentGrade") == osdu_ref.LOCAL
    assert osdu_ref.validate("TubularComponentGrade", "MillSpecial") \
        == "MillSpecial"
    assert not recwarn.list


def test_a_published_code_never_warns(recwarn):
    assert osdu_ref.validate("CementPlugType", "Abandonment") == "Abandonment"
    assert not recwarn.list


def test_none_is_always_allowed(recwarn):
    """Absence is not a validation question."""
    assert osdu_ref.validate("CementPlugType", None) is None
    assert not recwarn.list


def test_the_pa_plug_vocabulary_is_there():
    assert set(osdu_ref.codes("CementPlugType")) == {
        "Abandonment", "KickOff", "LostCirculation", "PlugBack", "Suspension"
    }


# --- ids ---------------------------------------------------------------------- #
def test_osdu_id_shape():
    assert osdu_ref.osdu_id("CementPlugType", "Abandonment") \
        == "reference-data--CementPlugType:Abandonment"
    assert osdu_ref.osdu_id("CementPlugType", "Abandonment", "ns") \
        == "ns:reference-data--CementPlugType:Abandonment"


# --- the datum, end to end ---------------------------------------------------- #
def test_datum_exposes_its_reference_as_a_code():
    assert Datum(name="D", reference="RKB").osdu_reference() == "RotaryTable"
    assert Datum(name="D").osdu_reference() == "MeanSeaLevel"     # "MSL" default


def test_datum_reference_string_is_never_rewritten():
    """Adoption is additive. A consumer reading `.reference` sees what it set."""
    d = Datum(name="D", reference="RKB")
    d.osdu_reference()
    assert d.reference == "RKB"


def test_exported_well_carries_the_reference_frame():
    """A vertical measurement without a frame says how far, not from what."""
    w = Well(id="w1", name="W", datum=Datum(name="D", elevation=25.0,
                                            reference="RKB"))
    rec = to_osdu(w)
    vm = rec["data"]["VerticalMeasurements"][0]
    assert vm["VerticalMeasurementTypeID"] == \
        "reference-data--VerticalMeasurementType:RotaryTable"


def test_an_unresolvable_frame_is_omitted_not_guessed():
    w = Well(id="w1", name="W", datum=Datum(name="D", elevation=25.0,
                                            reference="wellhead"))
    vm = to_osdu(w)["data"]["VerticalMeasurements"][0]
    assert "VerticalMeasurementTypeID" not in vm
    assert vm["VerticalMeasurement"] is not None      # the depth still exports


# --- how the match was reached ----------------------------------------------- #
@pytest.mark.parametrize("value,how", [
    ("RotaryTable", "exact"),
    ("Rotary Table", "exact"),   # folds to the code itself
    ("Mudline", "name"),         # display name of code MLS
    ("RKB", "alias"),
    ("no such thing", "none"),
])
def test_resolve_reports_how_it_matched(value, how):
    """An export layer must be able to refuse a weak match, and a bare
    code-or-None cannot express 'matched, but only on similarity'."""
    m = osdu_ref.resolve_match("VerticalMeasurementType", value)
    assert m.how == how
    assert bool(m) is (how != "none")


def test_resolve_and_resolve_match_agree():
    for v in ("RotaryTable", "Rotary Table", "RKB", "wellhead", None, ""):
        assert osdu_ref.resolve("VerticalMeasurementType", v) == \
            osdu_ref.resolve_match("VerticalMeasurementType", v).code


# --- log-curve mnemonics ------------------------------------------------------ #
def test_the_three_density_mnemonics_share_one_quantity():
    """The case that motivated curves_by_unit: a name list reported "no density
    curve" on 12 of 14 wells that had one."""
    for m in ("RHOB", "RHOZ", "BDCX"):
        assert osdu_ref.curve_quantity(m) == "mass per volume", m


def test_an_ambiguous_mnemonic_returns_none_rather_than_picking():
    """DT is compressional slowness to two vendors and a plain transit time to
    a third. Those are not the same curve."""
    assert osdu_ref.curve_quantity("DT") is None
    vendors = dict((v, q) for v, _p, q in osdu_ref.curve_vendors("DT"))
    assert len(set(vendors.values())) > 1


def test_naming_the_vendor_disambiguates():
    assert osdu_ref.curve_quantity("DT", vendor="Schlumberger") \
        == "time per length"


def test_property_is_stricter_than_quantity():
    """RHOB is 'density' to one vendor and 'bulk density' to another -- the
    same measurement under two names, so the quantity resolves and the
    property does not."""
    assert osdu_ref.curve_property("RHOB") is None
    assert osdu_ref.curve_quantity("RHOB") == "mass per volume"


def test_an_unknown_mnemonic_is_empty_not_an_error():
    assert osdu_ref.curve_property("NOTACURVE") is None
    assert osdu_ref.curve_vendors("NOTACURVE") == []


def test_las_groups_curves_by_curated_quantity():
    from welleng.exchange.las import open_las
    las = open_las(
        "~VERSION\nVERS. 2.0 :\nWRAP. NO :\n~WELL\nWELL. T :\n~CURVE\n"
        "DEPT.M    :\nRHOZ.G/C3 :\nGR  .GAPI :\n~ASCII\n0.0 2.4 12.0\n"
        "1.0 2.5 14.0\n"
    )
    assert las.curves_by_quantity("mass per volume") == ["RHOZ"]
    assert "GR" in las.curves_by_quantity("API gamma ray")


# --- survey header ------------------------------------------------------------ #
@pytest.mark.parametrize("local,code", [
    ("true", "TrueNorth"), ("grid", "GridNorth"), ("magnetic", "MagneticNorth"),
])
def test_survey_header_reports_its_azimuth_reference_as_a_code(local, code):
    from welleng.survey import SurveyHeader
    assert SurveyHeader(azi_reference=local).osdu_azi_reference() == code


def test_magnetic_model_label_is_stripped_of_its_source_tag():
    from welleng.survey import SurveyHeader
    h = SurveyHeader()
    h.mag_model = "WMM2020 (local-wmm)"
    assert h.osdu_magnetic_model() == "WMM2020"


def test_the_current_model_has_no_published_code_yet():
    """WMM2025 is not in the list — it stops at WMM2020. Recorded so the gap is
    visible rather than surprising; raised upstream separately."""
    from welleng.survey import SurveyHeader
    h = SurveyHeader()
    h.mag_model = "WMM2025 (local-wmm)"
    assert h.osdu_magnetic_model() is None
    assert "WMM2025" not in osdu_ref.codes("GeoMagneticModel")


def test_no_magnetic_model_is_none_not_an_error():
    from welleng.survey import SurveyHeader
    assert SurveyHeader().osdu_magnetic_model() is None


# --- version pinning ---------------------------------------------------------- #
def test_every_list_pins_the_upstream_commit_it_came_from():
    """A retrieval date says when we looked; only the commit says WHAT we
    looked at. Without it, drift is undetectable — and a code already stored in
    someone's data can be deprecated or renamed upstream with nothing in the
    stored record to show it."""
    for name in osdu_ref.available():
        sha = osdu_ref.pinned_commit(name)
        assert sha and len(sha) == 40, f"{name} has no pinned commit"
        assert osdu_ref.provenance(name)["committed"]


# --- error models ARE survey tool types --------------------------------------- #
def test_welleng_error_models_resolve_to_survey_tool_types():
    """Not an adoption: the OSDU codes are "<short name>_<OWSG model id>", the
    exact pair our model metadata already carries."""
    import glob
    import json as _json
    import os
    import welleng as we
    root = os.path.join(os.path.dirname(we.__file__), "errors", "iscwsa_json")
    models = [_json.load(open(f)).get("metadata", {})
              for f in glob.glob(root + "/**/*.json", recursive=True)]
    assert len(models) > 90
    hits = [m for m in models
            if osdu_ref.survey_tool_type(m.get("short_name"),
                                         m.get("model_id"))]
    assert len(hits) >= len(models) - 1, \
        f"only {len(hits)}/{len(models)} models resolve"


def test_the_six_axis_model_is_absent_upstream_and_says_so():
    """D007Ma is in no published code — verified, not a spelling difference.
    Locked so the gap stays visible."""
    assert osdu_ref.survey_tool_type("ISCWSA Rot 6Axis MWD+SRGM", "D007Ma") \
        is None
    assert not [c for c in osdu_ref.codes("SurveyToolType") if "D007Ma" in c]


def test_min_curve_and_balanced_tangential_are_both_published():
    """Our dp_basis distinction is OSDU's CalculationMethodType — and it is the
    distinction that has faked residuals in analytical-vs-MC comparison."""
    codes = osdu_ref.codes("CalculationMethodType")
    assert "MinimumCurvature" in codes and "BalancedTangential" in codes
    assert osdu_ref.resolve("CalculationMethodType", "min curve") is None
    assert osdu_ref.resolve("CalculationMethodType", "MinimumCurvature") \
        == "MinimumCurvature"
