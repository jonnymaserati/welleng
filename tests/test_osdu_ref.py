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
