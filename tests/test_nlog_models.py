"""Typed NLOG payload views: the caveats, and refusing to over-claim."""
import pytest

from welleng.exchange.nlog_models import (
    ASSET_TYPES,
    DOCUMENT_KINDS,
    BoreholeSummary,
    DocumentRecord,
    LogFileRecord,
    SuggestHit,
    classify_document,
)


# --- a reverse-engineered reader must never DROP a field -------------------- #
def test_an_unmodelled_field_survives():
    """NLOG can add a field tomorrow. Losing it silently is worse than not
    modelling it, and the loss would be invisible."""
    d = DocumentRecord.model_validate(
        {"assetBfileDbk": 1, "fullTitle": "x", "somethingNew": 42})
    assert d.unmodelled == {"somethingNew": 42}
    assert d.model_dump()["somethingNew"] == 42


def test_every_model_tolerates_an_empty_payload():
    """We do not have NLOG's schema, so nothing may be declared required."""
    for model in (DocumentRecord, LogFileRecord, SuggestHit, BoreholeSummary):
        assert model.model_validate({}) is not None


# --- document classification ------------------------------------------------ #
@pytest.mark.parametrize("title,kind", [
    ("End of Well Report(12 Sep 2006) # 1", "end_of_well_report"),
    ("P11-A-02A Final Well Report(30 Aug 2007)", "final_well_report"),
    ("P11-A-02/02A, DE RUYTER Geological Final Well Report",
     "geological_final_well_report"),
    ("MWD End of Well Report (P11-A-02)", "mwd_report"),
    ("Composite Log(47-2710)(23 Oct 2006)", "composite_log"),
    ("Drilling Parameters Log(157-2691)", "drilling_parameters"),
])
def test_real_titles_classify(title, kind):
    """Every one of these is a title NLOG actually serves for one well."""
    h = classify_document(title, "AERA")
    assert h.kind == kind and h.confident


def test_the_specific_pattern_wins():
    """'Geological Final Well Report' is not a final well report, and an 'MWD
    End of Well Report' is a contractor document, not the operator's."""
    assert classify_document("Geological Final Well Report").kind == \
        "geological_final_well_report"
    assert classify_document("MWD End of Well Report").kind == "mwd_report"


def test_an_unrecognised_title_is_unknown_and_says_so():
    """'Gamma Ray - Resistivity - Bulk Density' is a real ADBM title and names
    no document kind. Returning a guess would be worse than returning none."""
    h = classify_document("Gamma Ray - Resistivity - Bulk Density", "ADBM")
    assert h.kind == "unknown"
    assert h.confident is False
    assert "borehole measurement" in h.description


def test_the_asset_type_alone_is_never_confident():
    """⚠️ 'Final Well Report' is served under BOTH AERA and ADPB on the same
    well, so the code is a bucket and cannot identify a document."""
    for code in ASSET_TYPES:
        assert classify_document(None, code).confident is False
    assert classify_document(None, "NOT_A_CODE").kind == "unknown"


def test_a_hint_is_labelled_as_a_genre_not_a_claim():
    h = classify_document("End of Well Report")
    assert "casing and cement record" in h.likely_contains
    assert "NOT a claim about this file" in \
        type(h).model_fields["likely_contains"].description


def test_every_kind_has_a_description():
    for kind, (desc, contains) in DOCUMENT_KINDS.items():
        assert desc
        assert isinstance(contains, tuple)


# --- the caveats live on the fields ----------------------------------------- #
@pytest.mark.parametrize("model,field,must_say", [
    (LogFileRecord, "top_depth", "not in this payload"),
    (DocumentRecord, "asset_type", "BUCKET"),
    (SuggestHit, "title", "P11-A02A"),
    (SuggestHit, "xcoordinate", "NOT in this payload"),
    (BoreholeSummary, "confidentiality_date", "NOT evidence"),
])
def test_the_trap_is_documented_where_it_is_met(model, field, must_say):
    """A caveat in a module docstring is read if you go looking; one on the
    field is read at the point of use. A consumer told us plainly: 'I had not
    read _provenance in your own data file.'"""
    assert must_say in (model.model_fields[field].description or "")


# --- interval logic --------------------------------------------------------- #
def test_covers_uses_the_stated_interval():
    r = LogFileRecord.model_validate({"topDepth": 718.1, "bottomDepth": 897.4})
    assert r.covers(800.0)
    assert not r.covers(1200.0)
    assert r.covers(718.1) and r.covers(897.4)


def test_an_unstated_interval_does_not_cover_anything():
    """Absence is not coverage — and a file with no stated interval would
    otherwise match every depth asked for."""
    assert not LogFileRecord.model_validate({"topDepth": 100.0}).covers(150.0)
    assert not LogFileRecord.model_validate({}).covers(0.0)


def test_reversed_bounds_still_work():
    r = LogFileRecord.model_validate({"topDepth": 900.0, "bottomDepth": 700.0})
    assert r.covers(800.0)


def test_suggest_hit_exposes_the_int_id_every_other_call_wants():
    h = SuggestHit.model_validate({"objectId": "163212895", "title": "P11-A-02A"})
    assert h.borehole_id == 163212895
    assert SuggestHit.model_validate({}).borehole_id is None


def test_document_title_falls_back_to_the_barcode():
    assert DocumentRecord.model_validate({"barCodeTitle": "b"}).title == "b"
    assert DocumentRecord.model_validate(
        {"fullTitle": "f", "barCodeTitle": "b"}).title == "f"
    assert DocumentRecord.model_validate({}).title == ""
