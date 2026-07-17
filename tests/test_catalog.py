"""Tests for welleng.catalog (API 5CT casing + tubing catalogue + resolver)."""
import pytest

from welleng.catalog import (
    CatalogError,
    ConnectionSpec,
    list_sizes,
    resolve,
    resolve_coupling,
)

# (od_in, weight_ppf, expected id_in, expected drift_in) — VERIFIED anchor rows.
CASING_ANCHORS = [
    (4.5, 11.6, 4.000, 3.875),
    (5.5, 17.0, 4.892, 4.767),
    (7.0, 29.0, 6.184, 6.059),
    (7.0, 32.0, 6.094, 5.969),
    (9.625, 47.0, 8.681, 8.525),
    (9.625, 53.5, 8.535, 8.379),
    (13.375, 68.0, 12.415, 12.259),
    (13.375, 72.0, 12.347, 12.191),
    (20.0, 94.0, 19.124, 18.936),
]
TUBING_ANCHORS = [
    (2.875, 6.5, 2.441, 2.347),
    (3.5, 9.3, 2.992, 2.867),
]


@pytest.mark.parametrize("od, wt, id_in, drift_in", CASING_ANCHORS)
def test_casing_anchor_rows(od, wt, id_in, drift_in):
    spec = resolve(od, wt, kind="casing")
    assert spec.id_in == id_in
    assert spec.drift_in == drift_in
    assert spec.od_in == od
    assert spec.nominal_weight_ppf == wt


@pytest.mark.parametrize("od, wt, id_in, drift_in", TUBING_ANCHORS)
def test_tubing_anchor_rows(od, wt, id_in, drift_in):
    spec = resolve(od, wt, kind="tubing")
    assert spec.id_in == id_in
    assert spec.drift_in == drift_in


def test_grade_sets_yield():
    spec = resolve(9.625, 47, grade="L80", kind="casing")
    assert spec.yield_psi == 80000
    for grade, y in [("J55", 55000), ("K55", 55000), ("N80", 80000),
                     ("P110", 110000)]:
        assert resolve(7.0, 29.0, grade=grade).yield_psi == y


def test_no_grade_no_yield():
    spec = resolve(7.0, 29.0, kind="casing")
    assert spec.yield_psi is None
    assert spec.yield_pa is None


def test_si_conversion():
    spec = resolve(9.625, 47, grade="L80", kind="casing")
    assert spec.id_m == pytest.approx(8.681 * 0.0254)
    assert spec.od_m == pytest.approx(9.625 * 0.0254)
    assert spec.drift_m == pytest.approx(8.525 * 0.0254)
    assert spec.wall_m == pytest.approx(0.472 * 0.0254)
    assert spec.yield_pa == pytest.approx(80000 * 6894.757)


def test_unknown_weight_raises_with_suggestion():
    with pytest.raises(CatalogError) as exc:
        resolve(9.625, 48.0, kind="casing")  # 47 and 53.5 exist, not 48
    msg = str(exc.value)
    assert "47" in msg and "nearest" in msg


def test_unknown_od_raises_with_available():
    with pytest.raises(CatalogError) as exc:
        resolve(99.0, 47.0, kind="casing")
    assert "available ODs" in str(exc.value)


def test_unknown_grade_raises():
    with pytest.raises(CatalogError) as exc:
        resolve(7.0, 29.0, grade="X999")
    assert "L80" in str(exc.value)


def test_list_sizes():
    sizes = list_sizes("casing")
    assert (9.625, 47.0) in sizes
    assert sizes == sorted(sizes)
    assert len(sizes) == 71
    assert len(list_sizes("tubing")) == 13


# --- couplings (API Spec 5CT Tables E.27-E.30) --------------------------------
# (od, connection, kind, coupling_od_in W, coupling_length_in NL) VERIFIED
# against the rendered 5CT tables.
COUPLING_ANCHORS = [
    # STC (Table E.27, Short NL)
    (7.0, "STC", "casing", 7.875, 7.25),
    (9.625, "STC", "casing", 10.625, 7.75),
    (13.375, "STC", "casing", 14.375, 8.0),
    # LTC (Table E.27, Long NL)
    (7.0, "LTC", "casing", 7.875, 9.0),
    (9.625, "LTC", "casing", 10.625, 10.5),
    # BTC (Table E.28)
    (7.0, "BTC", "casing", 7.875, 10.0),
    (9.625, "BTC", "casing", 10.625, 10.625),
    (13.375, "BTC", "casing", 14.375, 10.625),
    # tubing NUE (Table E.29) / EUE (Table E.30)
    (2.875, "NUE", "tubing", 3.500, 5.125),
    (2.875, "EUE", "tubing", 3.668, 5.25),
]


@pytest.mark.parametrize("od, conn, kind, w, nl", COUPLING_ANCHORS)
def test_resolve_coupling_anchor(od, conn, kind, w, nl):
    cpl = resolve_coupling(od, conn, kind=kind)
    assert cpl.coupling_od_in == w  # W (regular OD)
    assert cpl.coupling_length_in == nl  # NL (min length)
    assert cpl.connection == conn
    assert cpl.od_in == od


def test_coupling_special_clearance():
    # BTC 7" special-clearance Wc = 7.375 (Table E.28); STC has no Wc.
    assert resolve_coupling(7.0, "BTC").special_clearance_od_in == 7.375
    assert resolve_coupling(7.0, "STC").special_clearance_od_in is None
    # EUE 2-7/8" Wc = 3.460 (Table E.30); NUE has no Wc.
    assert resolve_coupling(2.875, "EUE", kind="tubing").special_clearance_od_in \
        == 3.460
    assert resolve_coupling(2.875, "NUE", kind="tubing").special_clearance_od_in \
        is None


def test_coupling_si_conversion():
    cpl = resolve_coupling(9.625, "BTC")
    assert cpl.coupling_od_m == pytest.approx(10.625 * 0.0254)
    assert cpl.coupling_length_m == pytest.approx(10.625 * 0.0254)


def test_resolve_coupling_unknown_connection_lists_available():
    with pytest.raises(CatalogError) as exc:
        resolve_coupling(7.0, "VAM-TOP")
    msg = str(exc.value)
    assert "available connections" in msg
    for conn in ("STC", "LTC", "BTC", "NUE", "EUE"):
        assert conn in msg


def test_resolve_coupling_wrong_kind_raises():
    with pytest.raises(CatalogError) as exc:
        resolve_coupling(2.875, "NUE", kind="casing")  # NUE is tubing
    assert "tubing" in str(exc.value)


def test_resolve_coupling_untabulated_size_raises():
    with pytest.raises(CatalogError) as exc:
        resolve_coupling(13.375, "LTC")  # no Long NL for 13-3/8 in E.27
    assert "available ODs" in str(exc.value)


# --- ConnectionSpec: API fills dimensional, premium fields stay None ----------
def test_connectionspec_from_api_fills_dimensional_only():
    cs = ConnectionSpec.from_api(9.625, "BTC", grade="L80")
    # dimensional (from 5CT)
    assert cs.connection_od_in == 10.625
    assert cs.coupling_length_in == 10.625
    assert cs.connection_type == "BTC"
    assert cs.grade == "L80"


PREMIUM_FIELDS = [
    "tension_eff_pct", "compression_eff_pct", "internal_pressure_eff_pct",
    "external_pressure_eff_pct", "tension_strength_klb",
    "compression_strength_klb", "internal_pressure_resistance_psi",
    "external_pressure_resistance_psi", "max_bending_deg_per_100ft",
    "max_load_coupling_face_klb", "makeup_torque_min_ftlb",
    "makeup_torque_opt_ftlb", "makeup_torque_max_ftlb",
    "shouldering_torque_min_ftlb", "shouldering_torque_max_ftlb",
    "delta_turn_min", "delta_turn_max", "connection_id_in", "makeup_loss_in",
]


@pytest.mark.parametrize("field", PREMIUM_FIELDS)
def test_connectionspec_premium_fields_default_none(field):
    cs = ConnectionSpec.from_api(7.0, "LTC")
    assert getattr(cs, field) is None


# --- body performance (API TR 5C3) --------------------------------------------
def test_performance_matches_published_api_values():
    # 9-5/8" 47 lb/ft L80: published API pipe-body yield 1086 klb,
    # internal yield 6870 psi, collapse 4760 psi.
    spec = resolve(9.625, 47.0, grade="L80", kind="casing")
    assert spec.pipe_body_yield_klb == 1086
    assert spec.internal_yield_pressure_psi == 6870
    assert spec.collapse_pressure_psi == 4760
    assert spec.max_yield_psi == 95000
    assert spec.min_uts_psi == 95000
    assert spec.min_wall_pct == 87.5
    # plain-end mass is geometry-only (present without a grade)
    assert spec.plain_end_weight_ppf == pytest.approx(46.18, abs=0.05)


def test_performance_none_without_grade():
    spec = resolve(9.625, 47.0, kind="casing")
    assert spec.pipe_body_yield_klb is None
    assert spec.internal_yield_pressure_psi is None
    assert spec.collapse_pressure_psi is None
    assert spec.max_yield_psi is None
    # geometry-only value still populated
    assert spec.plain_end_weight_ppf > 0


# --- cross-validation of the body catalogue vs API 5CT Table E.1 --------------
# (od, weight, published wall_in from Table E.1) - the ID must be OD - 2*wall.
TABLE_E1_ROWS = [
    (7.0, 29.0, 0.408),
    (9.625, 47.0, 0.472),
    (13.375, 68.0, 0.480),
]


@pytest.mark.parametrize("od, wt, wall", TABLE_E1_ROWS)
def test_casing_cross_validate_table_e1(od, wt, wall):
    spec = resolve(od, wt, kind="casing")
    assert spec.wall_in == wall
    assert spec.id_in == pytest.approx(od - 2 * wall, abs=1e-6)


# --- schematic model integration (needs pydantic + the schematic module) ---
pytest.importorskip("pydantic")
# schematic lives on its own branch; skip ONLY these integration tests until it
# merges (the catalog module itself does NOT depend on schematic -- only these
# tests do). NB: welleng.schematic imports as an (empty) namespace package here,
# so guard on the Casing symbol, not on module importability. Per-test skipif
# (not a module-level skip) so the pure-catalogue tests above still run.
try:
    from welleng.schematic import Casing as _Casing  # noqa: F401
    _HAS_SCHEMATIC = True
except ImportError:
    _HAS_SCHEMATIC = False

needs_schematic = pytest.mark.skipif(
    not _HAS_SCHEMATIC,
    reason="welleng.schematic.Casing not on this branch (schematic module unmerged)",
)


@needs_schematic
def test_casing_autofills_id_and_drift():
    from welleng.schematic import Casing

    c = Casing(name="9-5/8", od_in=9.625, nominal_weight_ppf=47, shoe_md=2600)
    assert c.id_in == 8.681
    assert c.drift_in == 8.525


@needs_schematic
def test_explicit_id_overrides_catalogue():
    from welleng.schematic import Casing

    c = Casing(name="9-5/8", od_in=9.625, nominal_weight_ppf=47,
               id_in=8.5, shoe_md=2600)
    assert c.id_in == 8.5  # explicit wins, no catalogue overwrite


@needs_schematic
def test_casing_grade_carried():
    from welleng.schematic import Casing

    c = Casing(name="9-5/8", od_in=9.625, nominal_weight_ppf=47,
               grade="L80", shoe_md=2600)
    assert c.id_in == 8.681
    assert c.grade == "L80"


@needs_schematic
def test_casing_bad_weight_raises_in_model():
    from welleng.schematic import Casing

    with pytest.raises(Exception) as exc:
        Casing(name="bad", od_in=9.625, nominal_weight_ppf=48, shoe_md=2600)
    assert "nearest" in str(exc.value)


@needs_schematic
def test_casing_id_only_still_works():
    """Back-compat: existing callers pass id_in with no weight."""
    from welleng.schematic import Casing

    c = Casing(name="20in", od_in=20, id_in=18.7, shoe_md=400)
    assert c.id_in == 18.7
    assert c.drift_in is None


@needs_schematic
def test_casing_connection_fills_coupling_dims():
    from welleng.schematic import Casing

    c = Casing(name="9-5/8", od_in=9.625, nominal_weight_ppf=47,
               connection="BTC", shoe_md=2600)
    assert c.coupling_od_in == 10.625  # W (Table E.28)
    assert c.coupling_length_in == 10.625  # NL


@needs_schematic
def test_casing_unknown_connection_label_is_noncrashing():
    """A premium/non-API connection label must not break the model."""
    from welleng.schematic import Casing

    c = Casing(name="9-5/8", od_in=9.625, nominal_weight_ppf=47,
               connection="VAM-TOP", shoe_md=2600)
    assert c.id_in == 8.681  # dimensions still resolved
    assert c.coupling_od_in is None  # non-API label -> no coupling fill
