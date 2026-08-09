"""API TR 5C3 (7th ed., 2018) connection-performance CI tests.

Locks the buttress (BTC) + round-thread (STC/LTC) tensile joint strength
(Sec 9) and the coupling internal yield pressure (Sec 10.2) against:

1. a PUBLISHED API value (the universally-tabulated 7 in. 26 lb/ft P110 BTC
   joint strength = 853,000 lbf) - the independent oracle anchor;
2. formula regression locks across grades (J55..P110) and sizes (5-1/2, 7,
   9-5/8, 13-3/8) - "no formula drift";
3. structural invariants of the standard (min-of-two, coupling non-governing
   for standard couplings per Annex J, grade monotonicity, the Eq. 54
   D**-0.59 superscript that pdftotext drops).

Oracle note: API Spec 5CT (11th ed.) does NOT tabulate joint strength (it is a
product spec); API TR 5C3 gives the equations + Annex J discussion but no
result tables. The classic multi-row tabulation lived in the withdrawn API
Bull 5C2, which is not in the local reference library. The one value asserted
as PUBLISHED below (853 klb) is the ubiquitously-cited canonical figure and is
reproduced by Eq. (59) to <0.1 %; the remaining rows are regression-locked.
"""
import math

import pytest

import welleng.catalog as cat

TOL_PUBLISHED = 0.005  # 0.5 % vs published tabulation convention (nearest klb)


# --- 1. PUBLISHED anchor --------------------------------------------------

def test_btc_published_anchor_7in_26ppf_p110():
    """7 in. 26 lb/ft P110 BTC joint strength = 853,000 lbf (canonical API)."""
    s = cat.resolve(7.0, 26, grade="P110")
    pj = cat.buttress_joint_strength_klb(
        s.od_in, s.id_in, s.yield_psi, s.min_uts_psi
    )
    assert pj == pytest.approx(853, rel=TOL_PUBLISHED)


# --- 2. regression locks (no formula drift) -------------------------------

# (od_in, weight_ppf, grade): buttress joint strength (klb), from Eq. 59/60.
_BTC_REGRESSION = {
    (7.0, 26, "P110"): 853,
    (7.0, 29, "P110"): 955,
    (7.0, 26, "N80"): 667,
    (7.0, 23, "L80"): 565,
    (5.5, 17, "P110"): 568,
    (5.5, 17, "L80"): 428,
    (9.625, 47, "P110"): 1500,
    (9.625, 47, "N80"): 1161,
    (9.625, 53.5, "P110"): 1718,
    (9.625, 40, "L80"): 947,
    (13.375, 68, "N80"): 1585,
    (13.375, 72, "P110"): 2221,
    (13.375, 72, "N80"): 1693,
    (9.625, 43.5, "L80"): 1038,
}


@pytest.mark.parametrize("key,expected", list(_BTC_REGRESSION.items()))
def test_btc_joint_strength_regression(key, expected):
    od, wt, g = key
    s = cat.resolve(od, wt, grade=g)
    pj = cat.buttress_joint_strength_klb(
        s.od_in, s.id_in, s.yield_psi, s.min_uts_psi
    )
    assert pj == expected


# --- 3. structural invariants of the standard -----------------------------

@pytest.mark.parametrize("key", list(_BTC_REGRESSION))
def test_btc_coupling_non_governing_for_standard_couplings(key):
    """Annex J.2.2.3: for standard API couplings the coupling term does not
    govern - the pipe-thread term (Eq. 59) is the lesser, so joint == pipe."""
    od, wt, g = key
    s = cat.resolve(od, wt, grade=g)
    pipe = cat.buttress_pipe_thread_strength_klb(
        s.od_in, s.id_in, s.yield_psi, s.min_uts_psi
    )
    coupling = cat.buttress_coupling_thread_strength_klb(
        od, cat.resolve_coupling(od, "BTC").coupling_od_in, s.min_uts_psi
    )
    joint = cat.buttress_joint_strength_klb(
        s.od_in, s.id_in, s.yield_psi, s.min_uts_psi
    )
    assert coupling > pipe
    assert joint == pipe
    assert joint == min(pipe, coupling)


def test_btc_joint_is_least_when_coupling_governs():
    """A deliberately thin/weak coupling must make Eq. 60 govern the min."""
    s = cat.resolve(7.0, 26, grade="P110")
    weak = cat.buttress_joint_strength_klb(
        s.od_in,
        s.id_in,
        s.yield_psi,
        s.min_uts_psi,
        coupling_od_in=7.30,  # thinner than the 7.875 in. standard coupling
    )
    pipe = cat.buttress_pipe_thread_strength_klb(
        s.od_in, s.id_in, s.yield_psi, s.min_uts_psi
    )
    assert weak < pipe


def test_btc_monotonic_in_grade():
    """Higher grade -> higher joint strength for a fixed size/weight."""
    vals = []
    for g in ("N80", "P110"):
        s = cat.resolve(9.625, 47, grade=g)
        vals.append(
            cat.buttress_joint_strength_klb(
                s.od_in, s.id_in, s.yield_psi, s.min_uts_psi
            )
        )
    assert vals[0] < vals[1]


def test_btc_d1_matches_5b_pitch_diameter_rule():
    """d1 uses E7 = D4 - 0.062 with D4 = D + 0.016 (<=13-3/8 in.)."""
    d1 = cat.catalog._buttress_d1_in(7.0)
    e7 = (7.0 + 0.016) - 0.062  # API 5B buttress pitch diameter
    # d1 = E7 - (L7 + IB) Td + hB, L7=2.216, IB=0.5, Td=0.0625, hB=0.062
    expected = e7 - (2.2160 + 0.500) * 0.0625 + 0.062
    assert d1 == pytest.approx(expected, abs=1e-9)


def test_btc_unknown_size_raises():
    with pytest.raises(cat.CatalogError):
        cat.catalog._buttress_d1_in(3.5)  # not an API buttress casing size


# --- coupling internal yield (Sec 10.2, Eq. 65) ---------------------------

_COUPLING_YIELD_REGRESSION = {
    (7.0, 26, "P110"): 14370.0,
    (9.625, 47, "N80"): 8830.0,
    (13.375, 68, "N80"): 6530.0,
}


@pytest.mark.parametrize(
    "key,expected", list(_COUPLING_YIELD_REGRESSION.items())
)
def test_coupling_internal_yield_regression(key, expected):
    od, wt, g = key
    s = cat.resolve(od, wt, grade=g)
    w = cat.resolve_coupling(od, "BTC").coupling_od_in
    piyc = cat.buttress_coupling_internal_yield_psi(od, w, s.yield_psi)
    assert piyc == expected


@pytest.mark.parametrize("key", list(_COUPLING_YIELD_REGRESSION))
def test_coupling_yield_exceeds_pipe_internal_yield(key):
    """Sec 10.1: coupling yield limits only when below pipe internal yield;
    for standard buttress couplings it stays above (does not govern)."""
    od, wt, g = key
    s = cat.resolve(od, wt, grade=g)
    w = cat.resolve_coupling(od, "BTC").coupling_od_in
    piyc = cat.buttress_coupling_internal_yield_psi(od, w, s.yield_psi)
    assert piyc > s.internal_yield_pressure_psi


def test_coupling_internal_yield_matches_eq65_hand_calc():
    """piYc = fymnc (W - d1)/W, evaluated directly."""
    od, w, fy = 7.0, 7.875, 110000.0
    d1 = cat.catalog._buttress_d1_in(od)
    expected = round(fy * (w - d1) / w / 10.0) * 10.0
    assert cat.buttress_coupling_internal_yield_psi(od, w, fy) == expected


# --- round thread (STC/LTC), Sec 9.2.2 ------------------------------------

def test_round_thread_fracture_geometry_eq53_56():
    """Eq. 53/56: Pj = 0.95 (pi/4)[(D-0.1425)^2 - d^2] fumnp, exact."""
    s = cat.resolve(4.5, 11.6, grade="L80")
    ajp = (math.pi / 4.0) * ((s.od_in - 0.1425) ** 2 - s.id_in ** 2)
    expected = round(0.95 * ajp * s.min_uts_psi / 1000.0)
    got = cat.round_thread_pipe_fracture_strength_klb(
        s.od_in, s.id_in, s.min_uts_psi
    )
    assert got == expected


def test_round_thread_pullout_uses_D_power_minus_0p59_not_subtraction():
    """Eq. 54 carries 0.74 * D**-0.59 * fumnp (a SUPERSCRIPT that pdftotext
    drops). A subtraction misread (0.74*D - 0.59*fumnp) would be negative and
    give a nonsensical pull-out - guard against reintroducing it."""
    s = cat.resolve(4.5, 11.6, grade="L80")
    lt = 3.000 - 0.704  # Let = L4 - M, API 5B Table 4, 4-1/2 in. long thread
    po = cat.round_thread_pullout_strength_klb(
        s.od_in, s.id_in, lt, s.yield_psi, s.min_uts_psi
    )
    # correct term is strongly positive; the misread term is negative
    correct = 0.74 * s.od_in ** -0.59 * s.min_uts_psi
    misread = 0.74 * s.od_in - 0.59 * s.min_uts_psi
    assert correct > 0 > misread
    assert po == 223  # regression lock (Let = 2.296 in.)
    # sane joint efficiency band for 4-1/2 in. LTC
    eff = po / s.pipe_body_yield_klb
    assert 0.7 < eff < 1.1


def test_round_thread_joint_is_least_and_let_never_increases_it():
    """Joint strength is the LEAST term; supplying Let can only lower it
    (safety: omitting Let - fracture only - must not underrate vs the min)."""
    s = cat.resolve(4.5, 11.6, grade="L80")
    lt = 3.000 - 0.704
    frac_only = cat.round_thread_joint_strength_klb(
        s.od_in, s.id_in, s.yield_psi, s.min_uts_psi
    )
    with_let = cat.round_thread_joint_strength_klb(
        s.od_in, s.id_in, s.yield_psi, s.min_uts_psi,
        engaged_thread_length_in=lt,
    )
    assert with_let <= frac_only
    assert frac_only == cat.round_thread_pipe_fracture_strength_klb(
        s.od_in, s.id_in, s.min_uts_psi
    )


# --- ConnectionSpec integration -------------------------------------------

def test_connectionspec_from_tubular_fills_btc_tension():
    s = cat.resolve(7.0, 26, grade="P110")
    spec = cat.ConnectionSpec.from_tubular(s, "BTC")
    assert spec.connection_type == "BTC"
    assert spec.tension_strength_klb == pytest.approx(853, rel=TOL_PUBLISHED)
    assert spec.connection_od_in == 7.875


def test_connectionspec_from_tubular_leaves_round_tension_none():
    """STC/LTC tension needs the API 5B engaged length - not auto-filled."""
    s = cat.resolve(7.0, 26, grade="P110")
    spec = cat.ConnectionSpec.from_tubular(s, "LTC")
    assert spec.connection_type == "LTC"
    assert spec.tension_strength_klb is None
