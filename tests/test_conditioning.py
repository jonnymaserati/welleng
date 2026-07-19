"""welleng.conditioning — shared-error conditioning of two wells' combined
covariance (co-located wells share global / systematic error realisations, which
cancel in the position DIFFERENCE; the naive Sigma_A + Sigma_B over-states it).

Pinned here: the exact cancellation algebra per share-mode, the input contracts,
the PSD defence, the public import surface, and an end-to-end run on two real
error-modelled Surveys.
"""
import numpy as np
import pytest

from welleng.conditioning import (
    CombinedCovariance,
    ShareMode,          # noqa: F401  (the import IS the contract)
    combine_covariances,
)
from welleng.survey import Survey, SurveyHeader

MAG_REF = dict(b_total=50_000., dip=72., declination=-2.)


def _spd(rng, n, scale=1.0):
    """Random (n, 3, 3) SPD stack."""
    A = rng.normal(size=(n, 3, 3))
    return scale * np.einsum("nij,nkj->nik", A, A) + 1e-6 * np.eye(3)


def test_all_independent_is_naive_sum():
    rng = np.random.default_rng(0)
    a, b = _spd(rng, 5), _spd(rng, 5)
    r = combine_covariances(a, b, share_mode="all_independent")
    assert np.allclose(r.cov_combined, a + b)
    assert np.allclose(r.cov_naive, a + b)
    assert np.allclose(r.reduction_factor, 1.0)


def test_globals_shared_cancels_identical_global():
    # A = R_a + G, B = R_b + G with the SAME global realisation G:
    # cov(X_A - X_B) must reduce to R_a + R_b exactly.
    rng = np.random.default_rng(1)
    Ra, Rb, G = _spd(rng, 6), _spd(rng, 6), _spd(rng, 6)
    r = combine_covariances(Ra + G, Rb + G,
                            cov_global_a=G, cov_global_b=G,
                            share_mode="globals_shared")
    assert np.allclose(r.cov_combined, Ra + Rb, atol=1e-10)
    assert np.all(r.reduction_factor >= 1.0 - 1e-12)


def test_globals_and_systematic_shared_cancels_both():
    rng = np.random.default_rng(2)
    Ra, Rb, G, S = _spd(rng, 4), _spd(rng, 4), _spd(rng, 4), _spd(rng, 4)
    r = combine_covariances(
        Ra + G + S, Rb + G + S,
        cov_global_a=G, cov_global_b=G,
        cov_systematic_a=S, cov_systematic_b=S,
        share_mode="globals_and_systematic_shared")
    assert np.allclose(r.cov_combined, Ra + Rb, atol=1e-10)


def test_slightly_differing_globals_use_the_mean():
    # gA != gB (small along-hole variation): the shared estimate is the
    # arithmetic mean, so cov_combined = A + B - (gA + gB).
    rng = np.random.default_rng(3)
    Ra, Rb = _spd(rng, 3), _spd(rng, 3)
    gA = _spd(rng, 3, scale=0.5)
    gB = gA * 1.05                       # 5% different realisation-magnitude
    r = combine_covariances(Ra + gA, Rb + gB,
                            cov_global_a=gA, cov_global_b=gB,
                            share_mode="globals_shared")
    assert np.allclose(r.cov_combined, Ra + Rb, atol=1e-10)


def test_input_contracts():
    rng = np.random.default_rng(4)
    a, b = _spd(rng, 3), _spd(rng, 4)
    with pytest.raises(ValueError, match="same shape"):
        combine_covariances(a, b)
    a, b = _spd(rng, 3), _spd(rng, 3)
    with pytest.raises(ValueError, match="globals_shared requires"):
        combine_covariances(a, b, share_mode="globals_shared")
    with pytest.raises(ValueError, match="globals_and_systematic"):
        combine_covariances(a, b, cov_global_a=a, cov_global_b=b,
                            share_mode="globals_and_systematic_shared")
    with pytest.raises(ValueError, match="unknown share_mode"):
        combine_covariances(a, b, share_mode="nonsense")


def test_inconsistent_shares_clip_to_psd_with_warning():
    # "Global" bigger than the total it is claimed to be part of -> the
    # difference covariance would go negative; the module must warn + clip PSD.
    rng = np.random.default_rng(5)
    total = _spd(rng, 3, scale=0.1)
    g = _spd(rng, 3, scale=5.0)          # >> total: inconsistent share claim
    with pytest.warns(RuntimeWarning, match="negative eigenvalues"):
        r = combine_covariances(total, total,
                                cov_global_a=g, cov_global_b=g,
                                share_mode="globals_shared")
    assert np.all(np.linalg.eigvalsh(r.cov_combined) >= -1e-12)   # PSD


def test_result_type_and_fields():
    rng = np.random.default_rng(6)
    a, b = _spd(rng, 2), _spd(rng, 2)
    r = combine_covariances(a, b, share_mode="all_independent")
    assert isinstance(r, CombinedCovariance)
    for f in ("cov_combined", "cov_naive", "sigma_naive",
              "sigma_combined", "reduction_factor"):
        assert getattr(r, f) is not None


def test_end_to_end_two_error_modelled_surveys():
    # Two co-platform wells with the ISCWSA model: Survey exposes cov_nev /
    # cov_nev_global / cov_nev_systematic -- the documented feed-in. Sharing
    # globals must tighten (or at worst not widen) the combined uncertainty.
    def make(shift):
        md = np.linspace(0, 2400, 40)
        inc = np.clip(np.linspace(0, 80, 40), 0, 60)
        azi = (np.linspace(0, 90, 40) + shift) % 360
        return Survey(md=md, inc=inc, azi=azi, header=SurveyHeader(**MAG_REF),
                      error_model="ISCWSA MWD Rev5.11")
    a, b = make(0.0), make(12.0)
    assert a.cov_nev_global is not None          # the 0.19 Survey contract
    r = combine_covariances(
        a.cov_nev, b.cov_nev,
        cov_global_a=a.cov_nev_global, cov_global_b=b.cov_nev_global,
        share_mode="globals_shared")
    # station 0 has zero covariance; ignore nan reduction there
    red = r.reduction_factor[np.isfinite(r.reduction_factor)]
    assert np.all(red >= 1.0 - 1e-9)             # sharing never widens
    assert red.max() > 1.01                      # and genuinely tightens somewhere
    assert np.all(np.linalg.eigvalsh(r.cov_combined) >= -1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
