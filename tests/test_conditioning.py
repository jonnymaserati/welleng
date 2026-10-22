"""welleng.conditioning — shared-error conditioning of two wells' combined
covariance. A shared source is one realisation acting through each well's own
signed vector, so it contributes (v_A - v_B)(v_A - v_B)' to the difference: it
cancels where the wells weight it alike and widens where they weight it
oppositely.

Pinned here: the algebra per share-mode, the input contracts and refusals, the
public import surface, the signed vector against a finite-difference oracle, and
an end-to-end run on two real error-modelled Surveys.
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


def _vec(rng, n, k, scale=1.0):
    """Random signed per-source vectors, (n, k, 3)."""
    return scale * rng.normal(size=(n, k, 3))


def _cov(v):
    return np.einsum("nki,nkj->nij", v, v)


def test_shared_sources_contribute_the_square_of_their_difference():
    # A = R_a + sum v_a v_a', B = R_b + sum v_b v_b', one realisation per shared
    # source: cov(X_A - X_B) = R_a + R_b + sum (v_a - v_b)(v_a - v_b)'.
    rng = np.random.default_rng(1)
    Ra, Rb = _spd(rng, 6), _spd(rng, 6)
    ga, gb = _vec(rng, 6, 4), _vec(rng, 6, 4)
    r = combine_covariances(Ra + _cov(ga), Rb + _cov(gb),
                            vec_global_a=ga, vec_global_b=gb,
                            share_mode="globals_shared")
    assert np.allclose(r.cov_combined, Ra + Rb + _cov(ga - gb), atol=1e-10)


def test_identical_weights_cancel_and_opposite_weights_double():
    rng = np.random.default_rng(2)
    Ra, Rb, g = _spd(rng, 4), _spd(rng, 4), _vec(rng, 4, 3)
    same = combine_covariances(Ra + _cov(g), Rb + _cov(g), vec_global_a=g,
                               vec_global_b=g, share_mode="globals_shared")
    assert np.allclose(same.cov_combined, Ra + Rb, atol=1e-10)
    opposite = combine_covariances(Ra + _cov(g), Rb + _cov(g), vec_global_a=g,
                                   vec_global_b=-g, share_mode="globals_shared")
    assert np.allclose(opposite.cov_combined, Ra + Rb + 4.0 * _cov(g), atol=1e-10)
    assert np.all(opposite.reduction_factor < 1.0)      # sharing widened it


def test_globals_and_systematic_shared_adds_both_cross_terms():
    rng = np.random.default_rng(3)
    Ra, Rb = _spd(rng, 4), _spd(rng, 4)
    ga, gb, sa, sb = (_vec(rng, 4, 2) for _ in range(4))
    r = combine_covariances(
        Ra + _cov(ga) + _cov(sa), Rb + _cov(gb) + _cov(sb),
        vec_global_a=ga, vec_global_b=gb,
        vec_systematic_a=sa, vec_systematic_b=sb,
        share_mode="globals_and_systematic_shared")
    assert np.allclose(r.cov_combined, Ra + Rb + _cov(ga - gb) + _cov(sa - sb),
                       atol=1e-10)


def test_input_contracts():
    rng = np.random.default_rng(4)
    a, b = _spd(rng, 3), _spd(rng, 4)
    with pytest.raises(ValueError, match="same shape"):
        combine_covariances(a, b)
    a, b = _spd(rng, 3), _spd(rng, 3)
    with pytest.raises(ValueError, match="globals_shared requires"):
        combine_covariances(a, b, share_mode="globals_shared")
    g = _vec(rng, 3, 2)
    with pytest.raises(ValueError, match="globals_and_systematic"):
        combine_covariances(a, b, vec_global_a=g, vec_global_b=g,
                            share_mode="globals_and_systematic_shared")
    with pytest.raises(ValueError, match="must both be"):
        combine_covariances(a, b, vec_global_a=g, vec_global_b=g[:, :1],
                            share_mode="globals_shared")
    with pytest.raises(ValueError, match="signed vector"):
        combine_covariances(a, b, cov_global_a=a, cov_global_b=b,
                            share_mode="globals_shared")
    with pytest.raises(ValueError, match="unknown share_mode"):
        combine_covariances(a, b, share_mode="nonsense")


def test_vectors_not_of_the_totals_are_refused():
    # vectors far larger than the totals they are claimed to be part of: the
    # difference covariance would be indefinite, so no result is given.
    rng = np.random.default_rng(5)
    total = _spd(rng, 3, scale=0.1)
    g = _vec(rng, 3, 2, scale=5.0)
    with pytest.raises(ValueError, match="not positive semi-definite"):
        combine_covariances(total, total, vec_global_a=g, vec_global_b=g,
                            share_mode="globals_shared")


def test_result_type_and_fields():
    rng = np.random.default_rng(6)
    a, b = _spd(rng, 2), _spd(rng, 2)
    r = combine_covariances(a, b, share_mode="all_independent")
    assert isinstance(r, CombinedCovariance)
    for f in ("cov_combined", "cov_naive", "sigma_naive",
              "sigma_combined", "reduction_factor"):
        assert getattr(r, f) is not None


def _well(azi, error_model="ISCWSA MWD Rev5.11", dazi=0.0):
    md = np.arange(0.0, 3001.0, 30.0)
    inc = np.clip((md - 600.0) / 10.0, 0.0, 60.0)
    return Survey(md=md, inc=inc, azi=np.full_like(md, azi + dazi),
                  header=SurveyHeader(**MAG_REF), error_model=error_model)


def test_shared_declination_vector_is_the_position_derivative():
    # Oracle independent of the error model: DEC-U is one azimuth offset for
    # the whole well, so its signed vector is d(position)/d(azimuth) times its
    # magnitude, by finite difference of the minimum-curvature positions. The
    # error model propagates on the balanced-tangential basis, hence the
    # tolerance.
    from welleng.conditioning import source_vectors
    a, b = _well(30.0), _well(210.0)
    codes, va, vb = source_vectors(a, b, "global")
    k = codes.index("DEC-U")
    mag = float(np.linalg.norm(a.err.errors.errors["DEC-U"].e_DIA[-1]))
    d = 1e-6
    for s, v, azi in ((a, va, 30.0), (b, vb, 210.0)):
        p0 = np.c_[s.n, s.e, s.tvd]
        s1 = _well(azi, None, np.rad2deg(d))
        jac = (np.c_[s1.n, s1.e, s1.tvd] - p0) / d * mag
        assert np.allclose(v[:, k], jac, rtol=1e-3, atol=1e-6 * np.abs(jac).max())


def test_end_to_end_opposite_wells_share_globals_wider_than_independent():
    # Two wells from one slot, departing in opposite azimuths: one declination
    # error rotates them apart, so the shared globals widen their relative
    # uncertainty at TD; the same wells on one azimuth cancel the globals.
    from welleng.conditioning import source_vectors
    a = _well(0.0)
    for azi_b, widens in ((180.0, True), (0.0, False)):
        b = _well(azi_b)
        codes, va, vb = source_vectors(a, b, "global")
        assert "DEC-U" in codes and "DSTG" in codes
        r = combine_covariances(a.cov_nev, b.cov_nev, vec_global_a=va,
                                vec_global_b=vb, share_mode="globals_shared")
        assert bool(r.sigma_combined[-1] > r.sigma_naive[-1]) is widens
    assert np.allclose(r.cov_combined, a.cov_nev + b.cov_nev
                       - 2.0 * np.einsum("nki,nkj->nij", va, va), atol=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
