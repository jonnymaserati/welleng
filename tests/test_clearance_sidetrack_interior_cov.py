"""Interior covariance of a sidetrack reference below the KOP.

The reference below ``kop_depth`` is rebuilt with its covariance re-zeroed at
the KOP (ISCWSA paragraph 81) and no error model, so between stations it used
to interpolate that covariance linearly. It now takes the error model's own
interior covariance, re-zeroed source by source, which must equal the
re-zeroed station covariance at every station and be zero at the KOP.

Data: the ISCWSA standard set of well paths for evaluating clearance
scenarios (r4), reference well and offset 10 (sidetrack at MD 900 m).
"""
import numpy as np

from welleng.clearance import IscwsaClearance, MahalanobisClearance
from welleng.survey import _interior_cov_nev

from tests.test_clearance_iscwsa import data, generate_surveys

KOP = 900.0


def _clearances():
    surveys = generate_surveys(data)
    ref, off = surveys["Reference well"], surveys["10 - well"]
    return (IscwsaClearance(ref, off, kop_depth=KOP, minimize_sf=False),
            MahalanobisClearance(ref, off, kop_depth=KOP))


def _rel(a, b):
    return np.abs(a - b).max() / max(np.abs(b).max(), 1e-30)


def test_station_exact_and_zero_at_the_kop():
    c, _ = _clearances()
    cov = np.asarray(c.ref.cov_nev).reshape(-1, 3, 3)
    assert np.all(c._kop_cov_nev_at(KOP) == 0.0)
    assert np.all(cov[0] == 0.0)
    worst = max(_rel(c._kop_cov_nev_at(float(m)), cov[j])
                for j, m in enumerate(c.ref.md) if j > 0)
    assert worst < 1e-9, worst


def test_interior_rule_and_both_clearances_use_it():
    c, m = _clearances()
    md = np.asarray(c.ref.md)
    curve = m._curve(m.ref)
    for j in (1, len(md) // 2, len(md) - 2):
        q = 0.5 * (md[j] + md[j + 1])
        want = c._kop_cov_nev_at(float(q))
        assert np.array_equal(_interior_cov_nev(c.ref, j, q - md[j]), want)
        assert np.array_equal(m._at(curve, q)[1], want)


def test_interior_is_positive_semidefinite_and_continuous():
    c, _ = _clearances()
    md = np.asarray(c.ref.md)
    qs = np.linspace(md[0], md[-1], 400)  # grid-lint: ok query sample
    covs = np.array([c._kop_cov_nev_at(float(q)) for q in qs])
    assert np.all(np.linalg.eigvalsh(covs)[:, 0] > -1e-9)
    # no jump across a station: a step 1 mm either side
    for m_ in md[1:-1]:
        lo, hi = c._kop_cov_nev_at(m_ - 1e-3), c._kop_cov_nev_at(m_ + 1e-3)
        assert _rel(lo, hi) < 1e-3


def test_inserted_minimum_carries_the_re_zeroed_interior():
    surveys = generate_surveys(data)
    c = IscwsaClearance(surveys["Reference well"], surveys["10 - well"],
                        kop_depth=KOP)
    base = IscwsaClearance(surveys["Reference well"], surveys["10 - well"],
                           kop_depth=KOP, minimize_sf=False)
    inserted = sorted(set(np.round(c.ref.md, 9)) - set(np.round(base.ref.md, 9)))
    assert inserted
    cov = np.asarray(c.ref.cov_nev).reshape(-1, 3, 3)
    for q in inserted:
        j = int(np.argmin(np.abs(c.ref.md - q)))
        assert _rel(cov[j], base._kop_cov_nev_at(float(c.ref.md[j]))) < 1e-12
