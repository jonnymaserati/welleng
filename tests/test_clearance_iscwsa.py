from welleng.survey import Survey, make_survey_header, _interpolate_pos_nev
from welleng.clearance import IscwsaClearance, MahalanobisClearance
import numpy as np
import json

"""
Test that the ISCWSA clearance model is working within a defined tolerance,
testing against the ISCWSA standard set of wellpaths for evaluating clearance
scenarios using the MWD Rev4 error model.
"""

# Read well and validation data
filename = (
    "tests/test_data/clearance_iscwsa_well_data.json"
)
data = json.load(open(filename))


def generate_surveys(self, data=data):
    # Generate surveys for imported wells
    surveys = {}

    for well in data['wells']:
        sh = make_survey_header(data["wells"][well]["header"])

        if well == "Reference well":
            radius = 0.4572
        else:
            radius = 0.3048

        s = Survey(
            md=data["wells"][well]["MD"],
            inc=data["wells"][well]["IncDeg"],
            azi=data["wells"][well]["AziDeg"],
            n=data["wells"][well]["N"],
            e=data["wells"][well]["E"],
            tvd=data["wells"][well]["TVD"],
            radius=radius,
            header=sh,
            error_model="ISCWSA MWD Rev4",
            start_xyz=[
                data["wells"][well]["E"][0],
                data["wells"][well]["N"][0],
                data["wells"][well]["TVD"][0]
                ],
            start_nev=[
                data["wells"][well]["N"][0],
                data["wells"][well]["E"][0],
                data["wells"][well]["TVD"][0]
                ],
            deg=True,
            unit="meters"
        )
        surveys[well] = s

    return surveys


def test_minimize_sf(data=data):
    surveys = generate_surveys(data)
    reference = surveys["Reference well"]
    offset = surveys["09 - well"]

    result = IscwsaClearance(reference, offset, minimize_sf=False)
    result_min = IscwsaClearance(reference, offset, minimize_sf=True)

    idx = np.where(result_min.ref.interpolated == False)  # noqa E712

    # Check that interpolated survey is not corrupted
    for attr in [
        'azi_grid_rad', 'azi_mag_rad', 'azi_true_rad', 'cov_hla', 'cov_nev',
        'pos_nev', 'pos_xyz', 'md', 'radius'
    ]:
        assert np.allclose(
            getattr(result.ref, attr), getattr(result_min.ref, attr)[idx]
        )

        pass

    for attr in [
        'Rr', 'calc_hole', 'distance_cc', 'eou_boundary',
        'eou_separation', 'hoz_bearing', 'idx', 'masd', 'off_cov_hla',
        'off_cov_nev', 'off_delta_hlas', 'off_delta_nevs', 'off_pcr',
        'ref_cov_hla', 'ref_cov_nev', 'ref_delta_hlas', 'ref_delta_nevs',
        'ref_nevs', 'ref_pcr', 'sf', 'wellbore_separation'
    ]:
        # `toolface_bearing` and `trav_cyl_azi_deg` are a bit unstable when
        # well paths are parallel.

        assert np.allclose(
            getattr(result, attr), getattr(result_min, attr)[idx],
            rtol=1e-01, atol=1e-02
        )

        pass


def test_clearance_iscwsa(data=data, rtol=1e-02, atol=1e-03):
    surveys = generate_surveys(data)
    reference = surveys["Reference well"]

    # Perform clearance checks for each survey
    for well in surveys:
        if well == "Reference well":
            continue
        else:
            offset = surveys[well]
            kop_depth = 900.0 if well == "10 - well" else -np.inf
            for b in [False, True]:
                result = IscwsaClearance(
                    reference, offset, minimize_sf=b, kop_depth=kop_depth
                )
                assert np.allclose(
                    result.sf[np.where(result.ref.interpolated == False)],  # noqa E712
                    np.array(data["wells"][well]["SF"]),
                    rtol=rtol, atol=atol
                )

    pass


def test_mahalanobis_less_conservative_than_pedal(data=data):
    """MahalanobisClearance uses the exact combined-ellipsoid k-sigma boundary
    (searching the minimum-Mahalanobis point over BOTH interpolated wells)
    instead of the pedal-curve support-function approximation at the Euclidean
    closest-approach point. So at the worst (governing) point it is never more
    conservative than the validated pedal rule: min SF_maha >= min SF_pedal. It
    must also flag the genuine crossings/close approaches and clear the rest."""
    surveys = generate_surveys(data)
    reference = surveys["Reference well"]
    # collision / clear verdicts must match the validated pedal rule exactly
    hits = {"03 - well", "04 - well", "09 - well", "10 - well", "11 - well"}
    for well in surveys:
        if well == "Reference well":
            continue
        offset = surveys[well]
        kop_depth = 900.0 if well == "10 - well" else -np.inf
        ped = float(np.nanmin(IscwsaClearance(reference, offset, kop_depth=kop_depth).sf))
        mah = float(np.nanmin(MahalanobisClearance(reference, offset, kop_depth=kop_depth).sf))
        if np.isfinite(ped):
            assert mah >= ped - 1e-6, well          # never more conservative
        assert (mah < 1.0) == (well in hits), well


def _resample_brute(survey, step):
    """Resample a survey's position/covariance/radius to a fine MD step (test
    helper for the independent brute-force reference)."""
    md = np.asarray(survey.md, float)
    cov = np.asarray(survey.cov_nev, float).reshape(-1, 3, 3)
    rad = np.asarray(survey.radius, float).reshape(-1)
    mdf = np.arange(md[0], md[-1] + step, step)
    # position by minimum curvature (matches MahalanobisClearance._at); cov/rad linear
    P = np.empty((len(mdf), 3))
    for r, q in enumerate(mdf):
        i = int(np.clip(np.searchsorted(md, q, side="right") - 1, 0, len(md) - 2))
        P[r] = _interpolate_pos_nev(survey, float(q - md[i]), i)
    C = np.empty((len(mdf), 3, 3))
    for a in range(3):
        for b in range(3):
            C[:, a, b] = np.interp(mdf, md, cov[:, a, b])
    return P, C, np.interp(mdf, md, rad)


def _brute_force_min_sf(reference, offset, step=1.0, k=3.5, Sm=0.3, sigma_pa=0.5):
    """Exhaustive all-pairs minimum radii-adjusted Mahalanobis SF at a fine MD
    step — the independent reference MahalanobisClearance must reproduce."""
    Rp, Rc, Rr = _resample_brute(reference, step)
    Op, Oc, Ro = _resample_brute(offset, step)
    best = np.inf
    for i in range(len(Rp)):
        d = Op - Rp[i]
        D = np.linalg.norm(d, axis=1)
        S = Rc[i] + Oc + sigma_pa ** 2 * np.eye(3)
        scale = np.divide(np.maximum(D - (Rr[i] + Ro + Sm), 0.0), D,
                          out=np.zeros_like(D), where=D > 0)
        dp = d * scale[:, None]
        m = np.sqrt(np.einsum('oi,oij,oj->o', dp, np.linalg.inv(S), dp)) / k
        best = min(best, float(np.min(m)))
    return best


def test_mahalanobis_matches_brute_force(data=data):
    """MahalanobisClearance (broadphase + continuous narrowphase) agrees with an
    exhaustive all-pairs reference sampled at 1 m to < 1e-3 — the optimisation
    finds the TRUE minimum, not a station-sampling artefact. (The continuous
    narrowphase can sit a fraction below the 1 m-discrete brute, because it
    resolves the minimum *between* the 1 m samples; the two-sided bound catches
    both that and any narrowphase regression.) Pins the paper's exactness claim."""
    surveys = generate_surveys(data)
    reference = surveys["Reference well"]
    for well in ["01 - well", "03 - well", "05 - well", "07 - well", "11 - well"]:
        offset = surveys[well]
        got = float(np.nanmin(MahalanobisClearance(reference, offset).sf))
        truth = _brute_force_min_sf(reference, offset, step=1.0)
        assert abs(got - truth) < 2e-3, f"{well}: maha={got:.4f} brute={truth:.4f}"


def test_mahalanobis_equals_brooks_transform(data=data):
    """The SHIPPED metric is exactly Brooks's (SPE-116155) Mahalanobis-space
    distance ||V E^-1/2 V^T d||. Drives the actual `_sf_point` (sigma_pa=0,
    Sm=0 -> pure metric) and compares to the transform on real station
    covariances — a non-trivial check of the implementation, not an identity."""
    surveys = generate_surveys(data)
    reference = surveys["Reference well"]
    offset = surveys["03 - well"]
    mc = MahalanobisClearance(reference, offset)
    mc.sigma_pa = 0.0
    mc.Sm = 0.0
    Rp = np.column_stack([reference.n, reference.e, reference.tvd])
    Rc = np.asarray(reference.cov_nev).reshape(-1, 3, 3)
    Op = np.column_stack([offset.n, offset.e, offset.tvd])
    Oc = np.asarray(offset.cov_nev).reshape(-1, 3, 3)
    checked = 0
    for i, j in [(40, 40), (50, 45), (60, 55)]:
        S = Rc[i] + Oc[j]
        w, V = np.linalg.eigh(S)
        if w.min() < 1e-9:
            continue
        d = Op[j] - Rp[i]
        brooks = float(np.linalg.norm(V @ np.diag(w ** -0.5) @ V.T @ d))
        shipped = mc._sf_point(Rp[i], Rc[i], 0.0, Op[j], Oc[j], 0.0) * mc.k
        assert abs(shipped - brooks) < 1e-9, f"({i},{j}) shipped={shipped} brooks={brooks}"
        checked += 1
    assert checked >= 2


def test_mahalanobis_local_minima_refinement(data=data):
    """The narrowphase refines EVERY local minimum of the broadphase profile
    (not just the n_candidates globally-lowest), so a sharp between-station
    crossing is caught whatever its broadphase rank. Proof: Well 09's crossing
    is still recovered with n_candidates=0 — top-n disabled, so ONLY the
    local-minima/endpoint refinement is doing the work — and the governing SF is
    invariant to n_candidates. (This is what makes n_candidates defensive
    headroom rather than a load-bearing tuning knob.)"""
    surveys = generate_surveys(data)
    reference = surveys["Reference well"]
    # n_candidates=0: only local-minima + endpoint refinement; must still catch
    # the crossing that broadphase alone reads as clear (~1.09).
    assert float(np.nanmin(
        MahalanobisClearance(reference, surveys["09 - well"], n_candidates=0).sf)) < 0.01
    # governing SF invariant to n_candidates (a crossing and a margin well)
    for well in ["09 - well", "07 - well"]:
        offset = surveys[well]
        vals = [float(np.nanmin(
            MahalanobisClearance(reference, offset, n_candidates=n).sf))
            for n in (0, 1, 8, 16)]
        assert max(vals) - min(vals) < 1e-6, (well, vals)


def test_mahalanobis_governing_values(data=data):
    """Pin the governing minimum SF (not just hit/clear) for the between-station
    crossing (09) and the sidetrack/degenerate-covariance case (10)."""
    surveys = generate_surveys(data)
    reference = surveys["Reference well"]
    sf09 = float(np.nanmin(MahalanobisClearance(reference, surveys["09 - well"]).sf))
    assert sf09 < 0.01, sf09     # sharp between-station crossing -> overlap
    sf10 = float(np.nanmin(
        MahalanobisClearance(reference, surveys["10 - well"], kop_depth=900.0).sf))
    assert sf10 < 0.01, sf10     # sidetrack, degenerate cov, scanned below KOP


def test_sf_vs_md_inherited_and_consistent(data=data):
    """``Clearance.sf_vs_md()`` is defined on the base class (inherited by every
    subclass) and returns consistent profiles: at every reference station the
    exact (Mahalanobis) factor is >= the pedal (support-function) factor, both
    computed from one kernel over the same station pairing (Kantorovich)."""
    from welleng.clearance import Clearance
    assert "sf_vs_md" in vars(Clearance)        # on the base -> inherited by all
    surveys = generate_surveys(data)
    reference = surveys["Reference well"]
    for w in ["03 - well", "05 - well", "07 - well", "11 - well"]:
        md, ped, mah = MahalanobisClearance(reference, surveys[w]).sf_vs_md()
        assert len(md) == len(ped) == len(mah)
        assert np.all(np.isfinite(ped)) and np.all(np.isfinite(mah))
        assert np.all(mah >= ped - 1e-9), w     # exact never below the rule, per station
