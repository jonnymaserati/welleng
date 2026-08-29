"""Geomagnetic field: an offline WMM model + the BGS web-service client.

Two ways to get declination / dip / total field for a survey location:

- :func:`local_field` -- **offline**. The bundled World Magnetic Model (WMM2025,
  US-Gov public domain) evaluated locally by spherical-harmonic synthesis: no
  network, deterministic, fast, valid for its five-year window (2025-2030). This
  is the default path of :func:`lookup_field` when the request is in window.
- :func:`lookup_field` with ``source="bgs"`` -- the British Geological Survey
  ``GMModels`` web service, for IGRF / historic / out-of-window dates and other
  models the local WMM cannot serve. Requires a network.

``lookup_field`` (``source="auto"``, the default) uses the offline model when it
applies and falls back to BGS otherwise, so the common case (current-epoch WMM)
needs no network while the long tail still works online.

The local model reproduces NOAA's official WMM2025 test values to < 0.001 nT in
every field component (the model's own numerical-agreement spec is 0.1 nT), far
below the model's ~physical uncertainty. Coefficients: ``welleng/data/wmm2025.json``
(converted from ``WMM2025.COF``); see its ``provenance`` field.

Service documentation:
https://geomag.bgs.ac.uk/web_service/GMModels/help/parameters
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np

BGS_GEOMAG_URL = "https://geomag.bgs.ac.uk/web_service/GMModels"

#: models useful to welleng: 'wmm' (current-epoch default) and 'igrf'
#: (historic coverage — the fallback for pre-window survey dates).
KNOWN_MODELS = ("wmm", "igrf")

# WGS84 ellipsoid + the WMM geomagnetic reference radius.
_WGS84_A = 6378137.0
_WGS84_F = 1.0 / 298.257223563


class GeomagLookupError(ValueError):
    """The geomagnetic lookup failed (bad request, out of window, no network)."""


@lru_cache(maxsize=1)
def _wmm_model():
    """Load + cache the bundled WMM coefficients as (epoch, N, g, h, gd, hd)."""
    text = files("welleng.data").joinpath("wmm2025.json").read_text()
    doc = json.loads(text)
    N = int(doc["max_degree"])
    g = np.zeros((N + 1, N + 1))
    h = np.zeros((N + 1, N + 1))
    gd = np.zeros((N + 1, N + 1))
    hd = np.zeros((N + 1, N + 1))
    for n, m, gv, hv, gdv, hdv in doc["coeffs"]:
        g[n, m], h[n, m], gd[n, m], hd[n, m] = gv, hv, gdv, hdv
    return (
        float(doc["epoch"]), N, g, h, gd, hd,
        float(doc["reference_radius_m"]),
        float(doc["valid_from"]), float(doc["valid_to"]),
    )


def wmm_validity():
    """(valid_from, valid_to) decimal years of the bundled WMM model."""
    _, _, _, _, _, _, _, vf, vt = _wmm_model()
    return vf, vt


@lru_cache(maxsize=4)
def _schmidt_norm(N):
    """Gauss-normalised -> Schmidt semi-normalised factors S[n,m].

    Depends only on the degree N (not on position), so it is model-invariant --
    cached rather than rebuilt on every field evaluation.
    """
    S = np.zeros((N + 1, N + 1))
    S[0, 0] = 1.0
    for n in range(1, N + 1):
        S[n, 0] = S[n - 1, 0] * (2 * n - 1) / n
        for m in range(1, n + 1):
            fac = (2.0 if m == 1 else 1.0) * (n - m + 1) / (n + m)
            S[n, m] = S[n, m - 1] * np.sqrt(fac)
    return S


def _schmidt_legendre(theta, N):
    """Schmidt semi-normalised ``P[n,m](cos theta)`` and ``dP/dtheta``."""
    ct, st = np.cos(theta), np.sin(theta)
    P = np.zeros((N + 1, N + 1))
    dP = np.zeros((N + 1, N + 1))
    P[0, 0] = 1.0
    for n in range(1, N + 1):
        for m in range(0, n + 1):
            if n == m:
                P[n, m] = st * P[n - 1, m - 1]
                dP[n, m] = st * dP[n - 1, m - 1] + ct * P[n - 1, m - 1]
            else:
                Pn2 = P[n - 2, m] if m <= n - 2 else 0.0
                dPn2 = dP[n - 2, m] if m <= n - 2 else 0.0
                Knm = (((n - 1) ** 2 - m ** 2)
                       / ((2 * n - 1) * (2 * n - 3))) if n > 1 else 0.0
                P[n, m] = ct * P[n - 1, m] - Knm * Pn2
                dP[n, m] = ct * dP[n - 1, m] - st * P[n - 1, m] - Knm * dPn2
    S = _schmidt_norm(N)
    return P * S, dP * S


def local_field(latitude, longitude, altitude=0.0, date=None):
    """Geomagnetic field from the bundled WMM, computed OFFLINE.

    Parameters
    ----------
    latitude, longitude : float
        Geodetic coordinates in decimal degrees.
    altitude : float
        Height in METRES above the WGS84 spheroid (negative below datum).
    date : float or str, optional
        Decimal year (e.g. ``2027.5``) or ``"yyyy-mm-dd"``. ``None`` -> the
        model epoch.

    Returns
    -------
    dict
        The same shape as :func:`lookup_field` -- a
        ``geomagnetic-field-model-result`` payload with ``field-value``
        (``total-intensity``/``inclination``/``declination``/
        ``horizontal-intensity``/``north-intensity``/``east-intensity``/
        ``vertical-intensity``, each ``{"value", "units"}``), plus
        ``model_base_date`` and ``model_name`` -- so it is a drop-in for the
        BGS payload.

    Raises
    ------
    GeomagLookupError
        If ``date`` is outside the bundled model's validity window.
    """
    epoch, N, g0, h0, gd, hd, re_m, vf, vt = _wmm_model()
    year = _decimal_year(date) if date is not None else epoch
    if not (vf <= year <= vt):
        raise GeomagLookupError(
            f"date {year:.4f} is outside the bundled WMM validity window "
            f"[{vf:.1f}, {vt:.1f}]; use lookup_field(source='bgs') for "
            "IGRF / historic / out-of-window dates"
        )
    dt = year - epoch
    g = g0 + dt * gd
    h = h0 + dt * hd

    lat = np.radians(latitude)
    lon = np.radians(longitude)
    # geodetic -> geocentric spherical (radius r, colatitude theta)
    f = _WGS84_F
    a = _WGS84_A
    e2 = 2 * f - f * f
    cl, sl = np.cos(lat), np.sin(lat)
    rc = a / np.sqrt(1 - e2 * sl * sl)                 # prime-vertical radius
    p = (rc + altitude) * cl
    z = (rc * (1 - e2) + altitude) * sl
    r = np.hypot(p, z)
    theta = np.arctan2(p, z)
    st = np.sin(theta)
    st = st if abs(st) > 1e-12 else 1e-12              # pole guard

    P, dP = _schmidt_legendre(theta, N)
    ratio = re_m / r
    Br = Bt = Bp = 0.0
    for n in range(1, N + 1):
        rn = ratio ** (n + 2)
        for m in range(0, n + 1):
            cml, sml = np.cos(m * lon), np.sin(m * lon)
            gh = g[n, m] * cml + h[n, m] * sml
            ghd = -g[n, m] * sml + h[n, m] * cml
            Br += rn * (n + 1) * gh * P[n, m]
            Bt += -rn * gh * dP[n, m]
            Bp += -rn * m * ghd * P[n, m] / st
    # geocentric spherical -> geodetic X(north) Y(east) Z(down)
    Xg, Yg, Zg = -Bt, Bp, -Br
    psi = (np.pi / 2 - lat) - theta                    # geocentric->geodetic tilt
    X = Xg * np.cos(psi) - Zg * np.sin(psi)
    Z = Xg * np.sin(psi) + Zg * np.cos(psi)
    Y = Yg
    H = np.hypot(X, Y)
    F = np.hypot(H, Z)
    D = np.degrees(np.arctan2(Y, X))
    incl = np.degrees(np.arctan2(Z, H))

    def _v(value, units):
        return {"value": float(value), "units": units}

    return {
        "field-value": {
            "total-intensity": _v(F, "nT"),
            "horizontal-intensity": _v(H, "nT"),
            "north-intensity": _v(X, "nT"),
            "east-intensity": _v(Y, "nT"),
            "vertical-intensity": _v(Z, "nT"),
            "inclination": _v(incl, "deg (down)"),
            "declination": _v(D, "deg (east)"),
        },
        "model_name": "WMM2025",
        "model_base_date": f"{year:.4f}",
        "source": "local-wmm",
    }


def _decimal_year(date):
    """A decimal year from a float year or a ``yyyy-mm-dd`` string."""
    if isinstance(date, (int, float)):
        return float(date)
    from datetime import date as _d
    y, m, d = (int(x) for x in str(date).split("-")[:3])
    start = _d(y, 1, 1).toordinal()
    frac = (_d(y, m, d).toordinal() - start) / (_d(y + 1, 1, 1).toordinal() - start)
    return y + frac


def _build_url(
    latitude,
    longitude,
    altitude=0.0,
    date=None,
    model="wmm",
    revision="current",
):
    """Build the GMModels request URL. ``altitude`` is in METRES."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        # BGS expects km above the WGS84 spheroid; welleng altitudes are m
        "altitude": altitude / 1000.0,
        "format": "json",
    }
    if date is not None:
        params["date"] = date
    return (
        f"{BGS_GEOMAG_URL}/{model}/{revision}?"
        f"{urllib.parse.urlencode(params)}"
    )


def _bgs_field(latitude, longitude, altitude, date, model, revision, timeout):
    """One BGS ``GMModels`` web-service call (network)."""
    url = _build_url(latitude, longitude, altitude, date, model, revision)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise GeomagLookupError(
            f"BGS geomag lookup failed ({exc.code}): {detail[:300]}"
        ) from exc
    except Exception as exc:
        raise GeomagLookupError(f"BGS geomag lookup failed: {exc}") from exc
    try:
        return payload["geomagnetic-field-model-result"]
    except (KeyError, TypeError) as exc:
        raise GeomagLookupError(
            f"BGS geomag lookup returned an unexpected payload: "
            f"{str(payload)[:300]}"
        ) from exc


def lookup_field(
    latitude,
    longitude,
    altitude=0.0,
    date=None,
    model="wmm",
    revision="current",
    timeout=10.0,
    source="auto",
):
    """Look up the geomagnetic field, offline where possible.

    Parameters
    ----------
    latitude, longitude : float
        Geodetic coordinates in decimal degrees.
    altitude : float
        Height in METRES above the WGS84 spheroid (negative below — e.g. a
        subsea wellhead). Converted to the service's km at the wire.
    date : str or float, optional
        ``yyyy-mm-dd`` or a decimal year. Omitted -> the model epoch.
    model : str
        ``'wmm'`` (default) or ``'igrf'`` (historic; BGS only).
    revision : str
        BGS model revision; ``'current'`` tracks the latest server-side.
    timeout : float
        Socket timeout in seconds (BGS path only).
    source : {"auto", "local", "bgs"}
        ``"auto"`` (default): the bundled offline WMM when it applies
        (``model="wmm"`` and ``date`` in its window), else BGS. ``"local"``
        forces the offline WMM (raises out of window). ``"bgs"`` forces the web
        service.

    Returns
    -------
    dict
        A ``geomagnetic-field-model-result`` payload:
        ``result['field-value']['total-intensity']['value']`` (nT),
        ``['inclination']`` (dip, deg down-positive),
        ``['declination']`` (deg east-positive). The offline and BGS payloads
        share this shape.

    Raises
    ------
    GeomagLookupError
        Out-of-window local request, HTTP error, connection failure or bad
        payload.
    """
    if source not in ("auto", "local", "bgs"):
        raise ValueError(f"source must be auto/local/bgs, got {source!r}")
    if source == "local":
        return local_field(latitude, longitude, altitude, date)
    if source == "auto" and model == "wmm":
        vf, vt = wmm_validity()
        year = _decimal_year(date) if date is not None else None
        if year is None or vf <= year <= vt:
            try:
                return local_field(latitude, longitude, altitude, date)
            except GeomagLookupError:
                pass                                    # fall through to BGS
    return _bgs_field(
        latitude, longitude, altitude, date, model, revision, timeout
    )


_M_PER_DEG_LAT = 111320.0


@dataclass(frozen=True)
class SurveyField:
    """Per-station geomagnetic reference field along a survey (offline WMM).

    Attributes
    ----------
    md : (n,) ndarray
        Station measured depths (echoed from the survey).
    b_total : (n,) ndarray
        Total field per station, nT.
    dip : (n,) ndarray
        Magnetic dip (inclination) per station, deg, positive down.
    declination : (n,) ndarray
        Declination per station, deg, positive east.
    d_b_total, d_dip, d_declination : (n,) ndarray
        Difference of each quantity from the wellhead reference (station 0) --
        the SIZE of the per-station refinement. ``d_declination`` is the one
        that matters: it rotates azimuth, so its span over the well is the
        potential lateral-position gain of using a per-station reference.
    reference : dict
        The single wellhead-reference field ``local_field`` returns, for
        comparison / the conventional single-value path.
    """

    md: np.ndarray            # (n,) station measured depths, echoed
    b_total: np.ndarray       # (n,) total field per station, nT
    dip: np.ndarray           # (n,) magnetic dip per station, deg (down +)
    declination: np.ndarray   # (n,) declination per station, deg (east +)
    d_b_total: np.ndarray     # (n,) b_total minus the wellhead reference, nT
    d_dip: np.ndarray         # (n,) dip minus the wellhead reference, deg
    d_declination: np.ndarray  # (n,) declination minus wellhead ref, deg (the key one)
    reference: dict           # the single wellhead-reference local_field payload


def field_along_survey(survey, date=None, descend=False):
    """Per-station geomagnetic reference field along a survey, computed OFFLINE.

    An **opt-in refinement**: instead of one wellhead declination/dip/B for the
    whole well, evaluate the bundled WMM at each station's own geographic
    position. For an extended-reach well crossing a field gradient the
    declination varies across the trajectory; using a single surface value puts
    a small systematic azimuth error on the deep, stepped-out sections. The
    effect is negligible for ordinary wells (< 0.01 deg declination) and reaches
    ~0.1-0.2 deg (~10-20 m of lateral position over the reach) only for long
    extended-reach wells at high latitude or near a magnetic anomaly.

    By default this varies the horizontal position (latitude/longitude from the
    wellhead plus the survey's north/east) and holds altitude at the wellhead
    reference -- the LATERAL gradient, which is the real, physically-clean
    effect. The vertical (downward-continuation to true vertical depth) is
    deliberately NOT applied by default: it is both tiny (< 0.03 deg over 3 km)
    and outside the WMM's certified altitude range (its floor is -1 km, and the
    crustal field a well actually descends into is not in the model). Pass
    ``descend=True`` to evaluate at ``-tvd`` anyway, understanding it is a minor
    extrapolation.

    Horizontal position uses a local flat-earth step from the wellhead
    (``dlat = north / 111320``, ``dlon = east / (111320 cos lat)``); over a
    well's extent that is well within the smoothness of the geomagnetic field.

    Parameters
    ----------
    survey : welleng.survey.Survey
        A survey whose header carries ``latitude``/``longitude`` (the wellhead)
        and which exposes ``n``/``e``/``tvd`` station positions.
    date : str or float, optional
        ``yyyy-mm-dd`` or a decimal year. Defaults to the header's
        ``survey_date`` if set, else the model epoch.
    descend : bool, default False
        If True, evaluate at each station's ``-tvd`` (downward continuation);
        default holds altitude at the wellhead reference (lateral gradient only).

    Returns
    -------
    SurveyField

    Notes
    -----
    - This is the SCALAR reference applied per station (one ``local_field`` call
      each) -- the open, auditable form.
    - It changes the reference field's MEAN, not the ISCWSA declination
      UNCERTAINTY term (which still covers the spread). And it is a modelling
      refinement of the TRUE field -- do not use it to silently re-correct a
      survey whose azimuths already had a declination applied at acquisition;
      its value is on the raw magnetic-to-true path.
    """
    hdr = survey.header
    lat0 = getattr(hdr, "latitude", None)
    lon0 = getattr(hdr, "longitude", None)
    if lat0 is None or lon0 is None or getattr(hdr, "_location_defaulted", False):
        raise GeomagLookupError(
            "survey header has no real latitude/longitude (wellhead) to anchor "
            "the per-station field -- a defaulted location is fictitious; set "
            "the header's latitude/longitude"
        )
    if date is None:
        date = getattr(hdr, "survey_date", None)
    alt0 = float(getattr(hdr, "altitude", 0.0) or 0.0)

    n = np.asarray(survey.n, dtype=float)
    e = np.asarray(survey.e, dtype=float)
    tvd = np.asarray(survey.tvd, dtype=float)
    lats = lat0 + n / _M_PER_DEG_LAT
    lons = lon0 + e / (_M_PER_DEG_LAT * np.cos(np.radians(lat0)))
    alts = (alt0 - tvd) if descend else np.full(lats.shape, alt0)

    B = np.empty(lats.shape)
    D = np.empty(lats.shape)
    Dec = np.empty(lats.shape)
    for i in range(lats.shape[0]):
        fv = local_field(float(lats[i]), float(lons[i]),
                         float(alts[i]), date)["field-value"]
        B[i] = fv["total-intensity"]["value"]
        D[i] = fv["inclination"]["value"]
        Dec[i] = fv["declination"]["value"]

    reference = local_field(lat0, lon0, alt0, date)
    rfv = reference["field-value"]
    return SurveyField(
        md=np.asarray(survey.md, dtype=float),
        b_total=B, dip=D, declination=Dec,
        d_b_total=B - rfv["total-intensity"]["value"],
        d_dip=D - rfv["inclination"]["value"],
        d_declination=Dec - rfv["declination"]["value"],
        reference=reference,
    )
