"""Client for the BGS geomagnetic model web service.

Replaces the unmaintained ``magnetic_field_calculator`` package (which spoke
plain http, had no timeout and is no longer developed) with a small stdlib
client for the same service. The geomagnetic models themselves live server
side at the British Geological Survey, so this client stays current as BGS
revises them.

Service documentation:
https://geomag.bgs.ac.uk/web_service/GMModels/help/parameters

The service URL structure is ``GMModels/<model>/<revision>?<parameters>``
with altitude in km above the WGS84 spheroid and dates as ``yyyy-mm-dd``.
This module speaks welleng conventions at its boundary — altitude in METRES
(negative below datum) — and converts at the wire.
"""

import json
import urllib.error
import urllib.parse
import urllib.request

BGS_GEOMAG_URL = "https://geomag.bgs.ac.uk/web_service/GMModels"

#: models useful to welleng: 'wmm' (current-epoch default) and 'igrf'
#: (historic coverage — the fallback for pre-window survey dates).
KNOWN_MODELS = ("wmm", "igrf")


class GeomagLookupError(ValueError):
    """The BGS lookup failed (bad request, service down, no network)."""


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


def lookup_field(
    latitude,
    longitude,
    altitude=0.0,
    date=None,
    model="wmm",
    revision="current",
    timeout=10.0,
):
    """Look up the geomagnetic field from the BGS web service.

    Parameters
    ----------
    latitude, longitude : float
        Geodetic coordinates in decimal degrees.
    altitude : float
        Height in METRES above the WGS84 spheroid (negative below — e.g. a
        subsea wellhead). Converted to the service's km at the wire.
    date : str, optional
        ``yyyy-mm-dd``. Omitted -> the service's default epoch.
    model : str
        ``'wmm'`` (default) or ``'igrf'`` (historic coverage).
    revision : str
        Model revision; ``'current'`` tracks the latest server-side.
    timeout : float
        Socket timeout in seconds.

    Returns
    -------
    dict
        The service's ``geomagnetic-field-model-result`` payload, e.g.
        ``result['field-value']['total-intensity']['value']`` (nT),
        ``['inclination']`` (dip, deg down-positive) and
        ``['declination']`` (deg east-positive).

    Raises
    ------
    GeomagLookupError
        On any HTTP error (with the server's message, e.g. a date outside
        the model's validity window), connection failure or bad payload.
    """
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
        raise GeomagLookupError(
            f"BGS geomag lookup failed: {exc}"
        ) from exc
    try:
        return payload["geomagnetic-field-model-result"]
    except (KeyError, TypeError) as exc:
        raise GeomagLookupError(
            f"BGS geomag lookup returned an unexpected payload: "
            f"{str(payload)[:300]}"
        ) from exc
