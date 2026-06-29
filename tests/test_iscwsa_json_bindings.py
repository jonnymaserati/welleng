"""Every formula variable in every shipped OWSG JSON model must be bindable.

A silent binding-name mismatch -- a formula referencing ``Bfield`` while the
interpreter binds ``BField``, or ``GField`` vs ``Gfield`` -- makes the interpreter
throw, the term is caught, and it contributes ZERO covariance *silently*. That is
exactly how XCLTortuosity (whole XCL term) and GField/Bfield (the ABIXY/MFI axial-
interference terms) were dead for ~32 models without any test noticing.

This sweep evaluates every formula axis of every shipped model against the full
production binding set and fails if any variable is unbound, so the class of bug
can't regress.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from welleng.errors.conformance import _bindings_from_survey, standard_test_survey
from welleng.errors.interpreter import evaluate_formula

JSON_ROOT = Path(__file__).parent.parent / "welleng" / "errors" / "iscwsa_json"

# Per-tool parameter(s) that non-deferred terms reference; supplied so only
# GENUINELY unbound (mis-named) variables surface. XCLTortuosity is usually carried
# in the model's own parameters block, but a fallback here keeps models that omit it
# from false-flagging.
_EXTRA = {
    "XCLTortuosity": 5.726e-4,
}

# Gyro drift / random-walk / noise terms are evaluated station-by-station via the
# recurrence path (tool_errors), not a single vectorised formula eval -- and the
# Z-gyro variants (GZ-*) are not yet wired (the recurrence path is XY-only). Their
# per-tool / state variables (GXYRunningSpeed, GZRunningSpeed, XY/Z_Gyro_Drift, ...)
# are out of scope for this binding-name check; skip the family. Tracked as deferred
# gyro work (GYRO_SURVEY_PLAN.md / FUTURE_WORK).
_DEFERRED = re.compile(r"^G(XY|Z|XYZ)-(GD|GRW|RN)")

_AXES = (
    "depth_formula", "inclination_formula", "azimuth_formula",
    "north_singularity", "east_singularity", "vertical_singularity",
)

_MODELS = sorted(JSON_ROOT.glob("owsg_*/*.json"))


def _full_bindings() -> dict:
    b = dict(_bindings_from_survey(standard_test_survey()))
    b.update(_EXTRA)
    return b


@pytest.mark.parametrize(
    "json_path", _MODELS, ids=[f"{p.parent.name}/{p.stem}" for p in _MODELS]
)
def test_all_formula_variables_are_bindable(json_path):
    model = json.loads(json_path.read_text())
    bindings = _full_bindings()
    for k, v in (model.get("parameters") or {}).items():
        if isinstance(v, (int, float)):
            bindings[k] = float(v)

    undefined: dict[str, set] = {}
    for term in model["terms"]:
        if _DEFERRED.match(term["name"]):
            continue
        for axis in _AXES:
            f = term.get(axis)
            if not isinstance(f, str):
                continue
            try:
                with np.errstate(all="ignore"):
                    evaluate_formula(f, bindings)
            except Exception as exc:  # noqa: BLE001 -- only NameErrors are bugs here
                m = re.search(r"name '([^']+)' is not defined", str(exc))
                if m:
                    undefined.setdefault(m.group(1), set()).add(term["name"])

    assert not undefined, (
        f"{json_path.name}: formula variable(s) not in the interpreter bindings -- "
        f"these terms would silently contribute ZERO covariance: "
        + "; ".join(f"{var!r} in {sorted(terms)}" for var, terms in undefined.items())
    )
