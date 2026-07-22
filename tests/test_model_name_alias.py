"""The canonical welleng model name 'ISCWSA MWD Rev5.11' must resolve to the
shipped OWSG JSON (MWD+SRGM) so the JSON/interpreter path (and downstream
consumers that resolve term JSON by model name) work for the default model,
not only for the OWSG short name.

Guards the alias in tool_errors._MODEL_NAME_ALIASES against (a) the JSON going
missing/renamed and (b) the alias silently pointing at a non-equivalent model.
"""
import welleng as we
from welleng.errors.tool_errors import _resolve_json_model, _load_json_model


def test_rev511_resolves_to_mwd_srgm_json():
    p = _resolve_json_model("ISCWSA MWD Rev5.11")
    assert p is not None and p.endswith("MWD+SRGM.json")
    # the OWSG short name itself must still resolve (alias must not shadow it)
    assert _resolve_json_model("MWD+SRGM").endswith("MWD+SRGM.json")


def test_alias_target_is_term_equivalent():
    # The alias is only valid because the two names are the SAME model. Assert
    # the JSON term set equals welleng's 'ISCWSA MWD Rev5.11' term set.
    s = we.survey.Survey(
        md=[0, 100, 200], inc=[0, 30, 60], azi=[0, 45, 90], deg=True,
        header=we.survey.SurveyHeader(
            b_total=50000, dip=70, declination=0, azi_reference="grid"),
        error_model="ISCWSA MWD Rev5.11",
    )
    welleng_terms = set(s.err.errors.errors.keys())
    model = _load_json_model(_resolve_json_model("ISCWSA MWD Rev5.11"))
    json_terms = {t["name"] for t in model["terms"]}
    assert welleng_terms == json_terms, (
        f"alias points at a non-equivalent model: "
        f"welleng-only={welleng_terms - json_terms}, "
        f"json-only={json_terms - welleng_terms}"
    )
