"""SPE-187073 separation-factor acceptance criteria.

The exact bands, the tighten-only override, and the re-export location are
contract for consumers, so they are pinned here.
"""
import math

from welleng.acceptance import (
    ACCEPTABLE, CRITICAL, EXCLUDE, REVIEW,
    AcceptanceCriteria, Verdict, classify,
    SF_CRITICAL, SF_REVIEW, SF_EXCLUDE, K_HSE,
)


def test_the_standard_thresholds_are_the_papers_values():
    assert (SF_CRITICAL, SF_REVIEW, SF_EXCLUDE) == (1.0, 1.25, 5.0)
    assert K_HSE == 3.5
    c = AcceptanceCriteria.standard()
    assert (c.surface_margin_m, c.project_ahead_sigma_m) == (0.3, 0.5)
    assert "187073" in c.source


def test_the_standard_bands():
    c = AcceptanceCriteria.standard()
    assert c.classify(0.8).band == CRITICAL       # below mandatory -> STOP
    assert c.classify(0.999).band == CRITICAL
    assert c.classify(1.0).band == REVIEW          # mandatory met, below review
    assert c.classify(1.1).band == REVIEW
    assert c.classify(1.25).band == ACCEPTABLE     # at/above review
    assert c.classify(3.0).band == ACCEPTABLE
    assert c.classify(5.0).band == ACCEPTABLE
    assert c.classify(6.0).band == EXCLUDE         # drop from scanning


def test_only_critical_is_unacceptable_and_bool_reads():
    c = AcceptanceCriteria.standard()
    assert c.classify(0.8).acceptable is False
    assert bool(c.classify(0.8)) is False          # `if not verdict:` reads
    for sf in (1.0, 1.25, 6.0):
        assert c.classify(sf).acceptable is True
        assert bool(c.classify(sf)) is True


def test_operator_override_may_only_tighten():
    c = AcceptanceCriteria.standard()

    tight = c.with_operator_floor(1.5)             # raise the floor
    assert tight.operator_override is True
    assert tight.classify(1.4).band == CRITICAL    # now below the operator floor
    assert tight.classify(1.6).band == ACCEPTABLE
    assert "operator floor" in tight.classify(1.4).criterion.source

    # a floor below the mandatory SF = 1 is refused -- nothing may loosen it
    for bad in (0.9, 0.0, -1.0):
        try:
            c.with_operator_floor(bad)
        except ValueError as e:
            assert "TIGHTEN" in str(e)
        else:
            raise AssertionError(f"floor {bad} below SF=1 must be refused")


def test_the_verdict_carries_the_action_and_criterion():
    v = classify(0.8)
    assert isinstance(v, Verdict)
    assert "STOP DRILLING" in v.action
    assert v.criterion.operator_override is False   # the standard, unmodified
    assert math.isclose(v.sf, 0.8)


def test_re_exported_from_clearance_where_consumers_look():
    # pathfinder searched welleng.clearance for these; the re-export is contract
    from welleng.clearance import (  # noqa: F401
        AcceptanceCriteria as AC, classify as clf, SF_CRITICAL as sc,
    )
    assert AC.standard().classify(1.1).band == REVIEW
    assert clf(6.0).band == EXCLUDE
    assert sc == 1.0


def test_to_dict_is_the_canonical_json_serialisation():
    """welleng-pathfinder's A1 provenance stamps the criterion so a stored result
    records what "acceptable" meant when computed. `to_dict` is the blessed form so
    all four consumers stamp byte-identically instead of each hand-rolling asdict.
    """
    import dataclasses
    import json

    v = classify(1.1)
    d = v.to_dict()

    # JSON-clean, no round-trip surprises
    assert json.loads(json.dumps(d)) == d
    assert d["band"] == "review"
    assert d["criterion"]["sf_critical"] == 1.0        # recurses the nested criterion

    cd = AcceptanceCriteria.standard().to_dict()
    assert json.loads(json.dumps(cd)) == cd
    assert cd["operator_override"] is False

    # the stamp must cover EVERY dataclass field -- a new field cannot silently
    # fall out of the provenance record. This is the contract, pinned.
    for cls, obj in ((AcceptanceCriteria, AcceptanceCriteria.standard()),
                     (Verdict, v)):
        fields = {f.name for f in dataclasses.fields(cls)}
        assert set(obj.to_dict().keys()) == fields, (
            f"{cls.__name__}.to_dict must cover all dataclass fields; a new field "
            f"was added without updating to_dict: {fields ^ set(obj.to_dict().keys())}"
        )


def test_an_operator_override_is_recoverable_from_the_stamp():
    """The override must survive serialisation -- a stored result has to show an
    operator number was in force, not the standard's."""
    op = AcceptanceCriteria.standard().with_operator_floor(1.5)
    d = op.to_dict()
    assert d["operator_override"] is True
    assert d["sf_critical"] == 1.5
    assert "operator floor" in d["source"]
