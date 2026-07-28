"""SPE-187073 separation-factor acceptance criteria.

Promised open in core in the A2 ruling (2026-07-27) and consumed by welleng-api,
-probcol and -pathfinder, so the exact bands, the tighten-only override, and the
re-export location are all contract and are pinned here.
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
