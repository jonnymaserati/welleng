"""Datum realisation chain — handling a platform/wellhead RE-SURVEY.

An installed platform's location is often re-surveyed later (better tech): the
origin every well hangs off MOVES, and the industry habit of overwriting the
wellhead coordinates destroys the audit trail and silently mixes wells computed
under different origin realisations. Pinned here:

- the APPEND-ONLY document chain (supersedes contract, provenance),
- shift composition along the chain (forward, backward, identity),
- re-referencing a position between realisations,
- the physics: a datum re-survey leaves ON-PLATFORM relative covariance
  INVARIANT (all wells move together) -- only absolute positions change,
- mixed-realisation comparisons WARN instead of passing silently,
- JSON round-trip carries the whole chain + each wellbore's pinned realisation.
"""
import numpy as np
import pytest

from welleng.hierarchy import (
    Datum, DatumRealisation, Well, Wellbore, WellNetwork,
)
from welleng.survey import Survey, SurveyHeader


def _chain():
    d = Datum(name="PlatformA-RKB")
    d.add_realisation(DatumRealisation(
        id="v1", date="1995-06-01", document="SURV-1995-014"))
    d.add_realisation(DatumRealisation(
        id="v2", date="2005-03-12", document="SURV-2005-002",
        shift=(2.0, -1.5, 0.1), radial_error=0.5, supersedes="v1"))
    d.add_realisation(DatumRealisation(
        id="v3", date="2020-09-30", document="GNSS-2020-77",
        shift=(-0.4, 0.3, 0.0), radial_error=0.1, supersedes="v2"))
    return d


def test_chain_is_append_only():
    d = _chain()
    assert [r.id for r in d.realisations] == ["v1", "v2", "v3"]
    assert d.current_realisation.id == "v3"
    # wrong supersedes -> rejected (no history rewrite)
    with pytest.raises(ValueError, match="must supersede the current head"):
        d.add_realisation(DatumRealisation(id="v4", supersedes="v1"))
    # duplicate id -> rejected
    with pytest.raises(ValueError, match="already in the chain"):
        d.add_realisation(DatumRealisation(id="v3", supersedes="v3"))
    # realisations are immutable links
    with pytest.raises(Exception):
        d.realisations[0].shift = (9, 9, 9)


def test_provenance_is_the_document_chain():
    d = _chain()
    assert d.provenance() == [
        ("v1", "1995-06-01", "SURV-1995-014"),
        ("v2", "2005-03-12", "SURV-2005-002"),
        ("v3", "2020-09-30", "GNSS-2020-77"),
    ]


def test_shift_composition():
    d = _chain()
    assert d.shift_between("v1", "v2") == (2.0, -1.5, 0.1)
    # composes: v1 -> v3 = (v1->v2) + (v2->v3)
    assert np.allclose(d.shift_between("v1", "v3"), (1.6, -1.2, 0.1))
    # identity + reversal
    assert d.shift_between("v2", "v2") == (0.0, 0.0, 0.0)
    assert np.allclose(d.shift_between("v3", "v1"),
                       tuple(-x for x in d.shift_between("v1", "v3")))


def test_rereference_a_position():
    d = _chain()
    pos_v1 = np.array([100.0, 200.0, 1500.0])          # quoted under v1
    pos_v3 = pos_v1 + np.array(d.shift_between("v1", "v3"))
    assert np.allclose(pos_v3, [101.6, 198.8, 1500.1])


def _net(pin_a="v1", pin_b="v1"):
    d = _chain()
    well = Well(id="W1", name="W1", datum=d)

    def mk(shift):
        md = np.linspace(0, 2400, 40)
        inc = np.clip(np.linspace(0, 80, 40), 0, 60)
        azi = (np.linspace(0, 90, 40) + shift) % 360
        return Survey(md=md, inc=inc, azi=azi, header=SurveyHeader(),
                      error_model="ISCWSA MWD Rev5.11")

    net = WellNetwork()
    top = Wellbore(id="T", parent=well, survey=mk(0.0),
                   datum_realisation=pin_a)
    lat_a = Wellbore(id="A", parent=top, kickoff_md=1200.0, survey=mk(6.0),
                     datum_realisation=pin_a)
    lat_b = Wellbore(id="B", parent=top, kickoff_md=1200.0, survey=mk(-6.0),
                     datum_realisation=pin_b)
    for wb in (top, lat_a, lat_b):
        net.add(wb)
    return net


def test_on_platform_relative_covariance_invariant_under_resurvey():
    # THE physics: a datum re-survey moves every well on the platform together,
    # so the relative covariance between them is bit-identical whether both are
    # referenced under v1 or both re-referenced under v3.
    C_v1 = _net("v1", "v1").relative_covariance("A", "B")
    C_v3 = _net("v3", "v3").relative_covariance("A", "B")
    assert np.array_equal(C_v1, C_v3)


def test_mixed_realisations_warn():
    net = _net("v1", "v3")           # B computed under the re-survey, A not
    with pytest.warns(UserWarning, match="different datum realisations"):
        net.relative_covariance("A", "B")


def test_json_round_trip_carries_chain_and_pins():
    net = _net("v1", "v3")
    rebuilt = WellNetwork.from_dict(net.to_dict())
    d = rebuilt.node("W1").datum
    assert [r.id for r in d.realisations] == ["v1", "v2", "v3"]
    assert d.realisation("v2").document == "SURV-2005-002"
    assert np.allclose(d.shift_between("v1", "v3"), (1.6, -1.2, 0.1))
    assert rebuilt.node("A").datum_realisation == "v1"
    assert rebuilt.node("B").datum_realisation == "v3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
