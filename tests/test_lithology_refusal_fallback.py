"""A refusal that falls back to a plausible value is not a refusal.

`pattern_from_name` correctly declined to read "Grey Salt Clay Member" -- two
lithologies, picking one is a coin toss. Its caller then wrote

    pattern = pattern_from_name(name) or pattern

where `pattern` was the RGD GROUP pattern. In the Zechstein that is HALITE. So
the careful refusal did not leave the unit unlabelled; it labelled a clay member
as salt, inside the seal, on a barrier drawing.

Reported by a consumer as a scope divergence after it checked the ruling to drop
its own override table: 6 of 31 units in its store differed, all in the
direction of core drawing a confident wrong rock.
"""
import pytest

from welleng.lithology import (
    PatternMatch, pattern_from_name, pattern_match_from_name,
)


# --- the distinction the fallback turns on -------------------------------- #
@pytest.mark.parametrize("name,reason,fallback", [
    ("Ommelanden Formation", "no_lithology", True),    # says nothing -> group ok
    ("Grey Salt Clay Member", "ambiguous", False),     # salt AND clay
    ("Red Salt Clay Member", "ambiguous", False),
    ("Z1 Fringe Carbonate Member", "ambiguous", False),  # a class, not a rock
    ("Basal Anhydrite Member", "unmapped", False),     # no FGDC pattern exists
    ("Z4 Fringe Sandstone Member", "matched", False),
    ("", "no_name", False),
])
def test_the_reason_decides_whether_the_group_may_be_used(name, reason, fallback):
    m = pattern_match_from_name(name)
    assert m.reason == reason
    assert m.may_fall_back_to_group is fallback


def test_only_no_lithology_permits_the_group():
    """The whole point. Every other reason means the name has ACTIVELY
    contradicted the group, so substituting it is a claim, not a default."""
    for name in ("Grey Salt Clay Member", "Z1 Fringe Carbonate Member",
                 "Basal Anhydrite Member"):
        assert not pattern_match_from_name(name).may_fall_back_to_group


# --- the six divergences, by outcome -------------------------------------- #
HALITE, CHALK, MARL, SAND = 668, 626, 623, 607


@pytest.mark.parametrize("name", ["Grey Salt Clay Member", "Red Salt Clay Member",
                                  "Z1 Fringe Carbonate Member"])
def test_a_zechstein_member_is_never_drawn_as_salt_from_its_group(name):
    """The reported defect. Unpatterned is the correct answer: we know it is
    not halite, and the name does not let us say what it is."""
    m = pattern_match_from_name(name)
    assert m.code is None
    assert m.code != HALITE
    assert not m.may_fall_back_to_group


@pytest.mark.parametrize("name", ["Texel Greensand Member", "Holland Greensand Member"])
def test_greensand_is_sandstone_not_its_group(name):
    """Greensand is glauconitic SANDSTONE. Absent from the vocabulary it read
    as no lithology word at all, so it inherited chalk inside the Chalk Group
    and marl inside Rijnland."""
    m = pattern_match_from_name(name)
    assert m.code == SAND
    assert m.code not in (CHALK, MARL)


def test_marl_is_623():
    assert pattern_from_name("Plenus Marl Member") == 623


# --- the case the override was built for, still working ------------------- #
def test_the_original_motivating_case_is_unchanged():
    """A sandstone member inside the Zechstein must still escape the evaporite
    pattern -- that is why name-matching exists at all."""
    assert pattern_from_name("Z4 Fringe Sandstone Member") == SAND


def test_ambiguity_names_what_it_saw():
    """A refusal that does not say what it could not resolve moves the
    debugging cost to whoever has least context."""
    m = pattern_match_from_name("Grey Salt Clay Member")
    assert "clay" in m.words and "salt" in m.words


def test_the_old_signature_is_unchanged():
    """`pattern_from_name` is public and consumers call it; it keeps returning
    a bare code or None."""
    assert pattern_from_name("Z4 Fringe Sandstone Member") == SAND
    assert pattern_from_name("Grey Salt Clay Member") is None
    assert isinstance(pattern_match_from_name("x"), PatternMatch)


# --- rank: the group pattern describes the GROUP -------------------------- #
from welleng.exchange import rgd_nomenclature as rgd            # noqa: E402
from welleng.lithology import _GROUP_RANKS                      # noqa: E402


def _inherits_group(code):
    """Whether this unit would take its group's pattern under the current rule."""
    info = rgd.resolve(code) or {}
    name = info.get("name") or info.get("inherited_name") or ""
    m = pattern_match_from_name(name)
    return (m.code is None and m.may_fall_back_to_group
            and info.get("rank") in _GROUP_RANKS)


def test_a_formation_does_not_inherit_its_groups_lithology():
    """Reported by a consumer: the Lower Buntsandstein FORMATION inherited
    SANDSTONE from its group while its main member is claystone -- and that
    member is a caprock. A formation is narrower than the thing the group
    pattern describes."""
    assert (rgd.resolve("RBSH") or {}).get("rank") == "formation"
    assert not _inherits_group("RBSH")


def test_the_member_that_names_its_rock_still_resolves():
    """RBSHM is 'Claystone Member, Main' -- the name says it, so rank never
    enters into it."""
    assert pattern_from_name("Claystone Member, Main") == 620


def test_a_group_does_inherit_the_group_pattern():
    """The fallback is not removed, it is confined to the rank it describes."""
    assert (rgd.resolve("AT") or {}).get("rank") == "group"
    assert _inherits_group("AT")


def test_an_unknown_rank_refuses():
    """Not knowing what a unit IS is not a reason to assert what it is MADE OF.
    104 units in the RGD table carry no rank."""
    assert None not in _GROUP_RANKS
    assert "formation" not in _GROUP_RANKS and "member" not in _GROUP_RANKS


def test_most_units_reaching_the_fallback_were_never_entitled_to_it():
    """The measurement that justified the change: of the units that reached
    the group fallback, only a small minority are at group rank."""
    reaching, at_group_rank = 0, 0
    for code in rgd._units():
        info = rgd.resolve(code) or {}
        name = info.get("name") or info.get("inherited_name") or ""
        m = pattern_match_from_name(name)
        if m.code is None and m.may_fall_back_to_group:
            reaching += 1
            if info.get("rank") in _GROUP_RANKS:
                at_group_rank += 1
    assert reaching > 200
    assert at_group_rank < reaching * 0.15, (
        f"{at_group_rank}/{reaching} at group rank")
