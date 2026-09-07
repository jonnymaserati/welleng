"""Tests for the RGD lithostratigraphy nomenclature resolver.

Hierarchy is the verified deliverable (names are OCR-raw). Acceptance anchor: the
P11 De Ruyter codes welleng-drilling flagged must all resolve, and the prefix
pairs it could only flag "unresolved" must now classify as parent/child.
"""
import pytest

from welleng.exchange import rgd_nomenclature as rgd


def test_resolve_member_gives_parent_chain():
    u = rgd.resolve("KNGLU")
    assert u["rank"] == "member"
    assert u["parent"] == "KNGL"
    assert u["ancestors"] == ["KNGL", "KN"]      # member -> formation -> group


def test_resolve_ranks():
    assert rgd.resolve("KNGL")["rank"] == "formation"
    assert rgd.resolve("NU")["rank"] == "group"
    assert rgd.resolve("NU")["parent"] is None


def test_unknown_code_returns_none():
    assert rgd.resolve("ZZZZ") is None
    assert rgd.related("KNGL", "ZZZZ") == "unknown"


# -- the drilling blocker: prefix pairs that were only flaggable now resolve ----
def test_member_of_formation_is_ancestor_not_a_stacking_violation():
    # KNNCM is a member of KNNC (Vlieland Claystone) -- NOT an inversion
    assert rgd.related("KNNC", "KNNCM") == "ancestor"
    assert rgd.related("KNNCM", "KNNC") == "descendant"


def test_two_members_of_one_formation_are_siblings():
    assert rgd.related("KNGLU", "KNGLM") == "sibling"
    assert rgd.related("RBMVL", "RBMVU") == "sibling"


def test_unrelated_across_groups():
    assert rgd.related("KNGL", "ZEZ1") == "unrelated"


def test_common_ancestor_for_cross_surface_comparison():
    # comparing a formation top against one of its members: the surface family is
    # the formation
    assert rgd.common_ancestor("KNNC", "KNNCM") == "KNNC"
    assert rgd.common_ancestor("KNGLU", "KNGLM") == "KNGL"


@pytest.mark.parametrize("code,parent", [
    ("KNNC", "KN"), ("KNNCM", "KNNC"), ("NLLF", "NL"),
    ("RBSH", "RBS"), ("RBSHM", "RBSH"), ("RBSHN", "RBSH"), ("RBSHR", "RBSH"),
    ("RBMV", "RBM"), ("RBMVL", "RBMV"), ("RBMVC", "RBMV"), ("RBMVU", "RBMV"),
    ("RNSO", "RN"), ("RNSOB", "RNSO"), ("RNSOC", "RNSO"),
    ("ZEZ1", "ZE"), ("ZEZ1F", "ZEZ1"), ("ZEZ3", "ZE"), ("ZEZ3C", "ZEZ3"),
    ("CKGR", "CK"), ("CKTX", "CK"),
])
def test_p11_deruyter_codes_all_resolve(code, parent):
    u = rgd.resolve(code)
    assert u is not None, f"{code} not in nomenclature"
    assert u["parent"] == parent


def test_hierarchy_has_no_dangling_parents():
    units = rgd._units()
    for c, u in units.items():
        if u["parent"] is not None:
            assert u["parent"] in units, f"{c} -> dangling parent {u['parent']}"


def test_only_stratigraphic_codes_survive_the_cleanup():
    # non-lithostratigraphic tokens (organisation abbreviations, legend words)
    # were swept out of the scanned pages; the invariant that removed them is
    # "every real RGD code roots to a 2-letter group". Assert it holds — no code
    # roots to a non-group short token (welleng-projects 2026-09-07).
    units = rgd._units()
    groups = {c for c, u in units.items() if u["parent"] is None and len(c) == 2}

    def root(c):
        while units.get(c, {}).get("parent"):
            c = units[c]["parent"]
        return c
    for c in units:
        assert root(c) in groups, f"{c} does not root to a group (non-strat sweep-in?)"
    # a plausible non-code is simply absent
    assert rgd.resolve("QQ") is None and rgd.resolve("ZZZZ") is None


def test_ocr_names_repaired():
    # the scan read w as vv, and inserted spaces mid-word; repaired names must be
    # clean enough to print in a regulator-read P&A document
    assert rgd.resolve("KNNC")["name"] == "Vlieland Claystone Formation"
    assert rgd.resolve("RBSH")["name"] == "Lower Buntsandstein Formation"
    assert rgd.resolve("NL")["name"] == "Lower North Sea Group"
    assert "vv" not in (rgd.resolve("KNGLL")["name"] or "")


def test_provenance_flags_ocr_and_version():
    p = rgd.provenance()
    assert "1993" in p["source"]
    assert "OCR" in p["names"] or "unverified" in p["names"].lower()
