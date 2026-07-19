"""Tests for the IPM (Instrument Performance Model) reader.

Uses a small synthetic model in the IPM format — no third-party/vendor error models.
"""
import textwrap

from welleng.exchange.ipm import read_ipm, loads_ipm, IPMModel, IPMTerm


# A minimal, made-up IPM model (a few ISCWSA-style terms) — not a vendor model.
SAMPLE = textwrap.dedent(
    """\
    #Tool Name  :TEST_MWD
    #ShortName  :TEST
    #Description:synthetic test model
    #Remarks    :for unit tests only
    #ToolGroup  :1
    #Name\tVector\tTie-On\tUnit\tValue\tFormula
    abx\ti\ts\t-\t0.004\t(-cos(inc)*sin(tfo))/gtot
    abz\ti\ts\t-\t0.004\t(-sin(inc))/gtot
    mbz\ta\ts\tnt\t70\t(-sin(inc)*sin(azm))/(mtot*cos(dip))
    dref\te\tr\t-\t0.35\t1.0
    dsf\te\ts\t-\t0.00024\ttmd
    decg\ta\tg\tdeg\t0.36\t1.0
    """
)


def test_loads_header_and_terms():
    m = loads_ipm(SAMPLE)
    assert isinstance(m, IPMModel)
    assert m.name == "TEST_MWD"
    assert m.short_name == "TEST"
    assert m.description == "synthetic test model"
    assert len(m.terms) == 6
    assert m.header["ToolGroup"] == "1"


def test_term_fields_parsed():
    m = loads_ipm(SAMPLE)
    abx = m.terms[0]
    assert isinstance(abx, IPMTerm)
    assert (abx.name, abx.vector, abx.tie_on, abx.unit) == ("abx", "i", "s", "-")
    assert abx.value == 0.004
    assert abx.formula == "(-cos(inc)*sin(tfo))/gtot"
    # a magnetic term keeps its unit + full formula
    mbz = [t for t in m.terms if t.name == "mbz"][0]
    assert mbz.unit == "nt" and mbz.value == 70.0
    assert mbz.formula == "(-sin(inc)*sin(azm))/(mtot*cos(dip))"


def test_sources_and_tie_on():
    m = loads_ipm(SAMPLE)
    assert m.sources() == ["abx", "abz", "decg", "dref", "dsf", "mbz"]
    assert {t.name for t in m.by_tie_on("s")} == {"abx", "abz", "mbz", "dsf"}
    assert [t.name for t in m.by_tie_on("r")] == ["dref"]
    assert [t.name for t in m.by_tie_on("g")] == ["decg"]


def test_to_dict_json_safe():
    import json
    m = loads_ipm(SAMPLE)
    d = m.to_dict()
    json.dumps(d)                       # must not raise
    assert d["terms"][0]["name"] == "abx"
    assert len(d["terms"]) == 6


def test_whitespace_delimited_fallback():
    # some exports use runs of spaces instead of tabs
    txt = (
        "#Tool Name  :SPACED\n"
        "#Name  Vector  Tie-On  Unit  Value  Formula\n"
        "abx    i       s       -     0.004  (-cos(inc)*sin(tfo))/gtot\n"
    )
    m = loads_ipm(txt)
    assert m.name == "SPACED"
    assert len(m.terms) == 1
    assert m.terms[0].name == "abx" and m.terms[0].value == 0.004


def test_read_ipm_from_file(tmp_path):
    p = tmp_path / "TEST.IPM"
    p.write_text(SAMPLE)
    m = read_ipm(p)
    assert m.name == "TEST_MWD" and len(m.terms) == 6
