"""Sodir FactPages reader -- parse-level tests, no network.

The network path is exercised by hand; these pin the behaviour that decides
whether an answer is right: date parsing that must not raise, a BOM that must
not corrupt the first column name, and an HTML error page that must not be
mistaken for a table.
"""
import datetime
import io

import pytest

from welleng.exchange.sodir import (SodirClient, SodirError, parse_date,
                                    wellbore_name)


class TestParseDate:
    def test_factpages_format(self):
        assert parse_date("10.07.2026") == datetime.datetime(2026, 7, 10)

    def test_iso_also_accepted(self):
        assert parse_date("2026-07-10") == datetime.datetime(2026, 7, 10)

    @pytest.mark.parametrize("value", ["", "   ", None, "not a date", "31.02.2026"])
    def test_absent_or_bad_returns_none_never_raises(self, value):
        # Most date columns are sparsely populated. An absent date is the normal
        # case; raising here would make a routine row an error.
        assert parse_date(value) is None

    def test_day_month_order_is_not_ambiguous(self):
        # 03.05 is 3 May, not 5 March. Getting this backwards silently shifts
        # every date that is valid under both readings.
        assert parse_date("03.05.2026") == datetime.datetime(2026, 5, 3)


class _FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, content_type: str = "text/csv; charset=utf-8"):
        super().__init__(payload)
        self.headers = {"Content-Type": content_type}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


class TestTableStreaming:
    def test_bom_does_not_corrupt_the_first_column_name(self, monkeypatch):
        # FactPages sends a UTF-8 BOM. Decoded as plain utf-8 it lands inside the
        # first column NAME, so row["wlbWellboreName"] raises KeyError while the
        # table looks perfectly fine.
        payload = "﻿wlbWellboreName,wlbStatus\r\n30/11-W-2 AH,P&A\r\n".encode()
        monkeypatch.setattr(
            SodirClient, "_open", lambda self, t: _FakeResponse(payload))
        rows = list(SodirClient().table("wellbore_development_all"))
        assert rows == [{"wlbWellboreName": "30/11-W-2 AH", "wlbStatus": "P&A"}]

    def test_rows_are_raw_strings_and_blanks_stay_blank(self, monkeypatch):
        # An unpopulated date must not become None, 0 or today. The caller
        # decides what absent means.
        payload = "﻿wlbWellboreName,wlbPluggedAbandonDate\r\nA,\r\n".encode()
        monkeypatch.setattr(
            SodirClient, "_open", lambda self, t: _FakeResponse(payload))
        row = next(iter(SodirClient().table("t")))
        assert row["wlbPluggedAbandonDate"] == ""

    def test_streams_rather_than_materialising(self, monkeypatch):
        payload = ("﻿a\r\n" + "".join(f"{i}\r\n" for i in range(1000))).encode()
        monkeypatch.setattr(
            SodirClient, "_open", lambda self, t: _FakeResponse(payload))
        it = SodirClient().table("t")
        assert next(it)["a"] == "0"          # first row available before the rest
        assert sum(1 for _ in it) == 999

    def test_html_error_page_is_refused_not_parsed(self, monkeypatch):
        # FactPages answers an unknown table with status 200 and an HTML error
        # page. Fed to a CSV reader that yields junk rows, so it must raise.
        monkeypatch.setattr(
            SodirClient, "_open",
            lambda self, t: (_ for _ in ()).throw(
                SodirError("expected CSV, got 'text/html'")),
        )
        with pytest.raises(SodirError, match="CSV"):
            list(SodirClient().table("does_not_exist"))

    def test_columns_reads_only_the_first_row(self, monkeypatch):
        payload = "﻿one,two\r\nx,y\r\nz,w\r\n".encode()
        monkeypatch.setattr(
            SodirClient, "_open", lambda self, t: _FakeResponse(payload))
        assert SodirClient().columns("t") == ["one", "two"]


class TestContentTypeGuard:
    def test_non_csv_content_type_raises(self, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda req, timeout=None: _FakeResponse(b"<html>", "text/html"),
        )
        with pytest.raises(SodirError, match="expected CSV"):
            SodirClient(retries=1)._open("bogus_table")

    def test_retries_then_raises_rather_than_returning_empty(self, monkeypatch):
        calls = []

        def boom(req, timeout=None):
            calls.append(1)
            raise OSError("read timed out")

        monkeypatch.setattr("urllib.request.urlopen", boom)
        monkeypatch.setattr("time.sleep", lambda s: None)
        with pytest.raises(SodirError, match="after 3 attempts"):
            SodirClient(retries=3)._open("wellbore_development_all")
        # A truncated or empty table is indistinguishable from a short one, so
        # the failure must surface rather than yield zero rows.
        assert len(calls) == 3


class TestNameColumn:
    """The wellbore name is in a different column depending on the table.

    Joining the wellbore tables to history/mud/core/dst on wlbWellboreName
    returns zero rows, which is indistinguishable from "this wellbore has no
    history". That happened; hence these.
    """

    def test_wellbore_tables_use_wlbWellboreName(self):
        assert wellbore_name({"wlbWellboreName": "35/2-U-8"}) == "35/2-U-8"

    def test_history_and_mud_use_wlbName(self):
        assert wellbore_name({"wlbName": "35/2-1"}) == "35/2-1"

    def test_falls_back_to_wlbWell(self):
        assert wellbore_name({"wlbWell": "35/2-1"}) == "35/2-1"

    def test_unrecognised_row_returns_none_rather_than_guessing(self):
        assert wellbore_name({"someOtherColumn": "35/2-1"}) is None

    def test_blank_name_is_not_a_name(self):
        row = {"wlbWellboreName": "   ", "wlbName": "35/2-1"}
        assert wellbore_name(row) == "35/2-1"

    def test_for_wellbores_matches_across_name_columns(self, monkeypatch):
        payload = (
            "﻿wlbName,wlbHistory\r\n"
            "35/2-1,spudded 2005\r\n"
            "35/2-9,unrelated\r\n"
        ).encode()
        monkeypatch.setattr(
            SodirClient, "_open", lambda self, t: _FakeResponse(payload))
        got = list(SodirClient().for_wellbores("wellbore_history", ["35/2-1"]))
        assert [r["wlbHistory"] for r in got] == ["spudded 2005"]
