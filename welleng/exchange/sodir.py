"""Reader for the Norwegian Offshore Directorate (Sodir) FactPages tables.

FactPages publishes the Norwegian continental shelf wellbore, field and licence
tables as CSV over plain HTTP -- no key, no token, no session. This module
streams them.

Streaming is the point. The tables are a few megabytes and most questions want a
handful of rows, so every accessor is a GENERATOR that yields one row at a time
and never materialises the table. ``wellbores_development()`` over 6253 rows
holds one row in memory::

    from welleng.exchange.sodir import SodirClient

    c = SodirClient()
    recent = [
        w for w in c.wellbores_development()
        if (d := parse_date(w["wlbPluggedAbandonDate"])) and d.year >= 2020
    ]

Rows are returned RAW -- the column names are Sodir's, the values are strings,
and interpretation belongs to the caller. An empty string means the field is not
populated; it is never coerced to 0, "" is not None, and nothing is guessed.

⚠️ Dates are ``dd.mm.yyyy``. Use :func:`parse_date`, which returns ``None``
rather than raising, because most date columns are sparsely populated.

Column notes that change answers
--------------------------------
``wlbDrillingOperator`` is the operator that DRILLED the wellbore, which for an
abandonment question is the wrong company: the wellbores plugged since 2020 were
drilled between 1974 and 2026, so this column names a driller from decades
earlier, not whoever plugged it. The current licensee is not in the wellbore
tables at all -- join ``wlbProductionLicence`` against the licence tables.

``wlbPluggedDate`` and ``wlbPluggedAbandonDate`` are DIFFERENT columns and are
populated independently (2909 and 413 rows respectively at the time of writing).
A wellbore can carry one and not the other.

``wlbReleasedDate`` is when the wellbore's data is released from confidentiality
and IS ROUTINELY IN THE FUTURE -- 294 rows carried a release date beyond today
when this module was written. A filter written as ``date <= today`` silently
drops them.
"""
from __future__ import annotations

import csv
import datetime
import io
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Iterator

__all__ = ["SodirClient", "SodirError", "parse_date", "TABLES"]

BASE = "https://factpages.sodir.no/public"

#: The FactPages tables this module has been used against. Any other FactPages
#: table name works too -- :meth:`SodirClient.table` takes an arbitrary name.
TABLES = {
    "wellbore_all": "wellbore_all_long",
    "wellbore_development": "wellbore_development_all",
    "wellbore_exploration": "wellbore_exploration_all",
    "wellbore_shallow": "wellbore_other_all",
    "field": "field_description",
    "licence": "licence_licensee_hst",
}


class SodirError(RuntimeError):
    """A FactPages request failed, or returned something that is not a table."""


def parse_date(value: str | None) -> datetime.datetime | None:
    """Parse a FactPages date (``dd.mm.yyyy``), or return ``None``.

    Returns ``None`` for an empty or unparseable value rather than raising: most
    date columns are sparsely populated, and an absent date is the normal case,
    not an error. The caller must distinguish "no date recorded" from "date is
    old" -- this function does not.
    """
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


class SodirClient:
    """Streams FactPages tables as dict rows.

    Parameters
    ----------
    timeout: float
        Per-request socket timeout in seconds.
    retries: int
        Attempts per request. A read timeout on a multi-megabyte table is
        common; each retry backs off linearly.
    user_agent: str
        Sent on every request so the archive can attribute the traffic.
    """

    def __init__(
        self,
        timeout: float = 60.0,
        retries: int = 3,
        user_agent: str = "welleng-sodir/1.0",
    ):
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.user_agent = user_agent

    def _url(self, table: str) -> str:
        query = urllib.parse.urlencode(
            {"rs:Format": "CSV", "Top100": "false"}, safe=":"
        )
        return f"{BASE}?/Factpages/external/tableview/{table}&{query}"

    def _open(self, table: str):
        """Open the CSV response, retrying on transport failure.

        Raises rather than returning a partial or empty table: a truncated CSV
        parses cleanly into too few rows, which is indistinguishable from a
        table that is genuinely short.
        """
        url = self._url(table)
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = urllib.request.urlopen(req, timeout=self.timeout)
            except (urllib.error.URLError, OSError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(2.0 * attempt)
                continue
            ctype = response.headers.get("Content-Type", "")
            if "csv" not in ctype.lower():
                response.close()
                raise SodirError(
                    f"{table}: expected CSV, got {ctype!r}. FactPages serves an "
                    "HTML error page with status 200 when a table name is wrong, "
                    "so this is most likely an unknown table."
                )
            return response
        raise SodirError(
            f"{table}: request failed after {self.retries} attempts: {last}")

    def table(self, table: str) -> Iterator[dict]:
        """Yield each row of a FactPages table as a ``dict`` of raw strings.

        ``table`` is the FactPages table name, e.g. ``wellbore_development_all``.
        The names in :data:`TABLES` are the ones this module has been used
        against; any other FactPages table name is accepted.
        """
        with self._open(table) as response:
            # utf-8-sig: the CSV carries a BOM, which otherwise lands in the
            # first column NAME and makes that column unreachable by key.
            stream = io.TextIOWrapper(response, encoding="utf-8-sig", newline="")
            yield from csv.DictReader(stream)

    def wellbores_development(self) -> Iterator[dict]:
        """Stream the development wellbore table.

        Carries ``wlbPluggedDate`` and ``wlbPluggedAbandonDate`` -- see the
        module docstring, they are different columns.
        """
        return self.table(TABLES["wellbore_development"])

    def wellbores_exploration(self) -> Iterator[dict]:
        """Stream the exploration wellbore table."""
        return self.table(TABLES["wellbore_exploration"])

    def wellbores_shallow(self) -> Iterator[dict]:
        """Stream the shallow wellbore table.

        Holds the site-investigation and shallow boreholes -- purposes SOIL
        DRILLING, SHALLOW GAS, SCIENTIFIC, STRATIGRAPHIC. These are named
        ``<block>-U-<n>`` and are NOT in the development or exploration tables,
        so a name search over those two alone reports a shallow wellbore as
        absent.
        """
        return self.table(TABLES["wellbore_shallow"])

    def wellbores(self) -> Iterator[dict]:
        """Stream every wellbore -- development, exploration and shallow.

        One table, 95 columns, so a search here cannot miss a wellbore by
        looking in the wrong category.
        """
        return self.table(TABLES["wellbore_all"])

    def columns(self, table: str) -> list[str]:
        """The column names of a table, reading only its first row."""
        for row in self.table(table):
            return list(row)
        raise SodirError(f"{table}: no rows")
