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

``wlbDrillingOperator`` also records the company name AS IT WAS AT DRILLING, so
one company appears under every name it has ever had. Equinor's lineage spans
nine strings -- Equinor Energy AS, Statoil Petroleum AS, Statoil ASA (old),
StatoilHydro Petroleum AS, StatoilHydro ASA, Norsk Hydro Produksjon AS, Norsk
Hydro Petroleum AS, Den norske stats oljeselskap a.s, Equinor Low Carbon
Solution AS -- covering 5138 wellbores, of which searching "equinor" alone finds
1092. **A name search misses 79% of them.**

``wlbField`` is EMPTY for a discovery that was never developed; the name is in
``wlbDiscovery`` instead. Every wellbore in block 35/2 has a blank
``wlbField`` while ``wlbDiscovery`` reads ``35/2-1 (Peon)``, so a search of
``wlbField`` for that name returns nothing.

⚠️ An unrecognised table name returns HTTP 500 here, but **500 is not evidence
that a table does not exist**: ``wellbore_formation_top`` is a documented live
table and 500s on this endpoint. The CSV export name and the internal table name
are not always the same, so this endpoint cannot be used to enumerate what
FactPages holds.

⚠️ FactPages publishes no directional survey, and neither does the FactMaps REST
API -- every wellbore layer there is point geometry. Deviation data for the
Norwegian shelf is in DISKOS, not in either public service.

🔴 The registry does not list every physical hole
-------------------------------------------------
Wellbore names follow the Directorate's designation guidelines (Resource
regulations §13). Items I-VIII of a name are **determined by the Directorate**;
**item IX is maintained by the OPERATOR** and does not appear here:

* item V ``A``, ``B`` ... -- a planned sidetrack. Gets its OWN registry row.
* item IX ``T2``, ``T3`` ... -- a **technical** sidetrack, i.e. one drilled to
  get past a problem. Stays under the PARENT wellbore's row. ``T2`` is the
  first such sidetrack, not ``T1``.

⇒ ``35/2-U-7 T2`` is a real hole in the ground with no row of its own, and no
query of this API can return it. **One registry row can be several wellbores**,
so a clearance or anti-collision scene built from these tables is a lower bound
on what is actually down there. Item III also reserves ``U`` for other wellbores
(soil drilling, shallow gas, pilot, scientific, stratigraphic) and ``T`` for a
test production wellbore, so ``A-Z`` as an installation letter excludes both.
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

#: The authoritative definition of every ``wlb*`` column. Read it before
#: inferring what a field means from its name -- several of the caveats in this
#: module's docstring were learned the expensive way instead.
ATTRIBUTE_REFERENCE = "https://factpages.sodir.no/en/wellbore/Attributes"

#: The other public service: an ArcGIS REST endpoint with ``where=`` queries,
#: relationship traversal and geometry. Carries attributes these CSV tables do
#: not -- notably ``wlbStatus`` on shallow wellbores, which distinguishes
#: ``JUNKED`` from ``P&A``. Capped at 1000 features per query; page on OBJECTID.
#: Every wellbore layer is POINT geometry, so it holds no well paths either.
FACTMAPS = ("https://factmaps.sodir.no/api/rest/services/Factmaps/"
            "FactMapsED50UTM32/MapServer")

#: The FactPages tables this module has been used against. Any other FactPages
#: table name works too -- :meth:`SodirClient.table` takes an arbitrary name.
TABLES = {
    "wellbore_all": "wellbore_all_long",
    "wellbore_development": "wellbore_development_all",
    "wellbore_exploration": "wellbore_exploration_all",
    "wellbore_shallow": "wellbore_other_all",
    "history": "wellbore_history",
    "mud": "wellbore_mud",
    "core": "wellbore_core",
    "dst": "wellbore_dst",
    "field": "field_description",
    "licence": "licence_licensee_hst",
}

#: The wellbore name is not in the same column in every table: the wellbore
#: tables use ``wlbWellboreName``, while ``wellbore_history``, ``wellbore_mud``,
#: ``wellbore_core`` and ``wellbore_dst`` use ``wlbName``. Joining on the wrong
#: one returns ZERO ROWS, which reads exactly like "this wellbore has no
#: history" -- use :func:`wellbore_name`.
NAME_COLUMNS = ("wlbWellboreName", "wlbName", "wlbWell")


def wellbore_name(row: dict) -> str | None:
    """The wellbore name from a row of ANY FactPages table.

    Returns ``None`` when the row carries no recognised name column, rather
    than guessing -- a silently wrong join is how a populated table reads as
    empty.
    """
    for key in NAME_COLUMNS:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return None


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

    def history(self) -> Iterator[dict]:
        """Stream the wellbore history table (``wlbHistory``, free HTML prose).

        Populated for exploration and development wellbores; a shallow wellbore
        typically has no row at all. Keyed on ``wlbName``, not
        ``wlbWellboreName`` -- see :data:`NAME_COLUMNS`.
        """
        return self.table(TABLES["history"])

    def mud(self) -> Iterator[dict]:
        """Stream the mud table -- weight, viscosity and yield point by depth."""
        return self.table(TABLES["mud"])

    def cores(self) -> Iterator[dict]:
        """Stream the core table."""
        return self.table(TABLES["core"])

    def dst(self) -> Iterator[dict]:
        """Stream the drill stem test table."""
        return self.table(TABLES["dst"])

    def for_wellbores(self, table: str, names) -> Iterator[dict]:
        """Stream the rows of ``table`` belonging to any of ``names``.

        Resolves the name column per table, so this works across the wellbore
        tables and the history/mud/core/dst tables without the caller tracking
        which one uses ``wlbName``.
        """
        wanted = {str(n).strip() for n in names}
        for row in self.table(table):
            if wellbore_name(row) in wanted:
                yield row

    def columns(self, table: str) -> list[str]:
        """The column names of a table, reading only its first row."""
        for row in self.table(table):
            return list(row)
        raise SodirError(f"{table}: no rows")
