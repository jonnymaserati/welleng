"""Well hierarchy + wellbore network graph — the container surveys hang on.

This models the master-data hierarchy that a survey (and its error/clearance
results) attaches to, and the **wellbore network graph** used to propagate
*relative* position uncertainty correctly between wellbores that share ancestry.

OSDU schema source (the canonical Well-Known-Schemas this maps to):
``https://community.opengroup.org/osdu/data/data-definitions`` — WKS JSON under
``Generated/master-data/`` (e.g. ``Wellbore.1.x.0.json``, ``Well``, ``Field``,
``Organisation``) and entity-relationship docs under ``E-R/master-data/``. Design
+ mapping + the graph relative-error model are documented in
``docs/dev/HIERARCHY_FRAMEWORK.md`` (this framework is also a planned paper).

Hierarchy (maps to OSDU master-data — see that doc for the field-level mapping,
grounded separately):

    Organisation -> Field -> Site(WellSiteStructure) -> Well(slot + Datum) -> Wellbore* -> Survey

The **wellbore graph** is a *forest*: every wellbore section has a parent — a
parent wellbore (a sidetrack/lateral kicked off at ``kickoff_md``) or, for a
*root* wellbore, the **Well** (its surface location). Wells on one **Site**
share the site's geodetic CRS + convergence (common systematic -> cancels in
relative use); each Well carries its own slot position (+ ``slot_radial_error``)
and local datum (RKB). EDM cross-check confirms the spine
``CD_PROJECT(CRS) -> CD_SITE -> CD_WELL(slot,datum) -> CD_WELLBORE(parent_wellbore_id) -> survey``.

Why a graph (the load-bearing reason). For relative error between two
wellbores (e.g. two laterals off one parent, or two sidetracks) you must NOT
sum their full independent uncertainties — the survey they share up to the
divergence point carries the SAME systematic errors, which cancel in the
relative sense. Walking to the **lowest common ancestor (LCA)** splits the two
paths into a shared trunk (systematic-common) and two divergent branches
(independent); only the divergent parts sum. This is the correct multilateral /
sidetrack relative-uncertainty treatment and the basis for anti-collision
between wells of common ancestry (consumed by probcol).

Status: SCAFFOLD (2026-07-05). Graph + roots/leaves + LCA + the shared/divergent
split are implemented; the per-section covariance combination is the integration
point with the error engine (SurveyComposition, FUTURE_WORK) and is stubbed with
a documented interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Optional

import networkx as nx


# --------------------------------------------------------------------------- #
# master-data hierarchy (OSDU-aligned; exact kind/field names wired separately)
# --------------------------------------------------------------------------- #
@dataclass
class Datum:
    """A local spatial / depth datum (e.g. a platform RKB / wellhead reference).

    Represents the vertical reference a well's measured depths are quoted
    against (rotary kelly bushing, rotary table, wellhead, or mean sea level).
    A datum is attached per-Well but is typically physically shared by every
    wellbore drilled from one platform.

    Parameters
    ----------
    name : str
        Human-readable datum name / identifier.
    elevation : float, default 0.0
        Datum elevation, in metres, above mean sea level (or the field
        reference given by ``reference``).
    reference : str, default "MSL"
        The elevation reference frame — one of ``"MSL"``, ``"RKB"``, ``"RT"``,
        ``"wellhead"``, etc.

    Notes
    -----
    Because the datum is shared by every wellbore on a platform, the datum's
    own position error is a *common* systematic term between those wellbores
    and therefore cancels in relative (wellbore-to-wellbore) uncertainty use.
    """
    name: str
    elevation: float = 0.0                 # above MSL (or the field reference)
    reference: str = "MSL"                 # MSL | RKB | RT | wellhead | ...


@dataclass
class _Node:
    """Base for the master-data entities: an id, a human name, a parent ref.

    Parameters
    ----------
    id : str
        Unique node identifier (used as the graph key and for parent wiring).
    name : str, default ""
        Human-readable entity name.
    parent : _Node or None, default None
        The containing / kick-off parent entity, or ``None`` for a top node.
    """
    id: str
    name: str = ""
    parent: Optional["_Node"] = None


@dataclass
class Organisation(_Node):
    """Operating organisation — the top of the master-data hierarchy.

    Maps to OSDU ``master-data--Organisation``. Inherits the :class:`_Node`
    fields (``id``, ``name``, ``parent``); an Organisation normally has no
    parent.
    """


@dataclass
class Field(_Node):
    """A field / asset grouping the sites of a development.

    Maps to OSDU ``master-data--Field``. Inherits the :class:`_Node` fields;
    ``parent`` points to the owning :class:`Organisation`.
    """


@dataclass
class Site(_Node):
    """A site / surface structure grouping wells — a platform, pad, or subsea
    template.

    Maps to OSDU ``master-data--WellSiteStructure`` (EDM ``CD_SITE``). Carries
    the shared spatial reference for the wells drilled from it, so that the
    common geodetic terms can be identified and cancelled in relative use.

    Parameters
    ----------
    id : str
        Unique node identifier (inherited from :class:`_Node`).
    name : str, default ""
        Human-readable site name (inherited from :class:`_Node`).
    parent : _Node or None, default None
        The owning :class:`Field` (inherited from :class:`_Node`).
    crs : str or None, default None
        Geodetic coordinate reference system (zone / datum, e.g. ``"UTM-31N"``
        / ``"ED50"``) shared by every well on the site (EDM ``CD_PROJECT``).
    convergence : float or None, default None
        Grid-vs-true-north convergence, in radians, for the site origin.
    is_field_centre : bool, default False
        ``True`` if this site is the field-centre coordinate origin.
    location : tuple of float or None, default None
        Site map location as ``(northing, easting)``, in metres.

    Notes
    -----
    Wells on one site share ``crs`` and ``convergence`` — these are *common*
    systematic terms that cancel in the relative (wellbore-to-wellbore)
    uncertainty sense. Only the per-well slot offsets differ (see
    :class:`Well`).
    """
    crs: Optional[str] = None               # geodetic CRS (zone/datum), shared reference
    convergence: Optional[float] = None      # grid vs true north (rad)
    is_field_centre: bool = False
    location: Optional[tuple[float, float]] = None   # (northing, easting) map location


@dataclass
class Well(_Node):
    """A well — the surface location / wellhead the root wellbore(s) hang off.

    Maps to OSDU ``master-data--Well`` (EDM ``CD_WELL``). Holds the slot
    position and its uncertainty (the surface-location error source in relative
    uncertainty) together with the well's local depth datum.

    Parameters
    ----------
    id : str
        Unique node identifier (inherited from :class:`_Node`).
    name : str, default ""
        Human-readable well name (inherited from :class:`_Node`).
    parent : _Node or None, default None
        The owning :class:`Site` (inherited from :class:`_Node`).
    slot : tuple of float or None, default None
        Slot offset ``(ns, ew)`` from the site origin, in metres.
    slot_radial_error : float, default 0.0
        Radial (1-sigma) slot-position uncertainty, in metres.
    wellhead_depth : float or None, default None
        Wellhead depth, in metres.
    datum : Datum or None, default None
        The per-well RKB / rotary depth datum (EDM ``CD_DATUM`` is per-well,
        not per-site).

    Notes
    -----
    Two wells on one site share the site CRS, so in relative uncertainty only
    their slot offsets and ``slot_radial_error`` differ — the shared CRS /
    convergence cancels (see :class:`Site`).
    """
    slot: Optional[tuple[float, float]] = None   # (ns, ew) slot offset from the site origin
    slot_radial_error: float = 0.0               # slot-position uncertainty
    wellhead_depth: Optional[float] = None
    datum: Optional[Datum] = None                # per-well RKB/rotary datum


@dataclass
class Wellbore(_Node):
    """A wellbore *section* — a drilled hole.

    Maps to OSDU ``master-data--Wellbore`` (EDM ``CD_WELLBORE``). A wellbore is
    the unit the graph is built from: it is either a *root* wellbore (top hole,
    parented by a :class:`Well`) or a sidetrack / lateral kicked off another
    :class:`Wellbore`.

    Parameters
    ----------
    id : str
        Unique node identifier (inherited from :class:`_Node`).
    name : str, default ""
        Human-readable wellbore name (inherited from :class:`_Node`).
    parent : _Node or None, default None
        Either the :class:`Well` (root wellbore) or the :class:`Wellbore` this
        section kicked off from (sidetrack / lateral). The OSDU edge is
        ``Wellbore.KickOffWellbore`` / the EDM ``parent_wellbore_id``.
    kickoff_md : float or None, default None
        The tie / divergence measured depth on the parent, in metres — the
        point at which this section departs its parent. See Notes.
    survey : object, default None
        The section's survey (a ``welleng.survey.Survey`` / future MinCurve
        result; OSDU ``work-product-component--WellboreTrajectory``), covering
        ``[tie_on_md, td_md]``.
    survey_date : str or None, default None
        Survey acquisition date. An error-source correlation key: geomagnetic
        terms (declination / dip / B) are date + location dependent via secular
        variation.
    tool_id : str or None, default None
        Survey tool run identifier. An error-source correlation key: sections
        from the same tool run share that tool's systematic error.
    geomag_model : str or None, default None
        Geomagnetic model / IFR / IIFR reference used. An error-source
        correlation key.

    Notes
    -----
    ``kickoff_md`` is **derived**, not a native OSDU field — OSDU has no
    ``KickOffMD``. Infer it from the child trajectory's top MD relative to the
    parent (EDM supplies it directly via ``CD_SURVEY_HEADER.tie_on_depth``). It
    is the divergence point the relative-error split needs.

    The ``survey_date`` / ``tool_id`` / ``geomag_model`` fields are the
    error-source *context keys*. They decide how much two sections' errors
    correlate even when their trajectories are independent (divergent
    branches): sections sharing a key share that systematic error *source*, so
    it partly correlates between them — orthogonal to the graph ancestry. The
    datum and grid convergence come from the parent Well / Site and are shared
    keys too.
    """
    parent: Optional["_Node"] = None       # Well (root) or Wellbore (sidetrack)
    kickoff_md: Optional[float] = None      # DERIVED tie/divergence MD on the parent
    survey: object = None                   # welleng Survey / MinCurve result
    # --- error-source context (the correlation keys) ---------------------- #
    # These decide how much two sections' errors correlate even when their
    # *trajectories* are independent (divergent branches). Sections that share a
    # key share that systematic error SOURCE -> it partly correlates between
    # them (reduces relative independence), orthogonal to the graph ancestry.
    survey_date: Optional[str] = None       # acquisition date -> geomag (declination/dip/B)
                                            #   is date+location dependent (secular variation)
    tool_id: Optional[str] = None           # same tool RUN -> shared tool systematic
    geomag_model: Optional[str] = None      # geomag model/IFR/IIFR reference used
    # (datum + grid convergence come from the parent Well/Site; also shared keys)


# --------------------------------------------------------------------------- #
# the wellbore network graph
# --------------------------------------------------------------------------- #
class WellNetwork:
    """A forest of wellbore sections rooted at wells / platforms.

    Wraps a ``networkx.DiGraph`` whose nodes are the master-data entities
    (Wellbores and their Well / Site / Field / Organisation ancestors) and
    whose directed edges ``parent -> child`` encode "child kicks off / hangs
    off parent". Provides the ancestry queries (roots, leaves, lowest common
    ancestor) and the shared / divergent split that relative-error propagation
    needs, plus native JSON persistence.

    Notes
    -----
    The graph is a *forest*: every wellbore has a parent — a parent wellbore
    (sidetrack / lateral) or, for a root wellbore, its Well. The load-bearing
    reason for a graph is relative uncertainty: two wellbores of common
    ancestry share the same survey up to their divergence point, so those
    systematic errors are common and cancel in the relative sense. Walking to
    the lowest common ancestor splits the two paths into a shared trunk
    (systematic-common) and two independent divergent branches; only the
    divergent parts sum. See :meth:`relative_covariance`.

    Examples
    --------
    Build a small Site -> Well -> Wellbore tree with two laterals off one top
    hole, then query its topology.

    >>> from welleng.hierarchy import Site, Well, Wellbore, WellNetwork
    >>> net = WellNetwork()
    >>> site = Site(id='S1', name='PadA')
    >>> well = Well(id='W1', name='W1', parent=site)
    >>> top = Wellbore(id='WB1', name='TopHole', parent=well)
    >>> lat1 = Wellbore(id='WB2', name='Lat1', parent=top, kickoff_md=1000.0)
    >>> lat2 = Wellbore(id='WB3', name='Lat2', parent=top, kickoff_md=1200.0)
    >>> for n in (site, well, top, lat1, lat2):
    ...     _ = net.add(n)
    >>> [w.id for w in net.roots()]
    ['WB1']
    >>> sorted(w.id for w in net.leaves())
    ['WB2', 'WB3']
    >>> net.lowest_common_ancestor('WB2', 'WB3')
    'WB1'
    >>> net.shared_and_divergent('WB2', 'WB3')
    (['WB1', 'W1'], ['WB2'], ['WB3'])
    """

    def __init__(self) -> None:
        self._g: nx.DiGraph = nx.DiGraph()      # parent -> child
        self._nodes: dict[str, _Node] = {}

    # -- construction ------------------------------------------------------- #
    def add(self, node: _Node) -> _Node:
        """Add a node and wire its parent edge into the graph.

        Registers ``node`` in the network and, when its ``parent`` is a
        :class:`Well` (root wellbore) or a :class:`Wellbore` (sidetrack), adds
        the directed ``parent -> node`` edge. The full container ancestor chain
        (Well -> Site -> Field -> Organisation) is also registered so the whole
        hierarchy serialises and is queryable.

        Parameters
        ----------
        node : _Node
            The entity to add — typically a :class:`Wellbore` or :class:`Well`
            (container ancestors are pulled in automatically via ``parent``).

        Returns
        -------
        _Node
            The same ``node`` that was passed in (for chaining).

        Examples
        --------
        >>> from welleng.hierarchy import Well, Wellbore, WellNetwork
        >>> net = WellNetwork()
        >>> well = Well(id='W1', name='W1')
        >>> _ = net.add(Wellbore(id='WB1', name='TopHole', parent=well))
        >>> [w.id for w in net.roots()]
        ['WB1']
        """
        self._nodes[node.id] = node
        self._g.add_node(node.id)
        parent = node.parent
        if isinstance(parent, (Well, Wellbore)):
            self._nodes.setdefault(parent.id, parent)
            self._g.add_node(parent.id)
            self._g.add_edge(parent.id, node.id)
        # register the full container ancestor chain (Well -> Site -> Field ->
        # Organisation) in _nodes so the whole hierarchy serialises + is queryable.
        p = parent
        while p is not None:
            self._nodes.setdefault(p.id, p)
            p = getattr(p, "parent", None)
        return node

    def node(self, id_: str) -> _Node:
        """Return the registered entity for a node id.

        Parameters
        ----------
        id_ : str
            The node identifier.

        Returns
        -------
        _Node
            The stored entity.

        Raises
        ------
        KeyError
            If no node with ``id_`` has been added.
        """
        return self._nodes[id_]

    # -- topology ----------------------------------------------------------- #
    def roots(self) -> list[Wellbore]:
        """Return the root wellbores of the forest.

        A root wellbore is one whose parent is a :class:`Well` rather than
        another wellbore — i.e. the top hole of a drilled tree, with no parent
        wellbore.

        Returns
        -------
        list of Wellbore
            The root wellbores, in insertion order.

        Examples
        --------
        >>> from welleng.hierarchy import Well, Wellbore, WellNetwork
        >>> net = WellNetwork()
        >>> well = Well(id='W1', name='W1')
        >>> _ = net.add(Wellbore(id='WB1', name='TopHole', parent=well))
        >>> [w.id for w in net.roots()]
        ['WB1']
        """
        return [
            n for n in self._wellbores()
            if isinstance(self._nodes.get(self._parent_id(n.id)), Well)
        ]

    def leaves(self) -> list[Wellbore]:
        """Return the leaf wellbores of the forest.

        A leaf wellbore has no children — a TD section or the deepest lateral
        on its branch.

        Returns
        -------
        list of Wellbore
            The childless wellbores.

        Examples
        --------
        >>> from welleng.hierarchy import Well, Wellbore, WellNetwork
        >>> net = WellNetwork()
        >>> well = Well(id='W1', name='W1')
        >>> top = Wellbore(id='WB1', name='TopHole', parent=well)
        >>> lat = Wellbore(id='WB2', name='Lat1', parent=top, kickoff_md=1000.0)
        >>> for n in (well, top, lat):
        ...     _ = net.add(n)
        >>> [w.id for w in net.leaves()]
        ['WB2']
        """
        return [
            n for n in self._wellbores()
            if self._g.out_degree(n.id) == 0
        ]

    def ancestors(self, id_: str) -> list[str]:
        """Return the ancestry chain of a node, nearest-first.

        Walks parent edges from ``id_`` up to (and including) its root, so the
        returned list starts with ``id_`` itself and ends at the top container
        entity on that path.

        Parameters
        ----------
        id_ : str
            The node identifier to start from.

        Returns
        -------
        list of str
            Node ids from ``id_`` up to and including the root, nearest-first.
        """
        chain, cur = [], id_
        seen: set[str] = set()
        while cur is not None and cur not in seen:
            seen.add(cur); chain.append(cur)
            cur = self._parent_id(cur)
        return chain

    def lowest_common_ancestor(self, a: str, b: str) -> Optional[str]:
        """Return the lowest common ancestor (LCA) of two nodes.

        The LCA is the deepest node that is an ancestor of both ``a`` and ``b``
        — their divergence point in the forest.

        Parameters
        ----------
        a : str
            First node identifier.
        b : str
            Second node identifier.

        Returns
        -------
        str or None
            The id of the deepest shared ancestor, or ``None`` if the two nodes
            share no ancestry (e.g. different platforms).

        Examples
        --------
        >>> from welleng.hierarchy import Well, Wellbore, WellNetwork
        >>> net = WellNetwork()
        >>> well = Well(id='W1', name='W1')
        >>> top = Wellbore(id='WB1', name='TopHole', parent=well)
        >>> lat1 = Wellbore(id='WB2', name='Lat1', parent=top, kickoff_md=1000.0)
        >>> lat2 = Wellbore(id='WB3', name='Lat2', parent=top, kickoff_md=1200.0)
        >>> for n in (well, top, lat1, lat2):
        ...     _ = net.add(n)
        >>> net.lowest_common_ancestor('WB2', 'WB3')
        'WB1'
        """
        anc_a = self.ancestors(a)
        anc_b = set(self.ancestors(b))
        for node in anc_a:               # nearest-first from a
            if node in anc_b:
                return node
        return None

    def shared_and_divergent(self, a: str, b: str) -> tuple[list[str], list[str], list[str]]:
        """Split the two ancestry paths at their lowest common ancestor.

        Partitions the ancestry of ``a`` and ``b`` into the shared trunk (the
        LCA and everything above it) and the two divergent branches below the
        LCA — the partition relative-error propagation is built on.

        Parameters
        ----------
        a : str
            First node identifier.
        b : str
            Second node identifier.

        Returns
        -------
        tuple of (list of str, list of str, list of str)
            ``(shared, branch_a, branch_b)`` where ``shared`` is the common
            trunk (LCA and above), ``branch_a`` is the divergent portion of
            ``a``'s ancestry below the LCA (nearest-first), and ``branch_b`` the
            same for ``b``. If the two nodes share no ancestry, ``shared`` is
            empty and each branch is the node's full ancestry.

        Notes
        -----
        Systematic errors on the shared trunk are COMMON to both wellbores and
        cancel in the relative sense; only the independent divergent branches
        sum. See :meth:`relative_covariance`.

        Examples
        --------
        >>> from welleng.hierarchy import Well, Wellbore, WellNetwork
        >>> net = WellNetwork()
        >>> well = Well(id='W1', name='W1')
        >>> top = Wellbore(id='WB1', name='TopHole', parent=well)
        >>> lat1 = Wellbore(id='WB2', name='Lat1', parent=top, kickoff_md=1000.0)
        >>> lat2 = Wellbore(id='WB3', name='Lat2', parent=top, kickoff_md=1200.0)
        >>> for n in (well, top, lat1, lat2):
        ...     _ = net.add(n)
        >>> net.shared_and_divergent('WB2', 'WB3')
        (['WB1', 'W1'], ['WB2'], ['WB3'])
        """
        lca = self.lowest_common_ancestor(a, b)
        anc_a = self.ancestors(a)
        anc_b = self.ancestors(b)
        if lca is None:
            return [], anc_a, anc_b          # no shared trunk -> fully independent
        branch_a = anc_a[: anc_a.index(lca)]
        branch_b = anc_b[: anc_b.index(lca)]
        shared = anc_a[anc_a.index(lca):]
        return shared, branch_a, branch_b

    # -- relative uncertainty (integration point with the error engine) ----- #
    def relative_covariance(
        self, a: str, b: str,
        md_a: Optional[float] = None, md_b: Optional[float] = None,
        error_model: str = "ISCWSA MWD Rev5.11",
        share_mode: Optional[str] = None,
    ):
        """Relative position covariance between two wellbores of common
        ancestry.

        Computes the NEV covariance of the *difference* in position between the
        compared points of wellbores ``a`` and ``b`` (their ends, or their
        point of closest approach — the anti-collision use), correctly
        cancelling the systematic errors carried by the survey they share up to
        their divergence point.

        Parameters
        ----------
        a : str
            First wellbore node identifier.
        b : str
            Second wellbore node identifier.

        Returns
        -------
        numpy.ndarray
            The 3x3 relative-position covariance matrix in the NEV frame, in
            metres squared.

        Raises
        ------
        ValueError
            If either wellbore has no survey attached (raised by the underlying
            :meth:`_abs_cov` helper).

        Notes
        -----
        The result is NOT the naive independent sum ``Cov(a) + Cov(b)``. There
        are two correlation dimensions:

        1. **Trajectory ancestry (the graph).** The *shared trunk* is the same
           physical survey inherited by both wellbores, so ALL of its error
           terms (systematic and the once-realised random) are common and
           CANCEL in the difference — net shared-trunk contribution is ~0. Only
           the *divergent branches* have independent trajectories.
        2. **Error-source context (the keys).** Even the divergent branches are
           not fully independent: where they share a source (``tool_id`` = same
           tool run, ``survey_date`` + location = same geomagnetic
           declination / dip / B, ``geomag_model`` / IFR, the Well ``datum`` /
           Site convergence) the *systematic* part of that term correlates
           between them, so::

               Cov_rel ~ Cov(branch_a) + Cov(branch_b)
                         - 2 * Cov_shared_source(branch_a, branch_b)

           with the shared trunk dropped. Drop the cross term (treat as
           independent) only when the branches share no tool / geomag / datum
           context. This is the multilateral / sidetrack relative-uncertainty
           rule.

        This implements the ISCWSA Side-track Clearance RP (2022) §3.2.2
        method (b) — "subtract the covariance at the Side-track point" — which
        is Williamson (2000, SPE-67616-PA) Eq. A-24 specialised to a
        fully-correlated shared trunk: with ``C_A = C_st + C_branchA`` and
        ``C_B = C_st + C_branchB``::

            C_rel = C_A(a) + C_B(b) - 2 * C_st   ( = C_branchA + C_branchB )

        where ``C_st`` is the absolute covariance at the deepest common
        (side-track) point. Each wellbore's absolute covariance comes from
        welleng's ``ErrorModel`` (``survey.cov_nev``); the surveys are assumed
        full-from-surface (as EDM definitive surveys are), so they agree over
        the shared trunk.

        Implemented is the trunk cancellation (the dominant effect).
        **Deferred:** the source-context cross-term
        ``- 2 * Cov_shared_source`` between the *divergent* branches (same tool
        run / geomag date / datum — the RP's same-job / different-job
        correlation table, McGregor partial correlation). Without it two
        same-job laterals are treated as independent below the kickoff, which is
        slightly *conservative* (over-states relative uncertainty) — the safe
        direction. Add it with the per-term systematic / random classification
        from the error engine.
        """
        import numpy as np
        C_a = self._abs_cov(self.node(a), md_a, error_model)
        C_b = self._abs_cov(self.node(b), md_b, error_model)
        lca = self.lowest_common_ancestor(a, b)
        if lca is None:                          # no shared ancestry -> independent
            return C_a + C_b
        st_md = self._sidetrack_md(a, b)
        if st_md is None:
            # No kick-off MD on the divergent branch(es) -> can't locate the
            # shared trunk; fall back to the naive independent sum (no
            # cancellation), which is conservative (over-states relative).
            import warnings
            warnings.warn(
                "relative_covariance: divergence (kick-off) MD unavailable; "
                "returning the naive independent sum (no shared-trunk "
                "cancellation)", stacklevel=2,
            )
            return C_a + C_b
        # Covariance at the side-track (divergence) point on the shared trunk.
        # Both wellbores share the trunk (full-from-surface surveys), so evaluate
        # it on a's survey.
        C_st = self._abs_cov(self.node(a), st_md, error_model)
        if share_mode is None or share_mode == "all_independent":
            # Shared trunk cancels; divergent branches treated as independent.
            return C_a + C_b - 2.0 * C_st
        # Per-term source-context sharing between the DIVERGENT branches (same
        # tool run / geomag era). Subtract the trunk to get each branch's own
        # covariance components, then combine via welleng.conditioning under the
        # requested share_mode (RP same-job/different-job table).
        from .conditioning import combine_covariances
        _, Ga, Sa = self._abs_cov_components(self.node(a), md_a, error_model)
        _, Gb, Sb = self._abs_cov_components(self.node(b), md_b, error_model)
        _, Gst, Sst = self._abs_cov_components(self.node(a), st_md, error_model)
        br_a, br_b = C_a - C_st, C_b - C_st          # divergent-branch totals
        res = combine_covariances(
            br_a[None], br_b[None],
            cov_global_a=(Ga - Gst)[None], cov_global_b=(Gb - Gst)[None],
            cov_systematic_a=(Sa - Sst)[None], cov_systematic_b=(Sb - Sst)[None],
            share_mode=share_mode,
        )
        return np.asarray(res.cov_combined)[0]

    # -- covariance helpers ------------------------------------------------- #
    def _abs_cov(self, wb: "Wellbore", md, error_model: str):
        """Absolute (from-surface) NEV covariance of a wellbore at a depth.

        Returns the wellbore's own absolute position covariance at the nearest
        survey station to ``md`` (or at TD when ``md`` is ``None``), using the
        survey's cached ``cov_nev`` if present, otherwise computing it via
        welleng's ``ErrorModel``.

        Parameters
        ----------
        wb : Wellbore
            The wellbore whose survey covariance is required.
        md : float or None
            Measured depth, in metres, at which to sample the covariance; the
            nearest survey station is used. ``None`` selects the last (TD)
            station.
        error_model : str
            The welleng error-model name to apply when the survey has no cached
            covariance (e.g. ``"ISCWSA MWD Rev5.11"``).

        Returns
        -------
        numpy.ndarray
            The 3x3 absolute NEV covariance at the selected station, in metres
            squared.

        Raises
        ------
        ValueError
            If ``wb`` has no survey attached.
        """
        import numpy as np
        from .error import ErrorModel
        s = wb.survey
        if s is None:
            raise ValueError(f"wellbore {wb.id!r} has no survey")
        cov = getattr(s, "cov_nev", None)
        if cov is None:
            cov = ErrorModel(s, error_model=error_model).errors.cov_nev
        cov = np.asarray(cov)
        if md is None:
            return cov[-1]
        idx = int(np.argmin(np.abs(np.asarray(s.md) - md)))   # nearest station
        return cov[idx]

    def _abs_cov_components(self, wb: "Wellbore", md, error_model: str):
        """Total, global, and systematic NEV covariance components at ``md``.

        Returns the three ISCWSA propagation-mode covariance buckets a wellbore
        exposes (``cov_nev`` total, ``cov_nev_global``, ``cov_nev_systematic``)
        at the nearest station to ``md`` (or at TD when ``md`` is ``None``),
        computing them via welleng's ``ErrorModel`` if not cached. Consumed by
        :meth:`relative_covariance` when a ``share_mode`` folds the divergent
        branches' shared error sources.

        Parameters
        ----------
        wb : Wellbore
            The wellbore whose survey covariance is wanted.
        md : float or None
            Measured depth, in metres; ``None`` selects the last station.
        error_model : str
            welleng error-model name used if the survey has no cached error.

        Returns
        -------
        tuple of numpy.ndarray
            ``(total, global, systematic)`` — each a 3x3 NEV covariance in
            metres squared.

        Raises
        ------
        ValueError
            If ``wb`` has no survey attached.
        """
        import numpy as np
        from .error import ErrorModel
        s = wb.survey
        if s is None:
            raise ValueError(f"wellbore {wb.id!r} has no survey")
        tot = getattr(s, "cov_nev", None)
        glob = getattr(s, "cov_nev_global", None)
        sys = getattr(s, "cov_nev_systematic", None)
        if tot is None or glob is None or sys is None:
            em = ErrorModel(s, error_model=error_model).errors
            tot, glob, sys = em.cov_nev, em.cov_NEVs_global, em.cov_NEVs_systematic
        tot, glob, sys = np.asarray(tot), np.asarray(glob), np.asarray(sys)
        idx = -1 if md is None else int(np.argmin(np.abs(np.asarray(s.md) - md)))
        return tot[idx], glob[idx], sys[idx]

    def _sidetrack_md(self, a: str, b: str):
        """Return the deepest common measured depth of two wellbores.

        The side-track point is the shallower of the two branch kickoffs
        immediately below the wellbores' lowest common ancestor — the depth up
        to which both share a survey.

        Parameters
        ----------
        a : str
            First wellbore node identifier.
        b : str
            Second wellbore node identifier.

        Returns
        -------
        float or None
            The side-track measured depth, in metres, or ``None`` if one
            wellbore is an ancestor of the other (no distinct branch on one
            side) or a branch kickoff MD is missing.
        """
        _shared, branch_a, branch_b = self.shared_and_divergent(a, b)
        # The divergence is the shallower of the two branches' kick-offs. When one
        # wellbore is an ancestor of the other, only the descendant has a branch
        # below the LCA — its kick-off IS the divergence point (the ancestor's own
        # path below there is the other divergent branch).
        kos = []
        if branch_a and self.node(branch_a[-1]).kickoff_md is not None:
            kos.append(self.node(branch_a[-1]).kickoff_md)
        if branch_b and self.node(branch_b[-1]).kickoff_md is not None:
            kos.append(self.node(branch_b[-1]).kickoff_md)
        return min(kos) if kos else None

    # -- serialisation (native JSON save/load) ------------------------------ #
    def to_dict(self) -> dict:
        """Serialise the whole model to a JSON-safe dict.

        Emits the entity graph (each node's type + parent id) together with each
        wellbore's survey (raw md / inc / azi in radians + header + context
        keys). Round-trips via :meth:`from_dict`.

        Returns
        -------
        dict
            A plain, JSON-serialisable dict with keys
            ``"welleng_hierarchy_version"`` and ``"nodes"``.

        Notes
        -----
        Azimuth is serialised in the header's OWN north reference (grid / true /
        magnetic) and that reference is recorded alongside it, so the round-trip
        is frame-exact and never silently reinterprets one reference as another
        (grid and true differ by convergence; true and magnetic by declination,
        which is date + location dependent).

        This is the light welleng-native persistence form; :mod:`welleng.osdu`
        is the OSDU interchange form.
        """
        import numpy as np

        def _survey_dict(s):
            if s is None:
                return None
            hdr = getattr(s, "header", None)
            # azimuth is RELATIVE to a north reference (grid/true/magnetic) — grid
            # and true differ by convergence, true and magnetic by declination
            # (date+location dependent). Serialise azi in the header's OWN
            # reference and record it, so the round-trip is frame-exact and never
            # silently reinterprets one reference as another.
            azi_ref = getattr(hdr, "azi_reference", "grid") if hdr else "grid"
            return {
                "md": list(np.asarray(s.md).tolist()),
                "inc": list(np.asarray(s.inc_rad).tolist()), "deg": False,
                "azi": list(np.asarray(getattr(s, f"azi_{azi_ref}_rad")).tolist()),
                "azi_reference": azi_ref,
                "header": {k: v for k, v in vars(hdr).items()
                           if isinstance(v, (str, int, float, bool, type(None)))} if hdr else None,
                "start_nev": list(np.asarray(s.start_nev).tolist()),
                "error_model": getattr(getattr(s, "header", None), "error_model", None),
            }

        nodes = []
        for n in self._nodes.values():
            d = {"type": type(n).__name__, "id": n.id, "name": n.name,
                 "parent_id": n.parent.id if getattr(n, "parent", None) else None}
            if isinstance(n, Site):
                d.update(crs=n.crs, convergence=n.convergence,
                         is_field_centre=n.is_field_centre, location=n.location)
            elif isinstance(n, Well):
                d.update(slot=n.slot, slot_radial_error=n.slot_radial_error,
                         wellhead_depth=n.wellhead_depth,
                         datum=vars(n.datum) if n.datum else None)
            elif isinstance(n, Wellbore):
                d.update(kickoff_md=n.kickoff_md, survey_date=n.survey_date,
                         tool_id=n.tool_id, geomag_model=n.geomag_model,
                         survey=_survey_dict(n.survey))
            nodes.append(d)
        return {"welleng_hierarchy_version": 1, "nodes": nodes}

    @classmethod
    def from_dict(cls, data: dict) -> "WellNetwork":
        """Reconstruct a :class:`WellNetwork` from :meth:`to_dict` output.

        Rebuilds every entity, wires parents by id (recursively, so a child is
        built after its parent), and re-creates each wellbore's
        :class:`~welleng.survey.Survey` from the serialised md / inc / azi +
        header.

        Parameters
        ----------
        data : dict
            A dict in the shape produced by :meth:`to_dict`.

        Returns
        -------
        WellNetwork
            The reconstructed network.
        """
        from .survey import Survey, SurveyHeader
        net = cls()
        raw = {d["id"]: d for d in data["nodes"]}
        built: dict[str, _Node] = {}
        _classes = {"Organisation": Organisation, "Field": Field, "Site": Site,
                    "Well": Well, "Wellbore": Wellbore}

        def _build(nid):
            if nid in built:
                return built[nid]
            d = raw[nid]
            kind = d["type"]
            parent = _build(d["parent_id"]) if d.get("parent_id") else None
            if kind == "Site":
                n = Site(id=d["id"], name=d["name"], parent=parent, crs=d.get("crs"),
                         convergence=d.get("convergence"),
                         is_field_centre=d.get("is_field_centre", False),
                         location=tuple(d["location"]) if d.get("location") else None)
            elif kind == "Well":
                dm = d.get("datum")
                n = Well(id=d["id"], name=d["name"], parent=parent,
                         slot=tuple(d["slot"]) if d.get("slot") else None,
                         slot_radial_error=d.get("slot_radial_error", 0.0),
                         wellhead_depth=d.get("wellhead_depth"),
                         datum=Datum(**dm) if dm else None)
            elif kind == "Wellbore":
                sv = d.get("survey"); survey = None
                if sv:
                    survey = Survey(md=sv["md"], inc=sv["inc"], azi=sv["azi"], deg=sv.get("deg", False),
                                    header=SurveyHeader(**sv["header"]) if sv.get("header") else None,
                                    start_nev=sv.get("start_nev", [0., 0., 0.]),
                                    error_model=sv.get("error_model"))
                n = Wellbore(id=d["id"], name=d["name"], parent=parent,
                             kickoff_md=d.get("kickoff_md"), survey_date=d.get("survey_date"),
                             tool_id=d.get("tool_id"), geomag_model=d.get("geomag_model"),
                             survey=survey)
            else:
                n = _classes[kind](id=d["id"], name=d["name"], parent=parent)
            built[nid] = n
            if isinstance(n, (Well, Wellbore)):
                net.add(n)
            else:
                net._nodes[n.id] = n
            return n

        for nid in raw:
            _build(nid)
        return net

    def save_json(self, path: str) -> None:
        """Save the model to a JSON file.

        Parameters
        ----------
        path : str
            Filesystem path to write the JSON to (overwritten if it exists).

        Returns
        -------
        None

        Examples
        --------
        >>> import os, tempfile
        >>> from welleng.hierarchy import Well, Wellbore, WellNetwork
        >>> net = WellNetwork()
        >>> well = Well(id='W1', name='W1')
        >>> _ = net.add(Wellbore(id='WB1', name='TopHole', parent=well))
        >>> path = os.path.join(tempfile.mkdtemp(), 'net.json')
        >>> net.save_json(path)
        >>> reloaded = WellNetwork.load_json(path)
        >>> [w.id for w in reloaded.roots()]
        ['WB1']
        """
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "WellNetwork":
        """Load a model from a JSON file.

        Parameters
        ----------
        path : str
            Filesystem path to a JSON file written by :meth:`save_json`.

        Returns
        -------
        WellNetwork
            The reconstructed network. See :meth:`save_json` for a round-trip
            example.
        """
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # -- internals ---------------------------------------------------------- #
    def _wellbores(self) -> Iterator[Wellbore]:
        return (n for n in self._nodes.values() if isinstance(n, Wellbore))

    def _parent_id(self, id_: str) -> Optional[str]:
        preds = list(self._g.predecessors(id_))
        return preds[0] if preds else None
