# AGENTS.md — welleng

Well engineering: survey management, ISCWSA error models, clearance/anti-collision, trajectory connectors.

Instructions for coding agents. Read before changing code.

## Setup

Python 3.10+. Use `uv`, not bare `pip`:

```bash
uv venv .venv312 --python 3.12
source .venv312/bin/activate
uv pip install -e ".[easy]"
```

## Build, test, lint

```bash
python -m pytest tests/ -q -n auto     # full suite, ~80 s multi-core
uvx ruff check welleng/ tests/         # must be clean before committing
python -m welleng.lint welleng/        # see "Never interpolate a trajectory linearly"
```

Run all three before opening a pull request.

## Layout

| path | holds |
|---|---|
| `welleng/survey.py`, `utils.py`, `connector.py` | trajectory: survey container, minimum-curvature primitives, the pos/vec connector |
| `welleng/error.py`, `welleng/errors/` | ISCWSA/OWSG error models and the weight functions |
| `welleng/clearance.py`, `mesh.py` | anti-collision separation and wellbore meshes |
| `welleng/exchange/` | file/API readers — EDM, WITSML, LAS, NLOG, IPM, WBP |
| `welleng/schematic/` | well schematic models and CAD output |
| `welleng/kick_tolerance/` | kick tolerance and kill-sheet calculations |
| `welleng/units.py` | the unit registry |
| `tests/` | the suite; `docs/` the Sphinx sources |

## Conventions

These are mistakes actually made in this codebase. Cheap to avoid, expensive to miss: the wrong answer looks plausible.

**Never interpolate a trajectory linearly.** A survey between stations is a circular arc, not a chord. `np.interp(md, survey.md, survey.tvd)` returns a monotonic, in-range, wrong depth — right to centimetres on a dense survey, out by metres on a sparse one. The error scales with the CALLER's station spacing, not with anything this repo controls, so no test of yours will fail. Use `Survey.interpolate_md(md)` or `interpolate_mds(survey, mds)`; `Survey.interpolate_tvd(tvd)` for the inverse; `MinCurve.inc_azi_at(md)` for attitude. Inclination and azimuth are worse than depth: they are angles, and azimuth wraps at 0/360. `python -m welleng.lint` checks for this.

**Convert units through `welleng.units`.** One pint registry: `ureg`, `to_si`, and named helpers (`length`, `pressure`, `mud_weight`, `temperature`). No hard-coded conversion factors. Absolute temperature and temperature *difference* are different helpers, deliberately.

**Use the readers in `welleng/exchange/`.** No hand-rolled parser for EDM, WITSML, LAS or NLOG. If a reader has a gap, raise an issue describing it — the fix belongs in the reader.

**Cite the source for published maths.** SPE number, DOI or standard clause, plus the equation or section reference, in the docstring.

**A default must not assert what was never recorded.** Unknown value: prefer `None` and say so, not a plausible zero. Several fields are `Optional` for this reason — an unrecorded cement top is not cement at surface, and an unrecorded datum elevation is not sea level.

## Testing

- Assert against a reference: a published worked example, a standard's test case, or an independent implementation. State the tolerance.
- Where two code paths compute the same quantity, assert they agree — bitwise where the maths is identical.
- A test pinning current behaviour without a reference pins the bugs too.

## Pull requests

- One topic per branch.
- Suite green, `ruff` clean.
- Behaviour change: update the docstring. A changed edge-case return goes in the docstring, not just the PR.

## Citing welleng

See [`CITATIONS.md`](CITATIONS.md) — software DOI, published papers, and the methods and standards welleng implements.
