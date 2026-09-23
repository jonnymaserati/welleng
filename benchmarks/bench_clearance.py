"""Benchmark for clearance: ISCWSA pedal curve (``IscwsaClearance``) and the
Mahalanobis separation factor (``MahalanobisClearance``).

Times the reference well against each offset of the ISCWSA standard set of
wellpaths for evaluating clearance scenarios (the validation set in
``tests/test_data/clearance_iscwsa_well_data.json``): the pedal curve with and
without the interpolated-minimum refinement (``minimize_sf``), and the
Mahalanobis separation factor. Run from the repo root:

    python benchmarks/bench_clearance.py

Timings are per host and indicative -- compare RELATIVE change on the same
machine. Results are logged in ``benchmarks/BENCHMARK_LOG.md``.
"""
from __future__ import annotations

import json
import time
import warnings

import numpy as np

from welleng.clearance import IscwsaClearance, MahalanobisClearance
from welleng.survey import Survey, make_survey_header

DATA = "tests/test_data/clearance_iscwsa_well_data.json"


def load_surveys(path=DATA):
    wells = json.load(open(path))["wells"]
    surveys = {}
    for name, w in wells.items():
        surveys[name] = Survey(
            md=w["MD"], inc=w["IncDeg"], azi=w["AziDeg"],
            n=w["N"], e=w["E"], tvd=w["TVD"],
            radius=0.4572 if name == "Reference well" else 0.3048,
            header=make_survey_header(w["header"]),
            error_model="ISCWSA MWD Rev4",
            start_nev=[w["N"][0], w["E"][0], w["TVD"][0]],
            deg=True, unit="meters",
        )
    return surveys


def run(surveys, minimize_sf=False, cls=IscwsaClearance):
    ref = surveys["Reference well"]
    kwargs = {} if cls is MahalanobisClearance else {"minimize_sf": minimize_sf}
    for name, off in surveys.items():
        if name == "Reference well":
            continue
        cls(ref, off, kop_depth=900.0 if name == "10 - well" else -np.inf,
            **kwargs)


def best_of(fn, repeats=5):
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t)
    return min(times)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    surveys = load_surveys()
    n = len(surveys) - 1
    print(f"ISCWSA set: reference vs {n} offsets (best of 5)")
    for minimize_sf in (False, True):
        t = best_of(lambda: run(surveys, minimize_sf))
        print(f"  pedal, minimize_sf={minimize_sf!s:5}  {1e3 * t:8.1f} ms total"
              f"  {1e3 * t / n:7.1f} ms/pair")
    t = best_of(lambda: run(surveys, cls=MahalanobisClearance))
    print(f"  Mahalanobis               {1e3 * t:8.1f} ms total"
          f"  {1e3 * t / n:7.1f} ms/pair")
