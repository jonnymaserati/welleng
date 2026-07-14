"""Deterministic barrier envelope for kick tolerance (no probability, no MC).

Given a credible RANGE per operational input, find the worst-credible-case
margin for EACH case (drill, swab) SEPARATELY, using the per-case monotonicity
table (``monotonicity.directions``):

  * push each MONOTONE variable to its adverse bound (a box corner), and
  * SWEEP any NON-MONOTONE variable across its range, taking the minimum margin.

For each case the envelope returns:
  * PASS / FAIL at the worst credible corner,
  * the BINDING constraint (which variable, at which bound, governs the margin),
  * a tighten-by-how-much sensitivity d(margin)/dx at the binding point.

The two cases are reported SEPARATELY and never blind-min'd / AND-gated:
  * ``drill`` result  = drillability (the section PASS/FAIL),
  * ``swab``  result  = an operational (pump-out) limit; a swab failure means
    "pump out of the hole", which is the drill case, already assessed.

Purely deterministic: corners + a 1-D sweep of any non-monotone variable. NO
probability distribution and NO Monte Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .core import KickInputs, drill_kick, swab_kick
from .monotonicity import directions

# operational-var name (as used in monotonicity) -> KickInputs field
_FIELD = {
    "rho_mud": "rho_mud",
    "PP": "PP",
    "kick_intensity": "kick_intensity",
    "P_lot": "P_lot",
    "P_apl": "P_apl",
    "V_dpa": "V_dpa",
    "D_td": "D_td",
    "D_lot(D_shoe)": "D_lot",
}

_CASE_FN = {"drill": drill_kick, "swab": swab_kick}


@dataclass
class EnvelopeResult:
    case: str
    passed: bool
    worst_margin: float
    worst_capacity: float
    worst_inputs: KickInputs
    binding_var: str
    binding_bound: str          # the value the binding var sits at
    binding_swing: float        # margin cost of this var (mid -> adverse)
    binding_sensitivity: float  # d(margin)/dx at the worst corner


def _margin(case, inp):
    return _CASE_FN[case](inp).margin


def _apply(inp, name, value):
    return replace(inp, **{_FIELD[name]: value})


def worst_case(case, nominal: KickInputs, ranges: dict, n_sweep: int = 21):
    """Worst credible margin for one case over the given per-var ``ranges``.

    ``nominal`` supplies the fixed parameters (gas props, temps, threshold) and
    default values for any var not present in ``ranges``. ``ranges`` maps a
    monotonicity var-name (e.g. 'rho_mud', 'D_lot(D_shoe)') to (lo, hi).
    """
    dirn = directions(case)
    inp = nominal
    sweep_vars = []

    # 1) push monotone vars to their adverse bound; collect non-monotone sweeps.
    for name, (lo, hi) in ranges.items():
        d = dirn[name]
        if d == "low":
            inp = _apply(inp, name, lo)
        elif d == "high":
            inp = _apply(inp, name, hi)
        elif d == "zero":
            pass  # no effect -> leave at nominal
        else:  # 'sweep'
            sweep_vars.append(name)

    # 2) sweep any non-monotone var(s), take the worst (minimum) margin corner.
    #    (independent 1-D sweeps composed onto the corner; deterministic grid.)
    for name in sweep_vars:
        lo, hi = ranges[name]
        grid = np.linspace(lo, hi, n_sweep)
        best_val, best_margin = lo, np.inf
        for v in grid:
            m = _margin(case, _apply(inp, name, v))
            if m < best_margin:
                best_margin, best_val = m, v
        inp = _apply(inp, name, best_val)

    worst = _CASE_FN[case](inp)

    # 3) binding constraint: relax each ranged var back to its benign bound and
    #    measure the margin gain -> the var with the largest gain governs.
    binding_var, binding_swing, binding_bound = None, -np.inf, ""
    for name, (lo, hi) in ranges.items():
        d = dirn[name]
        if d == "zero":
            continue
        benign = hi if d == "low" else lo if d == "high" else None
        cur = getattr(inp, _FIELD[name])
        if benign is None:  # swept var: benign = the non-worst end
            benign = hi if abs(cur - lo) < abs(cur - hi) else lo
        gain = _margin(case, _apply(inp, name, benign)) - worst.margin
        if gain > binding_swing:
            binding_swing, binding_var = gain, name
            binding_bound = f"{cur:g}"

    # 4) tighten sensitivity d(margin)/dx at the worst corner (central diff).
    cur = getattr(inp, _FIELD[binding_var])
    step = (abs(cur) * 1e-4) or 1e-4
    dm = (_margin(case, _apply(inp, binding_var, cur + step))
          - _margin(case, _apply(inp, binding_var, cur - step))) / (2 * step)

    return EnvelopeResult(
        case=case,
        passed=worst.passed,
        worst_margin=worst.margin,
        worst_capacity=worst.capacity,
        worst_inputs=inp,
        binding_var=binding_var,
        binding_bound=binding_bound,
        binding_swing=binding_swing,
        binding_sensitivity=dm,
    )


def evaluate_envelope(nominal: KickInputs, ranges: dict, n_sweep: int = 21):
    """Return {'drill': EnvelopeResult, 'swab': EnvelopeResult} (never min'd)."""
    return {case: worst_case(case, nominal, ranges, n_sweep) for case in ("drill", "swab")}


def _report(res: EnvelopeResult):
    verdict = "PASS" if res.passed else "FAIL"
    tag = "drillability" if res.case == "drill" else "operational (pump-out)"
    print(f"\n[{res.case.upper()}]  {verdict}   ({tag})")
    print(f"  worst-credible capacity = {res.worst_capacity:8.3f} bbl "
          f"(threshold {res.worst_inputs.kt_threshold:g})")
    print(f"  worst-credible margin   = {res.worst_margin:+8.3f} bbl")
    print(f"  binding constraint      = {res.binding_var} at {res.binding_bound}")
    print(f"    margin cost of this var (relax->benign) = {res.binding_swing:+.3f} bbl")
    print(f"    tighten sensitivity  d(margin)/d({res.binding_var}) "
          f"= {res.binding_sensitivity:+.4g} bbl per unit")


def main():
    # Nominal (Table-1) case: temps + threshold live here. Gas properties are
    # computed ONCE by the clean-room Hall-Yarborough backend at the nominal
    # scenario, then held fixed as parameters for the deterministic sensitivity
    # analysis (consistent with the per-case monotonicity treatment).
    from .core import annular_capacity_dpa, resolve_gas_properties
    nominal = KickInputs(
        rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0, P_apl=210.0,
        D_td=10500.0, D_lot=6500.0, T_s=212.0, T_td=302.0,
        V_dpa=annular_capacity_dpa(6.125, 4.0), kt_threshold=25.0,
    )
    Z_s, Z_td, rho_gas_s = resolve_gas_properties(nominal)
    nominal = replace(nominal, Z_s=Z_s, Z_td=Z_td, rho_gas_s=rho_gas_s)
    print(f"computed gas props (Hall-Yarborough, methane): "
          f"Z_s={Z_s:.4f} Z_td={Z_td:.4f} rho_gas_s={rho_gas_s:.4f} ppg")

    # Sample credible box (deterministic ranges per operational input).
    ranges = {
        "rho_mud": (11.5, 12.3),
        "PP": (11.3, 11.8),
        "kick_intensity": (0.8, 1.4),
        "P_lot": (15.5, 16.3),
        "P_apl": (150.0, 280.0),
        "V_dpa": (annular_capacity_dpa(6.125, 4.5),
                  annular_capacity_dpa(6.125, 4.0)),
        "D_td": (10300.0, 10800.0),
        "D_lot(D_shoe)": (6300.0, 6600.0),
    }

    print("=== DETERMINISTIC BARRIER ENVELOPE (worst credible case per case) ===")
    print("credible box:")
    for k, (lo, hi) in ranges.items():
        print(f"    {k:<16} [{lo:g}, {hi:g}]")

    results = evaluate_envelope(nominal, ranges)
    for case in ("drill", "swab"):
        _report(results[case])

    print("\nNOTE: drill = drillability gate (section PASS/FAIL); swab = "
          "operational\n      pump-out limit. Reported SEPARATELY -- swab never "
          "overrides drill.")


if __name__ == "__main__":
    main()
