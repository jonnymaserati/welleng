"""Per-case monotonicity analysis (SymPy) of the kick-tolerance margin.

For EACH case (drill, swab) SEPARATELY we form the margin

    margin = capacity(A-7 / A-8) - kt_threshold

as a symbolic function of the credible-box OPERATIONAL inputs
(rho_mud, PP, kick_intensity, P_lot, P_apl, V_dpa, D_td, D_lot / D_shoe), and
treat the gas properties (Z_s, Z_td, rho_gas_s) and temperatures as PARAMETERS.
We compute d(margin)/d(x_i) symbolically and establish the SIGN of each partial
over the credible physical box -> a monotonicity table:

    var -> sign -> adverse bound (the corner value that MINIMISES the margin).

Any variable whose partial does NOT keep a constant sign over the box is flagged
NON-MONOTONE (needs a sweep, not a corner).

HONEST SCOPE NOTE
-----------------
Monotonicity here is PER-CASE ONLY. The interesting sign flips (most notably
mud weight rho_mud) live BETWEEN the drill case and the swab case -- which is
exactly why the two cases are kept separate and are never combined into one
global gate. There is NO global-monotonicity claim across the two cases.

  * drill: bottom-hole pressure P_td = g*(PP + kick)*D_td  (max-credible PP
    branch of A-1; the branch that matters when a kick is credible). Raising mud
    weight RAISES B (helps) while not entering P_td on this branch.
  * swab:  P_td = g*rho_mud*D_td. Here raising mud weight RAISES P_td (hurts).
    So the sign of d(margin)/d(rho_mud) differs between the two cases.
"""

from __future__ import annotations

import itertools

import sympy as sp

# --- Symbols ----------------------------------------------------------------
# Operational (credible-box) variables:
rho_mud, PP, kick, P_lot, P_apl, V_dpa, D_td, D_lot = sp.symbols(
    "rho_mud PP kick P_lot P_apl V_dpa D_td D_lot", positive=True
)
# Parameters (held fixed for the sign analysis):
Z_s, Z_td, rho_gas_s, T_s_R, T_td_R, g, P_atm, V_req = sp.symbols(
    "Z_s Z_td rho_gas_s T_s_R T_td_R g P_atm V_req", positive=True
)

OPERATIONAL_VARS = [rho_mud, PP, kick, P_lot, P_apl, V_dpa, D_td, D_lot]

# --- Closed form (symbolic) -------------------------------------------------
P_lot_psi = g * P_lot * D_lot
A_sym = (P_lot_psi - P_apl) * T_td_R * Z_td * V_dpa / (g * T_s_R * Z_s * (rho_mud - rho_gas_s))
B_sym = P_lot_psi - P_apl + g * rho_mud * (D_td - D_lot)

# drill: max-credible PP branch of A-1
P_td_drill = g * (PP + kick) * D_td
# swab: mud hydrostatic (A-8 substitution)
P_td_swab = g * rho_mud * D_td


def _margin(P_td_expr):
    return A_sym * (B_sym - P_td_expr) / (P_td_expr + P_atm) - V_req


MARGIN = {"drill": _margin(P_td_drill), "swab": _margin(P_td_swab)}

# --- Numeric parameter values + credible box --------------------------------
# Gas properties (fixed parameters for the sign analysis) are computed once by
# the clean-room Hall-Yarborough backend at the nominal Table-1 scenario.
from .core import KickInputs, annular_capacity_dpa, resolve_gas_properties

_nominal = KickInputs(
    rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0, P_apl=210.0,
    D_td=10500.0, D_lot=6500.0, T_s=212.0, T_td=302.0,
    V_dpa=annular_capacity_dpa(6.125, 4.0), kt_threshold=25.0,
)
_Z_s, _Z_td, _rho_gas_s = resolve_gas_properties(_nominal)

PARAMS = {
    Z_s: _Z_s, Z_td: _Z_td, rho_gas_s: _rho_gas_s,
    T_s_R: 212.0 + 460.0, T_td_R: 302.0 + 460.0,
    g: 0.0521, P_atm: 14.7, V_req: 25.0,
}

# Credible ranges (lo, hi) for each operational variable.
BOX = {
    rho_mud: (11.0, 13.0),
    PP: (11.0, 12.0),
    kick: (0.5, 1.5),
    P_lot: (15.0, 17.0),
    P_apl: (100.0, 300.0),
    V_dpa: (0.015, 0.025),
    D_td: (10000.0, 11000.0),
    D_lot: (6000.0, 7000.0),
}

NAMES = {
    rho_mud: "rho_mud", PP: "PP", kick: "kick_intensity", P_lot: "P_lot",
    P_apl: "P_apl", V_dpa: "V_dpa", D_td: "D_td", D_lot: "D_lot(D_shoe)",
}


def _branch_valid(case, pt):
    """Only sample where the assumed A-1 branch and positive capacity hold."""
    pp_press = pt[PP] + pt[kick]
    if case == "drill" and pp_press < pt[rho_mud]:
        return False  # would be the mud-hydrostatic branch, not PP branch
    return True


def _box_samples():
    """All 2^8 corners of the credible box (sign is checked at every corner)."""
    keys = list(BOX)
    for combo in itertools.product(*[BOX[k] for k in keys]):
        yield dict(zip(keys, combo))


def analyse(case):
    """Return a list of rows: (name, sign_str, adverse_bound_str, note)."""
    margin = MARGIN[case]
    rows = []
    for var in OPERATIONAL_VARS:
        deriv = sp.diff(margin, var)
        dfun = sp.lambdify(list(BOX) + list(PARAMS), deriv, "math")

        signs = set()
        for corner in _box_samples():
            if not _branch_valid(case, corner):
                continue
            args = [corner[k] for k in BOX] + [PARAMS[k] for k in PARAMS]
            val = dfun(*args)
            if abs(val) < 1e-12:
                signs.add(0)
            else:
                signs.add(1 if val > 0 else -1)

        lo, hi = BOX[var]
        if signs == {0} or not signs:
            rows.append((NAMES[var], "0 (no effect)", "n/a",
                         "case is independent of this variable"))
        elif signs == {1}:
            # margin increases with var -> adverse = LOW bound
            rows.append((NAMES[var], "+  (monotone up)", f"LOW = {lo:g}",
                         "push to lower bound"))
        elif signs == {-1}:
            # margin decreases with var -> adverse = HIGH bound
            rows.append((NAMES[var], "-  (monotone down)", f"HIGH = {hi:g}",
                         "push to upper bound"))
        else:
            rows.append((NAMES[var], "VARIES (non-monotone)", f"SWEEP [{lo:g},{hi:g}]",
                         "sign not constant over box -> sweep, not a corner"))
    return rows


def directions(case):
    """Machine-readable adverse direction per operational var for the envelope.

    Returns {var_name: one of 'low' | 'high' | 'zero' | 'sweep'} where the
    string is the bound the margin is MINIMISED at ('sweep' = non-monotone).
    """
    out = {}
    for name, sign, _bound, _note in analyse(case):
        if "VARIES" in sign:
            out[name] = "sweep"
        elif sign.startswith("+"):
            out[name] = "low"    # margin rises with var -> adverse = low
        elif sign.startswith("-"):
            out[name] = "high"   # margin falls with var -> adverse = high
        else:
            out[name] = "zero"
    return out


def _print_table(case, rows):
    print(f"\n=== MONOTONICITY TABLE -- case: {case.upper()} "
          f"(margin = capacity - kt_threshold) ===")
    print(f"{'variable':<16}{'d(margin)/dx sign':<24}{'adverse bound':<20}note")
    print("-" * 92)
    for name, sign, bound, note in rows:
        print(f"{name:<16}{sign:<24}{bound:<20}{note}")


def main():
    lines = []
    for case in ("drill", "swab"):
        rows = analyse(case)
        _print_table(case, rows)
        lines.append(f"=== {case.upper()} ===")
        lines.append(f"{'variable':<16}{'sign':<24}{'adverse bound':<20}note")
        for r in rows:
            lines.append(f"{r[0]:<16}{r[1]:<24}{r[2]:<20}{r[3]}")
        lines.append("")

    non_mono = {}
    for case in ("drill", "swab"):
        non_mono[case] = [r[0] for r in analyse(case) if "VARIES" in r[1]]
    print("\n--- NON-MONOTONE (need a sweep, not a corner) ---")
    for case in ("drill", "swab"):
        nm = non_mono[case] or ["(none)"]
        print(f"  {case:<6}: {', '.join(nm)}")
        lines.append(f"NON-MONOTONE {case}: {', '.join(nm)}")

    print("\nNOTE: monotonicity is PER-CASE only. The rho_mud sign flip lives "
          "BETWEEN\n      drill and swab -- the reason the two cases stay "
          "separate. No global claim.")

    with open("monotonicity_table.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")
        fh.write("\nNOTE: per-case monotonicity only; rho_mud sign flips "
                 "BETWEEN drill and swab. No global-monotonicity claim.\n")
    print("\nSaved: monotonicity_table.txt")


if __name__ == "__main__":
    main()
