# Citing welleng, and the work it builds on

welleng credits the software and science it stands on — and we'd appreciate the
same courtesy. This file is the single place for how to cite welleng, the
dependencies it is built on, and the published methods it implements. Consider
it a lead-by-example: this is how we'd like welleng cited, too.

The machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff) — the
[Citation File Format](https://citation-file-format.github.io/), from which
GitHub renders a **"Cite this repository"** button. This file is its
human-readable companion.

## How to cite welleng

**Software** — use the concept DOI (always resolves to the latest version):

> Corcutt, J. *welleng: open-source well-engineering tools.* Zenodo.
> <https://doi.org/10.5281/zenodo.20968887>

```bibtex
@software{corcutt_welleng,
  author    = {Corcutt, Jonathan},
  title     = {welleng: open-source well-engineering tools},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20968887},
  url       = {https://doi.org/10.5281/zenodo.20968887}
}
```

If your work relies on a specific method, please **also** cite the relevant paper:

- **Gyro error-model validation** — Corcutt, J. (2026). *Reproducing the ISCWSA
  Gyro Error-Model Test Cases.* Zenodo.
  [doi:10.5281/zenodo.20940515](https://doi.org/10.5281/zenodo.20940515)
- **Anti-collision (exact Mahalanobis separation factor)** — Corcutt, J. (2026).
  *Making the Exact Wellbore Anti-Collision Boundary Practical.* Zenodo.
  [doi:10.5281/zenodo.20976872](https://doi.org/10.5281/zenodo.20976872)
- **Analytical curve-hold-curve point-to-target solver** — Corcutt, J. (2026).
  *An Open, Vectorized Closed-Form Solver for the 3D Curve-Hold-Curve
  Point-to-Target Problem.* Zenodo.
  [doi:10.5281/zenodo.21130979](https://doi.org/10.5281/zenodo.21130979)
- **Tortuosity index (non-independence + dimensionless form)** — Corcutt, J.
  (2026). *A Note on the Well Trajectory Tortuosity Index: Non-Independence of
  the Inclination–Azimuth Combination, and a Dimensionless Form.* Zenodo.
  [doi:10.5281/zenodo.21206271](https://doi.org/10.5281/zenodo.21206271)
- **Kick-tolerance engine (derivation, provenance, validation)** — Corcutt, J.
  (2026). *The welleng kick-tolerance engine: derivation, provenance and public
  validation against published methods.* Zenodo.
  [doi:10.5281/zenodo.21412388](https://doi.org/10.5281/zenodo.21412388)

## Built on — software dependencies

**Core**

- **NumPy** — Harris, C. R., et al. (2020). *Array programming with NumPy.*
  Nature 585, 357–362. [doi:10.1038/s41586-020-2649-2](https://doi.org/10.1038/s41586-020-2649-2)
- **SciPy** — Virtanen, P., et al. (2020). *SciPy 1.0: Fundamental Algorithms
  for Scientific Computing in Python.* Nature Methods 17, 261–272.
  [doi:10.1038/s41592-019-0686-2](https://doi.org/10.1038/s41592-019-0686-2)
- **pandas** — McKinney, W. (2010). *Data Structures for Statistical Computing
  in Python.* Proc. 9th Python in Science Conf., 56–61.
  [doi:10.25080/Majora-92bf1922-00a](https://doi.org/10.25080/Majora-92bf1922-00a)
- **pyproj / PROJ** — PROJ contributors. OSGeo. <https://proj.org>
- **Pint** — <https://github.com/hgrecco/pint>
- **openpyxl**, **PyYAML**, **setuptools** — acknowledged by name.

**Optional** (mesh, visualisation, extras)

- **trimesh** — Dawson-Haggerty, M., et al. <https://trimesh.org>
- **python-fcl / FCL** — Pan, J., Chitta, S., Manocha, D. (2012). *FCL: A general
  purpose library for collision and proximity queries.* IEEE ICRA, 3859–3866.
  [doi:10.1109/ICRA.2012.6225337](https://doi.org/10.1109/ICRA.2012.6225337)
- **Matplotlib** — Hunter, J. D. (2007). *Matplotlib: A 2D Graphics Environment.*
  Computing in Science & Engineering 9(3), 90–95.
  [doi:10.1109/MCSE.2007.55](https://doi.org/10.1109/MCSE.2007.55)
- **CoolProp** — Bell, I. H., Wronski, J., Quoilin, S., Lemort, V. (2014). *Pure
  and Pseudo-pure Fluid Thermophysical Property Evaluation and the Open-Source
  Thermophysical Property Library CoolProp.* Industrial & Engineering Chemistry
  Research 53(6), 2498–2508.
  [doi:10.1021/ie4033999](https://doi.org/10.1021/ie4033999) — optional
  reference-EOS backend for kick-tolerance gas-property validation.
- **SymPy** — Meurer, A., et al. (2017). *SymPy: symbolic computing in Python.*
  PeerJ Computer Science 3:e103. [doi:10.7717/peerj-cs.103](https://doi.org/10.7717/peerj-cs.103)
- **VTK / vedo**, **NetworkX**, **utm**, **magnetic_field_calculator**,
  **tabulate** — acknowledged by name.

## Methods & standards welleng implements

### Standards bodies

- **ISCWSA** — Industry Steering Committee on Wellbore Survey Accuracy.
  <https://www.iscwsa.net> (error-model definitions and standard test data)
- **OWSG** — Operators Wellbore Survey Group (gyro tool error-model stacks),
  under ISCWSA.

### Wellbore position uncertainty (error models)

- **MWD** — Williamson, H. S. (2000). *Accuracy Prediction for Directional
  Measurement While Drilling.* SPE Drilling & Completion 15(4). SPE 67616-PA.
  [doi:10.2118/67616-PA](https://doi.org/10.2118/67616-PA)
- **Gyro** — Torkildsen, T., Håvardstein, S. T., Weston, J. L., Ekseth, R.
  (2004). *Prediction of Wellbore Position Accuracy When Surveyed With
  Gyroscopic Tools.* SPE 90408.
  [doi:10.2118/90408-MS](https://doi.org/10.2118/90408-MS)
- **Survey quality** — Ekseth, R., et al. (2010). *High-Integrity Wellbore
  Surveying.* SPE Drilling & Completion 25(4). SPE 133417-PA.
  [doi:10.2118/133417-PA](https://doi.org/10.2118/133417-PA)

### Trajectory geometry

- **Minimum curvature, TVD interpolation & turning points** — Sawaryn, S. J.,
  Thorogood, J. L. (2005). *A Compendium of Directional Calculations Based on
  the Minimum Curvature Method.* SPE Drilling & Completion. SPE 84246-PA.
  [doi:10.2118/84246-PA](https://doi.org/10.2118/84246-PA) — used for the
  minimum-curvature station geometry, and for the reversal-robust
  `interpolate_tvd` (closed-form *Interpolation at a Plane*, Eqs. 25-27 and
  Eq. 1, specialised to a horizontal plane; *Turning Point*, Eq. 31, to
  segment each arc into monotonic-TVD spans).

### Anti-collision / separation

- **Separation rule** — Sawaryn, S. J., et al. (2019). *Well-Collision-Avoidance
  Separation Rule.* SPE Drilling & Completion 34, 01–15. SPE 187073-PA.
  [doi:10.2118/187073-PA](https://doi.org/10.2118/187073-PA)
- **Mahalanobis collision probability** — Brooks, A. G. (2010). *A New Look at
  Wellbore-Collision Probability.* SPE Drilling & Completion 25(2), 223–232.
  SPE 116155-PA. [doi:10.2118/116155-PA](https://doi.org/10.2118/116155-PA);
  and Brooks, A. G., Wilson, H. (1996). SPE 36863-MS.
  [doi:10.2118/36863-MS](https://doi.org/10.2118/36863-MS)
- **Foundations** — Mahalanobis, P. C. (1936). *On the generalised distance in
  statistics.* Proc. National Institute of Sciences of India 2(1), 49–55;
  Alfano, S. (2006). *Satellite Collision Probability Enhancements.* J. Guidance,
  Control, and Dynamics 29(3), 588–592.
  [doi:10.2514/1.15523](https://doi.org/10.2514/1.15523);
  Bang, J. (2017). *Quantification of Wellbore-Collision Probability by Novel
  Analytic Methods.* SPE Drilling & Completion. SPE 184644-PA.
  [doi:10.2118/184644-PA](https://doi.org/10.2118/184644-PA)

### Kick tolerance

- **Single-bubble kick-tolerance closed form** — Thorogood, J. L., Robertson, E.,
  Castillo, D., Sawaryn, S. J. (2022). *An Assessment of the Kick Tolerance
  Calculation for Well Design and Drilling Operations.* SPE Drilling &
  Completion. SPE 208788-PA.
  [doi:10.2118/208788-PA](https://doi.org/10.2118/208788-PA) — the public
  Appendix-A closed form (Eqs. A-1…A-9) transcribed for the Tier-0 single-bubble
  drill/swab margin.
- **Real-gas Z-factor** — Hall, K. R., Yarborough, L. (1973). *A new equation of
  state for Z-factor calculations.* Oil & Gas Journal 71(25), 82–92 — the
  clean-room Hall & Yarborough compressibility-factor backend for the Tier-0
  pure-methane influx.
- **NOGEPA Industry Standard No. 50** — *Kick Tolerances for Well Design &
  Drilling Operations.* Netherlands Oil and Gas Exploration and Production
  Association — the operational kick-tolerance design basis.
- **Deviated-well form** — Kiani Nassab, K., Ting, S. Z., Buapha, S., MatNoh, N.,
  Hemmati, M. N. (2022). *How to Improve Accuracy of a Kick Tolerance Model by
  Considering the Effects of Kick Classification, Frictional Losses, Pore Pressure
  Profile, and Influx Temperature.* SPE Drilling & Completion 37(01), 15.
  SPE 202426-PA. [doi:10.2118/202426-PA](https://doi.org/10.2118/202426-PA) — the
  single-bubble method with true-vertical-depth pressures and the gas-column
  vertical-height → along-hole-length conversion (H_gas → L_gas) for deviated wells.

welleng is grateful to these projects, their maintainers, and the authors and
standards bodies whose methods it implements.
