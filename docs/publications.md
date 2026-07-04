# Publications

Peer-adjacent open research published alongside `welleng` by Jonathan Corcutt
(Corcutt Beheer B.V., Wassenaar, Netherlands;
[ORCID 0009-0008-1953-7760](https://orcid.org/0009-0008-1953-7760)). Each paper is
openly deposited with a citable DOI and is accompanied by an open-source reference
implementation in `welleng`. For how to cite the software and these papers, see
[CITATIONS.md](https://github.com/jonnymaserati/welleng/blob/main/CITATIONS.md).

The `welleng` software itself has a permanent concept DOI:
[10.5281/zenodo.20968887](https://doi.org/10.5281/zenodo.20968887).

---

## An Open, Vectorized Closed-Form Solver for the 3D Curve-Hold-Curve Point-to-Target Problem

Jonathan Corcutt (2026). **DOI: [10.5281/zenodo.21130979](https://doi.org/10.5281/zenodo.21130979)**.

The curve-hold-curve (circle-line-circle, CLC) point-to-target problem — connecting a
start station to a target station with two circular arcs joined by a straight tangent
under a dogleg-severity limit — is the core construction in directional-drilling
trajectory planning. This paper gives an open, vectorized closed-form solver (correcting
the degree-10 resultant polynomial), together with a maximum-radius / landing / radius-sweep
toolkit and a validated reference implementation.

*Keywords: directional drilling, well trajectory, curve-hold-curve, Dubins path, minimum
curvature, closed-form solver, dogleg severity, path planning.*

---

## Reproducing the ISCWSA Gyro Error-Model Test Cases

Jonathan Corcutt (2026). **DOI: [10.5281/zenodo.20940515](https://doi.org/10.5281/zenodo.20940515)**.

The ISCWSA gyro error model (Torkildsen et al., SPE 90408, 2004; Rev 5.13, 2023) is the
industry framework for the positional uncertainty of gyro-surveyed wellbores. This paper
reproduces its published test cases (SPE 90408 Appendix E), documents implementation
clarifications and reference-data inconsistencies, and provides an open-source reference
implementation validated against the standard wells.

*Keywords: wellbore position uncertainty, gyro survey, ISCWSA error model, SPE 90408,
covariance, directional surveying, MWD, reference implementation.*

---

## Making the Exact Wellbore Anti-Collision Boundary Practical

Jonathan Corcutt (2026). **DOI: [10.5281/zenodo.20976872](https://doi.org/10.5281/zenodo.20976872)**.

Wellbore anti-collision is governed by the ISCWSA separation rule (Sawaryn et al.,
SPE-187073-MS), which is conservative by construction. This paper presents an efficient,
validated, open implementation of the exact combined-ellipsoid boundary (a mesh-free
analytic Mahalanobis clearance), and quantifies the cost of the separation-rule
approximation — usable as a fast broadphase filter ahead of exact narrowphase checks.

*Keywords: anti-collision, wellbore clearance, separation factor, ISCWSA, Mahalanobis
distance, collision avoidance, uncertainty ellipsoid, directional drilling.*
