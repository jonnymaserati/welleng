---
title: "Reproducing the ISCWSA Gyro Error-Model Test Cases: Implementation Clarifications, Reference-Data Inconsistencies, and an Open-Source Reference Implementation"
author: "Jonathan Corcutt — Corcutt Beheer B.V., Wassenaar, Netherlands — ORCID 0009-0008-1953-7760"
date: "2026"
geometry: margin=2.2cm
fontsize: 10pt
header-includes: \usepackage{amssymb}
---

**Preprint — Version 1.0 (2026-06-26)**

---

## Abstract

The ISCWSA gyro error model (Torkildsen et al., SPE 90408, 2004; ISCWSA Error Model Rev 5.13, 2023) defines the industry framework for quantifying positional uncertainty of gyro-surveyed wellbores. Its published test cases — six example tool models on three standard wells, with tabulated position covariances (SPE 90408 Appendix E) — are the de-facto acceptance test for any implementation. Yet the ISCWSA Error Model Sub-Committee itself records (Rev 5.13, §7.3.1) that members "have struggled to exactly replicate these values," and no open-source implementation or per-error-source diagnostic set exists to arbitrate.

We present an independent, open-source implementation (welleng) whose minimum-curvature position-covariance propagation is verified exact against the ISCWSA MWD reference (relative tolerance $5\times10^{-5}$), and use it to reproduce the SPE 90408 Appendix E gyro covariances. We document the specific implementation details required to close the hardest cases — the re-gyrocompass carry at a near-vertical re-entry, minimum-curvature interpolation of the carried weight across the azimuth singularity, re-randomisation of random sources at re-initialisation (Rev 5.13 eqs 44–46), the true-versus-grid meridian-convergence frame, and the canted-accelerometer toolface-switching operator. With these, five of the six example models reproduce Appendix E within the paper's stated $\pm1\%$ / $\pm2$-unit inter-implementation band on every well checked, including the previously-intractable $I=110^\circ$ hybrid-gyro cells (north–east covariance $-147$ versus reference $-147$).

For the sixth model on the third well, an exhaustive permutation search (1584 combinations of every plausible modelling choice) demonstrates that **no self-consistent implementation can reproduce the published values**, because they are internally inconsistent: the north–east term requires the carried random seed to be simultaneously full-correlated and de-correlated; the east–east term is ordered backwards relative to a lower-noise model; and the north–north term lies below every physically-admissible carried-init treatment. We provide the open implementation, the validation harness, and per-error-source gyro diagnostics (which, to our knowledge, are not otherwise publicly available).

**Notation.** $I$ inclination, $A$ azimuth (true unless noted), $D$ measured depth, $\phi$ latitude, $\Omega$ Earth's rotation rate, $G$ gravity, $c$ tool running speed, $\sigma_{i,l}$ the magnitude of error source $i$ on leg $l$, $\varepsilon_i$ the error source, $f$ the gyro noise-reduction factor.

---

## 1. Introduction

Wellbore position uncertainty is computed industry-wide using the ISCWSA error model (Williamson, 2000; ISCWSA Rev 5.13, 2023), which propagates identified instrument error sources into a $3\times3$ position covariance matrix at each survey station. For magnetic (MWD) tools the model is mature, well-tested, and accompanied by published example workbooks and per-error-source diagnostics that let any implementer verify each term independently.

The **gyro** branch of the model is less mature. Gyro tools come in many sensor configurations (two- or three-axis accelerometers; one-, two- or three-axis gyros) and operate in two modes — *stationary* (gyrocompassing) and *continuous* — with the model transitioning between them at defined inclinations and, where the well drops back toward vertical, re-initialising (SPE 90408 Appendix C; Rev 5.13 §7). The authoritative numerical test cases are the six "example models" of SPE 90408 Appendix D, whose position covariances on the three ISCWSA standard wells are tabulated in Appendix E.

Two gaps motivate this work:

1. **Reproducibility.** Rev 5.13 §7.3.1 states plainly that committee members "have struggled to exactly replicate these values" and that a new test-definition document is intended. Several implementation details that materially affect the result are described only obliquely in the source paper.
2. **No open reference.** Unlike the MWD model, there is no open-source gyro implementation and no published per-error-source diagnostics; §7.1 advises users to "obtain appropriate models from their gyro service provider." Independent verification is therefore difficult.

This paper addresses both. We use welleng — an open-source (LGPL) Python library — whose propagation engine we first verify exact against the MWD reference, then apply to the gyro example models. We document each implementation detail needed to reproduce Appendix E, identify and prove inconsistencies in one model's reference data, and publish per-source diagnostics.

## 2. The propagation baseline

For each error source $i$, welleng forms a depth/inclination/azimuth weighting function per station and maps it to a north/east/vertical error vector via the minimum-curvature position Jacobian. Following Williamson (2000) (Rev 5.13 eq 6), the contribution of source $i$ at station $k$ on leg $l$ is

\begin{equation}
\mathbf{e}_{i,l,k} \;=\; \sigma_{i,l}\left(\frac{d\,\Delta\mathbf{r}_{k}}{d\mathbf{p}_{k}} + \frac{d\,\Delta\mathbf{r}_{k+1}}{d\mathbf{p}_{k}}\right)\frac{\partial \mathbf{p}_{k}}{\partial \varepsilon_{i}},
\end{equation}

i.e. the reading error at station $k$ propagates through **both** adjacent minimum-curvature legs $[k\!-\!1,k]$ and $[k,k\!+\!1]$ (the "$N\pm1$" / centred convention). The source covariance is then accumulated by propagation mode — systematic sources are fully correlated, random sources independent:

\begin{equation}
[C]^{\text{sys}} = \Big(\textstyle\sum_{k}\mathbf{e}_{i,l,k}\Big)\Big(\textstyle\sum_{k}\mathbf{e}_{i,l,k}\Big)^{\!\top},
\qquad
[C]^{\text{rand}} = \textstyle\sum_{k}\mathbf{e}_{i,l,k}\,\mathbf{e}_{i,l,k}^{\top}.
\end{equation}

Where a weighting function is singular in vertical hole (a $1/\sin I$ factor that the position Jacobian's $\sin I$ cancels, but which evaluates to NaN at exactly vertical), the position-space singular vector is substituted directly (Rev 5.13 §11.5, eqs 20–21):

\begin{equation}
\mathbf{e}_{i,l,k} = \sigma_{i,l}\,\frac{D_{k+1}-D_{k-1}}{2}\,
\big[\,w_{N}\;\; w_{E}\;\; w_{V}\,\big]^{\top},
\end{equation}

applied below a vertical-inclination limit of $0.0001^\circ$ (the ISCWSA workbook value).

To establish that any gyro discrepancy lies in the gyro layer and not in propagation, we validate the engine against the ISCWSA MWD Rev 5.1 example data: welleng reproduces the reference position covariance to **relative tolerance $5\times10^{-5}$** on every element of every station — including the off-diagonal north–east term and on the exact geometry of the standard wells. The propagation is therefore exact for the present purpose, and the gyro residuals reported below are attributable solely to the gyro error sources and their mode handling.

## 3. The validation: SPE 90408 Appendix E

We instantiate each of the six Appendix D example models (weighting functions from SPE 90408 Tables 1–10; magnitudes and mode parameters from Appendix D) and evaluate the position covariance on each ISCWSA standard well, comparing every element $[NN, NE, NV, EE, EV, VV]$ at each Appendix E checkpoint against the published value. The acceptance criterion is the paper's own inter-implementation standard: agreement within $\pm1\%$, or $\pm2$ units where the value is below 200 — the band within which the authors state the values were "verified … by independent implementations." Exact ($0\%$) agreement is neither expected nor claimed.

The six models span the configuration space: Model #1 (XY stationary gyro), #2 (XY accelerometer + external-reference init + Z continuous gyro), #3 (XY stationary$\to$continuous hybrid), #4 (canted XY accelerometer + XY stationary init + two continuous zones), #5 (XYZ stationary$\to$continuous), #6 (XYZ stationary).

The stationary XY-gyro azimuth weighting functions (SPE 90408 Table 4) carry the characteristic $1/\cos I$ that diverges as $I\to90^\circ$ — e.g. gyro bias-1 and g-dependent-4,

\begin{equation}
\frac{\partial A}{\partial \varepsilon_{\text{GB1}}} = \frac{\sin A}{\Omega\cos\phi\cos I},
\qquad
\frac{\partial A}{\partial \varepsilon_{\text{GD4}}} = \frac{\sin A\,\tan I}{\Omega\cos\phi},
\end{equation}

and the random gyro noise (with noise-reduction factor $f$),

\begin{equation}
\frac{\partial A}{\partial \varepsilon_{\text{GRN}}} = f\,\frac{\sqrt{1-\cos^{2}\!A\,\sin^{2}\!I}}{\Omega\cos\phi\cos I}.
\end{equation}

## 4. Implementation clarifications required to reproduce Appendix E

Five details, under-specified in the source paper, materially affect the result.

### 4.1 Mode switching and re-gyrocompassing

For a hybrid tool with a positive stationary init inclination (Models #3, #4), the well dropping below the init inclination switches the tool back to stationary mode and **de-initialises** the continuous survey (Appendix C boxes 2/3/9; Rev 5.13 §7.2). On the subsequent rebuild the tool **re-gyrocompasses**: the carried stationary-init azimuth error is recomputed at the *rebuild* azimuth, not the first-build azimuth. ISCWSA Standard Well #3 builds to $\sim50^\circ$, drops through vertical at $\sim2460$ m, then rebuilds past $90^\circ$ to $110^\circ$ at a different heading ($\approx283^\circ$). Freezing the carry at the first crossing produces the wrong **sign** on $NE$; re-gyrocompassing at the rebuild azimuth corrects it.

### 4.2 Interpolating the carried weight across the vertical singularity

The carried-init weight must be evaluated at the gate inclination on the rebuild. Where the survey straddles a near-vertical station (whose azimuth is an undefined placeholder), **linear interpolation of the azimuth-dependent weight is invalid** — it blends the placeholder azimuth into a $\sin A/\cos A$ term. The degeneracy-safe procedure interpolates the survey **direction vector** $\hat{\mathbf{t}}(I,A)=(\sin I\cos A,\,\sin I\sin A,\,\cos I)$ along the minimum-curvature (great-circle) arc,

\begin{equation}
\hat{\mathbf{t}}(t) = \frac{\sin\!\big((1-t)\,\theta\big)}{\sin\theta}\,\hat{\mathbf{t}}_1 + \frac{\sin\!\big(t\,\theta\big)}{\sin\theta}\,\hat{\mathbf{t}}_2,
\qquad \theta = \arccos(\hat{\mathbf{t}}_1\!\cdot\!\hat{\mathbf{t}}_2),
\end{equation}

to the gate inclination and re-evaluates the weight at the recovered azimuth. (At the pole the great circle is a constant-azimuth meridian, so this recovers the rebuild azimuth exactly.)

### 4.3 Re-randomisation of random sources at re-initialisation

Rev 5.13 §7.3 point 14 and eqs 44–46 state that a **random** static source becomes systematic when frozen into a continuous section but **re-randomises at each re-initialisation**: two continuous sections separated by a re-init are independent and root-sum-square rather than accumulating as one correlated sum. The covariance is therefore

\begin{equation}
[C]_K \;=\; \sum_{s\,\in\,\text{sections}}\Big(\textstyle\sum_{k\in s}\mathbf{e}_{i,l,k}\Big)\Big(\textstyle\sum_{k\in s}\mathbf{e}_{i,l,k}\Big)^{\!\top}
\quad\text{(correct)},
\end{equation}

rather than the single fully-correlated sum $\big(\sum_{k}\mathbf{e}_{i,l,k}\big)\big(\sum_{k}\mathbf{e}_{i,l,k}\big)^{\top}$ across both sections. This applies to **both** the carried random init seed and the **continuous random-walk** term, whose per-station recurrence (SPE 90408 Table 7), writing $\bar{I}_i \equiv \tfrac{1}{2}(I_{i-1}+I_i)$ for the mean inclination over the leg, is for drift and random walk respectively

\begin{align}
h_i &= h_{i-1} + \frac{\Delta D_i}{c\,\sin \bar{I}_i}, \\[4pt]
h_i &= \sqrt{\,h_{i-1}^{2} + \frac{\Delta D_i}{c\,\sin^{2}\! \bar{I}_i}\,}.
\end{align}

(Note the $\sqrt{\,\cdot\,}$ accumulates *inside* the coefficient $h_i$, so the azimuth error $\sigma\,h_i$ is **linear** in the random-walk magnitude $\sigma$ — dimensionally forced, since $[\sigma]=\text{deg}/\sqrt{\text{hr}}$ and $[h_i]=\sqrt{\text{hr}}$.) Implementing the RSS for both terms — while leaving the systematic biases fully correlated — closes the $I=110^\circ$ cells of Model #3 on Well #3 exactly ($NE=-147$ vs reference $-147$; $EE=2127$ vs $2129$).

### 4.4 True-versus-grid meridian convergence

Appendix E reports gyro covariances in the **UTM grid** frame, whereas the gyro weighting functions are referenced to **true** north (Rev 5.13 §7.2); the two differ by the meridian convergence $\gamma$, applied as both an input azimuth shift and an output frame rotation:

\begin{equation}
A_{\text{true}} = A_{\text{grid}} + \gamma,
\qquad
C_{\text{grid}} = R(-\gamma)\,C_{\text{true}}\,R(-\gamma)^{\top},
\qquad
R(\theta)=\begin{bmatrix}\cos\theta & -\sin\theta & 0\\ \sin\theta & \cos\theta & 0\\ 0 & 0 & 1\end{bmatrix}.
\end{equation}

The dominant effect is the input shift (it enters the $\sin A/\cos A$ weights); rotating only the output covariance moves the result the wrong way. ISCWSA publishes the UTM zone and latitude but not the surface longitude, so $\gamma$ is not directly recoverable; we back-calculate it (§6).

### 4.5 Canted-accelerometer toolface switching

Model #4's accelerometers are canted $\gamma_c=17^\circ$ from the tool axis; inclination weighting uses

\begin{equation}
\frac{\partial I}{\partial \varepsilon_{\text{AXY-B}}} = \frac{1}{G\cos\!\big(I - k\,\gamma_c\big)},
\qquad
k = \begin{cases}+1 & I\le 90^\circ\\[-1pt] -1 & I>90^\circ\end{cases}
\end{equation}

(SPE 90408 Table 2; Rev 5.13 §7.1). On wells exceeding $90^\circ$ (Well #3, to $110^\circ$) the operator must flip, or the weight passes through a spurious singularity at $I=90^\circ+\gamma_c=107^\circ$.

## 5. Methodology: exhaustive permutation search

Where a residual could in principle arise from any of several modelling choices, we avoid guessing by searching the full combination space. Each independent lever is made toggleable — random-walk correlation, random-init-seed correlation, secondary-zone random-walk correlation, carry-interpolation method, systematic-carry correlation, and convergence $\gamma$ over a continuous range — and we evaluate every combination ($N=1584$) against the reference, recording the number of cells in band and whether the combination uses only physically-admissible settings. A residual is declared irreducible only when no combination of admissible settings reaches the band — distinguishing a genuine reference inconsistency from an implementation choice not yet found.

## 6. Results

With the clarifications of §4, five of the six example models reproduce Appendix E within the acceptance band on every well checked (Table 1).

Table: Validation status against SPE 90408 Appendix E ($\pm1\%$ / $\pm2$u band). $\checkmark$ = all checked channels in band.

| Model | Well #1 | Well #2 | Well #3 |
|---|:---:|:---:|:---:|
| #1 XY stationary | $\checkmark$ | $\checkmark$ ($\gamma$) | $\checkmark$ |
| #2 XY accel + ext-ref + Z cont. | $\checkmark$ | $\checkmark$ | — |
| #3 XY stationary$\to$continuous | $\checkmark$ | $\checkmark$ | $\checkmark$ (closed at $I{=}110^\circ$) |
| #4 cant + two-zone | $\checkmark$ | $\checkmark$ | inconsistent (§7) |
| #5 XYZ continuous | $\checkmark$ | — | — |
| #6 XYZ stationary | $\checkmark$ | — | — |

**Model #3 on Well #3** — the hardest case, spanning build/drop/rebuild to $I=110^\circ$ — closes on **all six channels at all five depths**, requiring §4.1–4.3 together (Table 2).

Table: Model #3 on Well #3 at the two $I=110^\circ$ checkpoints (welleng / reference); units m².

| Channel | 3720 m | 4030 m |
|---|---|---|
| $NN$ | 1225 / 1229 | 1443 / 1447 |
| $NE$ | 317 / 318 | $-147$ / $-147$ |
| $NV$ | $-25$ / $-25$ | $-27$ / $-27$ |
| $EE$ | 2199 / 2202 | 2127 / 2129 |
| $EV$ | $-1$ / $-1$ | $-1$ / $-1$ |
| $VV$ | 40 / 40 | 43 / 43 |

**Well #2** reproduces with a single recovered meridian convergence $\gamma=1.25^\circ$: one constant brings **all 180 Well #2 cells** (6 models $\times$ 5 depths $\times$ 6 elements) into band for $\gamma\in[1.0^\circ,1.5^\circ]$, inside zone-15N's physical bound ($\pm1.4^\circ$ at $28^\circ$N) — an over-determined fit of a single parameter, strong evidence that $\gamma$ is the true convergence rather than a per-cell adjustment (Table 3). We request that ISCWSA publish the per-well surface convergence so this becomes an external check.

Table: Well #2 cells in band (of 180) versus assumed meridian convergence $\gamma$.

| $\gamma$ (°) | 0.0 | 0.5 | 0.8 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|
| cells in band | 175 | 175 | 177 | **180** | **180** | **180** | 177 |

## 7. Reference-data inconsistencies: Model #4 on Well #3, $I=110^\circ$

Model #4 on Well #3 at $I=110^\circ$ is the sole case that does not close. The permutation search of §5 reaches at most **14 of 18 cells in band, and only with known-incorrect settings** (the invalid linear carry interpolation of §4.2 with full correlation). With correct physics the residuals are not an implementation gap but three demonstrable inconsistencies in the published values (Table 4); welleng's values are the internally-consistent computation.

Table: Model #4 on Well #3 (welleng / reference); units m².

| Channel | 3000 m | 3720 m | 4030 m |
|---|---|---|---|
| $NN$ | 94 / 92 | 1335 / 1196 | 1574 / 1408 |
| $NE$ | 57 / 167 | 464 / 1444 | $-34$ / 1094 |
| $EE$ | 2426 / 2488 | 2391 / 2382 | 2220 / 1423 |

Decompose the north–east covariance into a non-seed part and the carried random-seed part, $C_{NE} = C_{NE}^{\text{non-seed}} + C_{NE}^{\text{seed}}$ (Table 5):

Table: North–east seed-contribution decomposition, Model #4 on Well #3 (m²). The contribution the published value requires flips correlation regime between depths — impossible for a single source.

| depth (m) | non-seed | seed (full-corr.) | seed (de-corr.) | seed *required* |
|---|---|---|---|---|
| 3000 | 159.2 | $+6.4$ | $+0.1$ | $+7.8$ ($\approx$ full) |
| 3720 | 1384.1 | $+55.2$ | $-0.4$ | $+59.9$ ($\approx$ full) |
| 4030 | 1097.4 | $+43.1$ | $-17.5$ | $-3.4$ ($\approx$ de-corr.) |

1. **North–east is internally self-contradictory.** The published cells require the *same* physical seed to be **full-correlated** at 3000 m and 3720 m ($C_{NE}^{\text{seed}}\approx+7.8,\,+59.9$) yet **de-correlated** at 4030 m ($C_{NE}^{\text{seed}}\approx-3.4$) — mutually exclusive for one source. Model #3, whose carried seed is internally consistent, closes exactly under the same engine; Model #4 is $\sim4\times$ more sensitive because its noise-reduction factor $f=1.0$ gives a seed covariance four times larger than Model #3's ($1.0^{2}$ vs $0.5^{2}$).

2. **East–east is ordered backwards.** welleng gives Model #4 $EE$ at 4030 m as $2220 > $ Model #3's $2127$ — the physically expected ordering, since Model #4's frozen init seed is the larger. The published values invert this: Model #4 $1423 < $ Model #3 $2129$.

3. **North–north is anomalously low.** The published $NN$ lies *below every* physically-admissible carried-init treatment. Correct re-gyrocompassing at the rebuild azimuth (§4.1–4.2) gives $+11.8\%$; the only way to reach the published value is the invalid linear-across-vertical interpolation.

We conclude that the Model #4 / Well #3 $I>90^\circ$ reference values are unreliable and should be re-derived or withdrawn. In welleng the $NE$ term is pinned to the internally-consistent computed value (with this finding cited) and $NN/EE$ recorded as known reference inconsistencies.

## 8. An open reference implementation and per-source diagnostics

The implementation, the validation harness, and the permutation-search script are open-source. Because the engine exposes each error source's per-station north/east/vertical contribution, we can publish **per-error-source gyro diagnostics** (which, to our knowledge, are not otherwise publicly available): the individual contribution of every error source at every checkpoint, for all six example models on all three wells. This is the gyro analogue of the MWD diagnostics workbook and directly addresses §7.3.1: an implementer can verify each source term independently, not only the summed total.

## 9. Clarifications and corrections

This paper establishes the following, relative to the published gyro test cases:

1. The re-gyrocompass / re-initialisation rule (§4.1), and the requirement to re-evaluate the carried weight at the rebuild azimuth via a degeneracy-safe (direction-vector / minimum-curvature) interpolation (§4.2).
2. Random sources re-randomise at re-initialisation — for **both** the carried init seed and the continuous random walk (§4.3, eqs 44–46).
3. The grid-frame transformation (§4.4) depends on the per-well surface meridian convergence, which the published test cases do not provide.
4. The Model #4 / Well #3 $I>90^\circ$ reference values are internally inconsistent and require re-derivation or withdrawal (§7).
5. Per-error-source gyro diagnostics (§8) are published with this paper.

## 10. Conclusion

An independent open-source implementation, with propagation verified exact against the MWD reference, reproduces five of the six SPE 90408 gyro example models within the published inter-implementation band on every well checked — including the previously-intractable $I=110^\circ$ hybrid case — once five under-specified implementation details are made explicit. The sixth model on the third well cannot be reproduced by any self-consistent implementation because its published values are internally inconsistent, as proved by exhaustive search. We provide the implementation, the harness, and per-source diagnostics, and document the specific clarifications and corrections needed to reproduce — and where necessary correct — the published gyro test cases.

---

## References

- Torkildsen, T., Håvardstein, S.T., Weston, J.L., Ekseth, R. (2004). *Prediction of Wellbore Position Accuracy When Surveyed With Gyroscopic Tools.* SPE 90408. <https://doi.org/10.2118/90408-MS>
- Williamson, H.S. (2000). *Accuracy Prediction for Directional Measurement While Drilling.* SPE Drilling & Completion 15(4). SPE 67616-PA. <https://doi.org/10.2118/67616-PA>
- Ekseth, R. (1998). *Uncertainties in Connection with the Determination of Wellbore Positions.* PhD thesis, Norwegian University of Science and Technology (NTNU). ISCWSA-hosted copy: <https://www.iscwsa.net/files/796/>; corrections (errata): <https://www.iscwsa.net/files/675/>
- Ekseth, R., Torkildsen, T., Brooks, A., Weston, J., Nyrnes, E., Wilson, H., Kovalenko, K. (2010). *High-Integrity Wellbore Surveying.* SPE Drilling & Completion 25(4). SPE 133417-PA. <https://doi.org/10.2118/133417-PA>
- ISCWSA (2023). *Definition of the ISCWSA Error Model, Rev 5.13.* Industry Steering Committee on Wellbore Survey Accuracy. <https://www.iscwsa.net/media/files/files/64bd61c2/definition-of-iscwsa-error-model-v5-13.pdf>
- Copsegrove, C., Grindrod, S. (2020). *ISCWSA Error Model — Test Profile Differences* (CDR-SM-03). <https://www.iscwsa.net/files/659/>
- Corcutt, J. *welleng: open-source well engineering.* <https://github.com/jonnymaserati/welleng> (software DOI on deposit).

---

*Reproducibility: all results are produced by the open-source welleng test suite (`tests/test_spe90408_appendix_e.py`) and the permutation-search script; version/commit cited on deposit.*
