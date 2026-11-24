# qiskit-lombscargle-zne: Equal-Budget, Out-of-Family & Optimizer-Instability ZNE Audit (v3.0)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24121693.svg)](https://doi.org/10.5281/zenodo.24121693)
[![CI](https://github.com/bruno-olivaes/qiskit-lombscargle-zne/actions/workflows/ci.yml/badge.svg)](https://github.com/bruno-olivaes/qiskit-lombscargle-zne/actions/workflows/ci.yml)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.5+-6133BD.svg)](https://qiskit.org/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF_Manuscript-red.svg)](manuscript/main.pdf)
[![Aer](https://img.shields.io/badge/Qiskit_Aer-0.17+-blue.svg)](https://qiskit.github.io/qiskit-aer/)
[![Tests](https://img.shields.io/badge/Pytest-7%2F7_Passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-source quantum error mitigation (QEM) benchmark suite for **Qiskit 2.5+ & Qiskit Aer**, auditing **Zero-Noise Extrapolation (ZNE)** under matched equal-shot budgets, out-of-family noise channels, non-linear optimizer instability diagnostics, and finite Pauli-twirl frame errors.

---

## 1. Scientific Scope & v3.0 Audit Controls

In Version 3.0, the suite directly resolves three core methodological controls in `src/lombscargle_zne.py` and `benchmarks/fair_zne_study.py`:
1. **Strict Equal-Shot-Budget & Twirl-Only Baselines (`Unmitigated_Equal_Budget`, `Pauli_Twirl_Only_Equal_Budget`):** Compares ZNE against spending the full shot budget ($32{,}768$ in 8-scale 1Q, $24{,}576$ in 6-scale 1Q, $24{,}000$ in 4Q TFIM) on a single unmitigated circuit ($\lambda=1$) or on 12 $\lambda=1$ Pauli-twirled variants, while tracking compiled circuit overhead (`n_circuits_per_seed`: $1$ vs.\ $4$ vs.\ $6$ vs.\ $12$ vs.\ $48$).
2. **Stability-Gated Auto Selection (`Auto_LombScargle_ZNE`):** `LombScargleZNE._fit_diagnostics` flags `at_bounds` ($2\%$ bound clamping) and `cond_number > 1e4` (`unstable_fit`). `method='auto'` triggers harmonic fitting via $\mathrm{FAP} < 0.05$ or $R^2_{\mathrm{peak}} \ge 0.60$ ($N \le 12$) and rejects unstable fits by falling back to `Local_Linear_ZNE`.
3. **Out-of-Family 1Q Stress Tests (A1–A4) & Noisy Pauli Twirl Frames (`x,y,z`):** Evaluates short-scale schedules, purely monotonic amplitude+depolarizing damping, bi-chromatic quasi-static Gaussian $1/f$ drift, and 4Q TFIM when inserted Clifford Pauli frame gates carry $p_1 = 10^{-3}$ depolarizing and $R_Z(0.25\epsilon)$ coherent errors.

---

## 2. Empirical Results

### Regime A: 1-Qubit Echo Stress Test (A1–A4, 25 Seeds, $E_{\mathrm{ideal}} = +1.0000$)

| Method | A1: Deep Mono-Mode ($s\in\{1..17\}$) | A2: Short Coherent ($s\in\{1..6\}$) | A3: Amp. Damping + Depol. | A4: Bi-Chromatic + $1/f$ Drift |
| :--- | :---: | :---: | :---: | :---: |
| **Unmitigated (1 scale, 4k shots)** | $0.0491 \pm 0.0011$ | $0.0496 \pm 0.0009$ | $0.0318 \pm 0.0007$ | $0.0892 \pm 0.0014$ |
| **Unmitigated Equal Budget (1 circ.)** | $0.0482 \pm 0.0004$ | $0.0487 \pm 0.0004$ | $0.0324 \pm 0.0002$ | $0.0904 \pm 0.0009$ |
| **Global Quadratic ZNE** | $0.3199 \pm 0.0011$ | **$0.0349 \pm 0.0030$** | $0.0046 \pm 0.0006$ | **$0.0387 \pm 0.0062$** |
| **Local Linear ZNE (4 circ.)** | $0.1197 \pm 0.0018$ | $0.0947 \pm 0.0018$ | $0.0044 \pm 0.0006$ | $0.1667 \pm 0.0041$ |
| **Exponential ZNE** | $0.5766 \pm 0.0014$ (`inst=0.00`) | $0.2758 \pm 0.0026$ (`inst=0.00`) | **$0.0037 \pm 0.0004$** (`inst=0.00`) | $0.6979 \pm 0.0063$ (`inst=0.00`) |
| **Damped Harmonic ZNE (Forced)** | **$0.0084 \pm 0.0012$** (`inst=0.00`) | $0.3601 \pm 0.1899$ (`inst=1.00`) | $0.2956 \pm 0.1612$ (`inst=1.00`) | $0.2007 \pm 0.0129$ (`inst=1.00`) |
| **Auto Lomb–Scargle ZNE** | **$0.0084 \pm 0.0012$** (`osc_sel=1.00`) | $0.0947 \pm 0.0018$ (`osc_sel=0.00`) | $0.0044 \pm 0.0006$ (`osc_sel=0.00`) | $0.1667 \pm 0.0041$ (`osc_sel=0.00`) |

* **Finding (Regime A):** Harmonic ZNE beats `Unmitigated_Equal_Budget` decisively **only** in the in-family A1 niche (**$-82.6\%$**, $z = +31.74\,\mathrm{SE}$). Across A2, A3, and A4, forced Harmonic ZNE suffers **$100\%$ optimizer instability**, while `Auto_LombScargle_ZNE` detects instability and falls back cleanly ($0\%$ instability)—matching monotonic mitigation on A3 ($z = +40.47\,\mathrm{SE}$) while documenting that simple local linear fallback loses to `Unmitigated_Equal_Budget` on A2 and A4.

---

### Regime B: 4-Qubit TFIM Trotter Circuit (3 Steps, 18 `cx` Gates, 20 Seeds, 24k Total Shots/Seed)

| Method *(Circuits / Shots)* | $\epsilon = 0.00$ | $\epsilon = 0.04$ | $\epsilon = 0.08$ | $\epsilon = 0.14$ | $\epsilon = 0.08$ **(Noisy Twirl Frames)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Unmitigated** ($1$ / $4\mathrm{k}$) | $0.0273 \pm 0.0027$ | $0.1484 \pm 0.0027$ | $0.2872 \pm 0.0032$ | $0.5347 \pm 0.0028$ | $0.2848 \pm 0.0026$ |
| **Unmitigated Equal Budget** ($1$ / $24\mathrm{k}$) | $0.0265 \pm 0.0014$ | $0.1450 \pm 0.0008$ | $0.2883 \pm 0.0017$ | $0.5322 \pm 0.0011$ | $0.2911 \pm 0.0012$ |
| **Pauli Twirl Only Eq. Budget** ($12$ / $24\mathrm{k}$) | $0.0259 \pm 0.0013$ | $0.0565 \pm 0.0037$ | $0.0991 \pm 0.0045$ | $0.1554 \pm 0.0084$ | $0.1167 \pm 0.0065$ |
| **Local Linear ZNE** ($4$ / $16\mathrm{k}$) | $0.0194 \pm 0.0036$ | $0.0407 \pm 0.0051$ | $0.0735 \pm 0.0056$ | $0.1002 \pm 0.0056$ | $0.0817 \pm 0.0063$ |
| **Quadratic ZNE** ($6$ / $24\mathrm{k}$) | $0.0396 \pm 0.0076$ | $0.1027 \pm 0.0089$ | $0.2086 \pm 0.0101$ | $0.2006 \pm 0.0086$ | $0.2370 \pm 0.0086$ |
| **Exponential ZNE** ($6$ / $24\mathrm{k}$) | **$0.0156 \pm 0.0027$** | $0.3763 \pm 0.0157$ | $0.5405 \pm 0.0017$ (`1.00`) | $0.7876 \pm 0.0012$ (`1.00`) | $0.5413 \pm 0.0013$ (`1.00`) |
| **Pauli Twirl + Linear ZNE** ($48$ / $24\mathrm{k}$) | $0.0243 \pm 0.0035$ | **$0.0347 \pm 0.0042$** | $0.0766 \pm 0.0105$ | $0.1212 \pm 0.0158$ | **$0.0383 \pm 0.0070$** |
| **Pauli Twirl + Exp ZNE** ($48$ / $24\mathrm{k}$) | $0.0307 \pm 0.0047$ | $0.0363 \pm 0.0044$ | **$0.0714 \pm 0.0082$** (`0.25`) | **$0.0997 \pm 0.0136$** (`0.20`) | $0.0406 \pm 0.0069$ |
| **Damped Harmonic ZNE (Forced)** ($6$ / $24\mathrm{k}$) | $1.2701$ (`inst=1.00`) | $2.2868$ (`inst=1.00`) | $0.3042$ (`inst=1.00`) | $0.5550$ (`inst=1.00`) | $0.2493$ (`inst=1.00`) |
| **Auto Lomb–Scargle ZNE** ($6$ / $24\mathrm{k}$) | $0.0194 \pm 0.0036$ | $0.0407 \pm 0.0051$ | $0.0735 \pm 0.0056$ | $0.1002 \pm 0.0056$ | $0.0817 \pm 0.0063$ |

* **Finding (Regime B):** Pauli Twirling + Linear/Exp ZNE outperforms `Unmitigated_Equal_Budget` by **$-75.2\%$ to $-81.3\%$** across $\epsilon \in [0.04, 0.14]$ and maintains **$-86.8\%$** ($z = +35.55\,\mathrm{SE}$) when Pauli frame gates (`x, y, z`) themselves incur realistic depolarizing and coherent errors—at the cost of $48$ compiled circuits/seed vs.\ $12$ (`Pauli_Twirl_Only_Equal_Budget`) or $1$ (`Unmitigated_Equal_Budget`).

---

## 3. Quickstart & Reproducibility

```bash
pip install qiskit qiskit-aer numpy scipy matplotlib pytest
pytest tests/ -v
python benchmarks/fair_zne_study.py
```

## License

MIT License. See [LICENSE](LICENSE) for details.
