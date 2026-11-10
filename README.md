# qiskit-lombscargle-zne: Spectral Diagnostic & Pre-Registered ZNE / Pauli Twirling Benchmark Suite (v2.0)

[![Qiskit](https://img.shields.io/badge/Qiskit-2.5+-6133BD.svg)](https://qiskit.org/)
[![Paper PDF](https://img.shields.io/badge/Paper-PDF_Manuscript-red.svg)](manuscript/main.pdf)
[![Aer](https://img.shields.io/badge/Qiskit_Aer-0.17+-blue.svg)](https://qiskit.github.io/qiskit-aer/)
[![Tests](https://img.shields.io/badge/Pytest-5%2F5_Passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An open-source quantum error mitigation (QEM) research and benchmarking repository for **Qiskit 2.5+ & Qiskit Aer**, investigating **Zero-Noise Extrapolation (ZNE)** under combined **coherent unitary over-rotation ($R_Z / R_{ZZ}$ crosstalk)** and **stochastic depolarizing noise**.

---

## 1. Scientific Scope & Pre-Registered Audit (v2.0)

In Version 2.0, this repository was upgraded from a single-qubit demonstration into a **pre-registered head-to-head methodological benchmark** (`benchmarks/fair_zne_study.py`) implementing:
1. **Gate-Level Non-Uniform Fractional Unitary Folding (`fold_circuit_fractional`):** Replaces subsets of 2-qubit and 1-qubit gates $G \to G G^\dagger G$ (weighted $10:1$ by hardware error rate and protected by transpiler barriers) to achieve fine-grained non-uniform scale factors $\lambda \in [1.0, 3.5]$ while preserving exact logical unitary equivalence ($U_{\mathrm{folded}} = U_{\mathrm{ideal}}$).
2. **Exact 2-Qubit Clifford Pauli Twirling (`pauli_twirl_circuit`):** Implements randomized compiling across all 16 two-qubit Pauli conjugations $(P'_c \otimes P'_t)\,\mathrm{CX}\,(P_c \otimes P_t) = \pm \mathrm{CX}$, converting coherent unitary over-rotations around `cx` gates into stochastic Pauli channels prior to ZNE extrapolation under an **identical total shot budget ($N_{\mathrm{total}} = 24,000$ shots per seed)**.
3. **Pre-Registered $2\text{-}\mathrm{SE}$ Decision Rule:** Damped Harmonic / Lomb–Scargle Spectral ZNE (`LombScargleZNE`) is evaluated against standard baselines (**Local Linear ZNE**, **Quadratic Richardson ZNE**, **Exponential ZNE**, and **Pauli Twirling + Linear/Exponential ZNE**) across $M \ge 20$ random seeds. A spectral niche is confirmed *only if* its mean absolute error (MAE) beats the best standard baseline by at least $2\times$ the combined standard error ($z > +2.0\,\mathrm{SE}$).

---

## 2. Empirical Results: Where Spectral ZNE Works vs. Where Pauli Twirling Dominates

### Regime A: 1-Qubit Coherent Idle / Stark-Shift Echo (`H - Id^s - H`, Deep Scales $s \in [1, 17]$, 25 Seeds)
In a single-qubit Ramsey/echo sequence subject to systematic phase drift $R_Z(\theta = 0.22\text{ rad/layer})$ and depolarizing rate $p = 0.025$ across 8 non-uniform scale factors $s \in \{1, 2, 3, 5, 7, 10, 13, 17\}$ (ideal $\langle Z \rangle = +1.0000$), the expectation curve follows an exact single-frequency damped cosine $E(s) = e^{-\gamma s}\cos(\omega s)$:

| Mitigation Method | Mean Absolute Error ($\mathrm{MAE} \pm \mathrm{SE}$) | Error vs. Unmitigated | Error vs. Global Quadratic ZNE |
| :--- | :---: | :---: | :---: |
| **Unmitigated ($s=1$)** | $0.0486 \pm 0.0009$ | Baseline | — |
| **Global Quadratic ZNE ($s \in [1, 17]$)** | $0.3195 \pm 0.0015$ | $+557\%$ (Diverges) | Baseline |
| **Local Linear ZNE ($s \in [1, 5]$)** | $0.1215 \pm 0.0015$ | $+150\%$ | $-62.0\%$ |
| **Exponential ZNE ($s \in [1, 17]$)** | $0.5770 \pm 0.0016$ | $+1087\%$ (Diverges) | $+80.6\%$ |
| **Damped Harmonic / Lomb–Scargle ZNE** | **$0.0071 \pm 0.0011$** | **$-85.4\%$** | **$-97.8\%$** |

* **Finding (Regime A):** When probing a single coherent oscillation mode over deep scale factors ($s \le 17$, spanning $> 1/2$ period), polynomial and exponential extrapolators fail catastrophically due to sign inversions, whereas **Damped Harmonic / Lomb–Scargle ZNE** accurately locks onto the spectral peak ($\omega \approx 0.22\text{ rad/scale}$) and reduces estimation error to $0.0071 \pm 0.0011$.

---

### Regime B: 4-Qubit Transverse-Field Ising Model (TFIM) Trotter Circuit (3 Steps, 18 `cx` Gates, 20 Seeds)
We evaluate a 4-qubit entangled TFIM Trotter circuit (Measuring $\langle Z_0 \rangle$, ideal $E_{\mathrm{ideal}} = -0.408093$) across 6 non-uniform fractional folding scales $\lambda \in \{1.000, 1.398, 1.796, 2.294, 2.851, 3.498\}$ under depolarizing noise ($p_1 = 0.001, p_2 = 0.012$) and coherent 2-qubit over-rotations $R_{ZZ}(\epsilon)\cdot R_Z(\epsilon)$ for $\epsilon \in \{0.00, 0.04, 0.08, 0.14\}\text{ rad}$ ($N_{\mathrm{total}} = 24,000$ shots/seed for every method):

| Method ($N_{\mathrm{tot}} = 24\mathrm{k}$ shots) | $\epsilon = 0.00\text{ rad}$ (Depol. Only) | $\epsilon = 0.04\text{ rad}$ (Moderate Coherent) | $\epsilon = 0.08\text{ rad}$ (Strong Coherent) | $\epsilon = 0.14\text{ rad}$ (Extreme Coherent) |
| :--- | :---: | :---: | :---: | :---: |
| **Unmitigated ($\lambda=1$)** | $0.0294 \pm 0.0036$ | $0.1484 \pm 0.0033$ | $0.2898 \pm 0.0032$ | $0.5308 \pm 0.0033$ |
| **Local Linear ZNE ($\lambda \le 2.29$)** | $0.0195 \pm 0.0029$ | $0.0375 \pm 0.0052$ | $0.0726 \pm 0.0055$ | **$0.0959 \pm 0.0041$** |
| **Quadratic Richardson ZNE** | $0.0380 \pm 0.0062$ | $0.0952 \pm 0.0097$ | $0.2088 \pm 0.0100$ | $0.2098 \pm 0.0100$ |
| **Exponential ZNE (Untwirled)** | **$0.0154 \pm 0.0019$** | $0.3601 \pm 0.0172$ | $0.5390 \pm 0.0011$ | $0.7858 \pm 0.0010$ |
| **Pauli Twirling + Linear ZNE** | $0.0213 \pm 0.0032$ | **$0.0285 \pm 0.0065$** | **$0.0551 \pm 0.0092$** | $0.1211 \pm 0.0169$ |
| **Pauli Twirling + Exponential ZNE** | $0.0263 \pm 0.0038$ | $0.0292 \pm 0.0047$ | $0.0562 \pm 0.0089$ | $0.1182 \pm 0.0123$ |
| **Damped Harmonic ZNE (Forced 4-Param)** | $1.3431 \pm 0.1590$ | $2.2868 \pm 0.0000$ | $0.3650 \pm 0.0643$ | $0.5578 \pm 0.0018$ |
| **Pre-Registered $2\text{-}\mathrm{SE}$ Niche Confirmed?** | **False** ($z = -8.35$) | **False** ($z = -347.2$) | **False** ($z = -4.77$) | **False** ($z = -103.7$) |

### Key Methodological Conclusions
1. **Why Pauli Twirling + Linear/Exponential ZNE Wins on Multi-Qubit Circuits:** At moderate-to-strong coherent over-rotations ($\epsilon \in [0.04, 0.08]\text{ rad}$), untwirled Exponential ZNE breaks down ($\mathrm{MAE} = 0.36\text{–}0.54$) because coherent errors violate pure exponential decay. Applying **2-qubit Clifford Pauli Twirling** converts the coherent `cx` over-rotations into an effective depolarizing channel, restoring **Exponential ZNE** ($\mathrm{MAE} = 0.0292\text{–}0.0562$) and **Linear ZNE** ($\mathrm{MAE} = 0.0285\text{–}0.0551$, an **81% error reduction** over unmitigated execution under the exact same 24,000-shot budget).
2. **Why Forced Damped Harmonic ZNE Fails on Multi-Qubit Short-Scale Schedules:** In an entangled multi-qubit Trotter circuit, coherent gate errors couple multiple non-commuting Pauli pathways rather than producing a single global sinusoid, and standard ZNE schedules ($\lambda \in [1.0, 3.5]$, 6 scale points) span less than half a period, making a 4-parameter damped sinusoid (`e0, gamma, omega, phi`) severely ill-conditioned. Thus, Lomb–Scargle analysis should be used as a **coherent-drift diagnostic tool** on calibration/echo sequences (Regime A), while **Pauli Twirling + Local Linear/Exponential ZNE** should be used for multi-qubit algorithmic mitigation (Regime B).

---

## 3. Quickstart & Reproducibility

```bash
# Install dependencies
pip install qiskit qiskit-aer numpy scipy matplotlib pytest

# Run the unit test suite (verifies exact unitary equivalence of integer folding, fractional folding, and Pauli Twirling)
pytest tests/

# Run the pre-registered head-to-head study (generates results/fair_zne_study.json and results/fair_zne_comparison.png)
python benchmarks/fair_zne_study.py
```

---

## 4. Repository Structure

```
qiskit-lombscargle-zne/
├── src/
│   ├── lombscargle_zne.py              # Fractional folding, Pauli Twirling & ZNE extrapolators
│   └── __init__.py
├── benchmarks/
│   ├── fair_zne_study.py               # Pre-registered v2.0 1q Echo & 4q TFIM benchmark (20-25 seeds)
│   ├── run_zne_benchmark.py            # Legacy v1 single-qubit benchmark script
│   └── benchmark_zne_comparison.png    # Generated v2.0 head-to-head comparison plot
├── results/
│   ├── fair_zne_study.json             # Complete numerical benchmark outputs & decision-rule metrics
│   └── fair_zne_comparison.png         # Publication figure
├── tests/
│   └── test_lombscargle_zne.py         # Pytest unit tests (5/5 passing)
└── README.md
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
