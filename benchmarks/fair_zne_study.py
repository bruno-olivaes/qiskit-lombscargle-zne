# -*- coding: utf-8 -*-
"""
Pre-Registered Head-to-Head Quantum Error Mitigation Study (`fair_zne_study.py`):
Standard Polynomial/Exponential ZNE & Pauli Twirling vs. Spectral/Harmonic ZNE.

PRE-REGISTERED DECISION RULE:
-----------------------------
Spectral / Damped-Harmonic ZNE (`LombScargleZNE`) is considered to have a statistically
valid niche in a given circuit/noise regime ONLY IF its mean absolute estimation error
|E_est(0) - E_ideal| across M >= 20 independent random seeds is strictly lower than the
best standard baseline method (Local Linear ZNE, Quadratic Richardson ZNE, Exponential ZNE,
and Pauli Twirling + Linear/Exponential ZNE under the exact same total shot budget) by at
least 2 combined standard errors (2 * SE). Otherwise, the null hypothesis (that standard
ZNE / Pauli Twirling dominates) holds and is reported transparently.

Circuits Evaluated:
-------------------
1. 1-Qubit Coherent Idle Echo (`H - Id^s - H`) with unitary over-rotation RZ(theta=0.22)
   and depolarizing rate p=0.025 (re-evaluating the v1 benchmark across 25 seeds).
2. 4-Qubit Transverse-Field Ising Model (TFIM) Trotter circuit (3 Trotter steps, brick-wall
   CX layers) with non-uniform fractional gate folding (`lambda` in [1.0, 3.5]) across
   coherent 2-qubit over-rotations RZZ(epsilon) / RZ(epsilon) for epsilon in {0.0, 0.04, 0.08, 0.14} rad
   and 1q/2q depolarizing noise (p1 = 0.001, p2 = 0.012), comparing equal total shot budgets
   (N_total = 36,000 shots per seed per method).
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Pauli
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, coherent_unitary_error, depolarizing_error
from qiskit.circuit.library import RZGate, RZZGate

from src.lombscargle_zne import (
    LombScargleZNE,
    fold_circuit_fractional,
    pauli_twirl_circuit,
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def build_tfim_trotter_circuit(n_qubits: int = 4, n_steps: int = 3, J: float = 0.35, h: float = 0.45) -> QuantumCircuit:
    """
    Builds a 1D Transverse-Field Ising Model (TFIM) Trotter evolution circuit using explicit
    CX - RZ - CX entangling blocks and RX single-qubit transverse field rotations.
    Observable: <Z_0>.
    """
    qc = QuantumCircuit(n_qubits)
    for _ in range(n_steps):
        # Even bond ZZ interactions
        for q in range(0, n_qubits - 1, 2):
            qc.cx(q, q + 1)
            qc.rz(2.0 * J, q + 1)
            qc.cx(q, q + 1)
        # Odd bond ZZ interactions
        for q in range(1, n_qubits - 1, 2):
            qc.cx(q, q + 1)
            qc.rz(2.0 * J, q + 1)
            qc.cx(q, q + 1)
        # Transverse field X rotations
        for q in range(n_qubits):
            qc.rx(2.0 * h, q)
    return qc


def ideal_expectation_z0(qc_no_meas: QuantumCircuit) -> float:
    sv = Statevector.from_instruction(qc_no_meas)
    op_str = "I" * (qc_no_meas.num_qubits - 1) + "Z"  # Qiskit little-endian: rightmost is qubit 0
    return float(np.real(sv.expectation_value(Pauli(op_str))))


def measure_z0_expectation(sim: AerSimulator, qc_no_meas: QuantumCircuit, shots: int, seed_sim: int) -> float:
    qc = qc_no_meas.copy()
    qc.measure_all()
    res = sim.run(qc, shots=shots, seed_simulator=seed_sim).result()
    counts = res.get_counts()
    exp_val = 0.0
    for bitstr, cnt in counts.items():
        # Rightmost bit of bitstr corresponds to qubit 0
        b0 = bitstr.replace(" ", "")[-1]
        exp_val += (1.0 if b0 == '0' else -1.0) * cnt
    return exp_val / shots


def make_transmon_noise_model(epsilon_coh: float, p1_dep: float = 0.001, p2_dep: float = 0.012) -> NoiseModel:
    noise = NoiseModel()
    # 1-qubit error: small coherent RZ(0.5*epsilon_coh) + depolarizing p1_dep
    err_1q_coh = coherent_unitary_error(RZGate(0.5 * epsilon_coh).to_matrix())
    err_1q_dep = depolarizing_error(p1_dep, 1)
    noise.add_all_qubit_quantum_error(err_1q_coh.compose(err_1q_dep), ['rx', 'rz', 'h'])

    # 2-qubit CX error: coherent RZZ(epsilon_coh) + local RZ(epsilon_coh) on control + depolarizing p2_dep
    # Note: a systematic control-qubit phase/Stark error + ZZ crosstalk around CX is twirled by Pauli Twirling!
    rzz_mat = RZZGate(epsilon_coh).to_matrix()
    rz_c_mat = np.kron(np.eye(2), RZGate(epsilon_coh).to_matrix())
    err_2q_coh = coherent_unitary_error(rzz_mat @ rz_c_mat)
    err_2q_dep = depolarizing_error(p2_dep, 2)
    noise.add_all_qubit_quantum_error(err_2q_coh.compose(err_2q_dep), ['cx'])
    return noise


def run_1q_echo_re_audit(n_seeds: int = 25) -> dict:
    """
    Re-audits the v1 1-qubit coherent idle echo (`H - Id^s - H`, theta=0.22 rad, p_dep=0.025)
    across 8 non-uniform scale factors s in {1, 2, 3, 5, 7, 10, 13, 17} over 25 random seeds,
    comparing Global Quadratic ZNE (v1 baseline), Local Linear ZNE (first 4 scales),
    Exponential ZNE, and Damped Harmonic / Lomb-Scargle ZNE.
    """
    print("\n--- Re-Auditing 1-Qubit Coherent Idle Echo (theta=0.22 rad, p=0.025, 8 scales, 25 seeds) ---")
    scales = np.array([1, 2, 3, 5, 7, 10, 13, 17], dtype=float)
    theta_coh = 0.22
    p_dep = 0.025
    shots = 4096
    E_ideal = 1.0

    noise = NoiseModel()
    err_u = coherent_unitary_error(RZGate(theta_coh).to_matrix())
    err_d = depolarizing_error(p_dep, 1)
    noise.add_all_qubit_quantum_error(err_u.compose(err_d), ['id'])
    sim = AerSimulator(noise_model=noise)
    zne_solver = LombScargleZNE()

    errs = {
        'Unmitigated': [],
        'Global_Quadratic_ZNE': [],
        'Local_Linear_ZNE': [],
        'Exponential_ZNE': [],
        'Damped_Harmonic_ZNE': [],
    }
    for seed in range(n_seeds):
        rng = np.random.default_rng(5000 + seed)
        y_vals = []
        for s in scales:
            qc = QuantumCircuit(1, 1)
            qc.h(0)
            for _ in range(int(s)):
                qc.id(0)
                qc.barrier(0)
            qc.h(0)
            qc.measure(0, 0)
            counts = sim.run(qc, shots=shots, seed_simulator=int(rng.integers(1, 10**8))).result().get_counts()
            n0 = counts.get('0', 0)
            n1 = counts.get('1', 0)
            # In H - RZ(s*theta) - H, <Z> = cos(s*theta)*exp(-gamma*s) -> ideal at s=0 is +1.0
            y_vals.append((n0 - n1) / shots)
        y_vals = np.array(y_vals)

        p_quad = np.polyfit(scales, y_vals, 2)
        p_lin = np.polyfit(scales[:4], y_vals[:4], 1)
        e_exp = float(zne_solver.extrapolate(scales, y_vals, method='exponential')['zero_noise_value'])
        e_harm = float(zne_solver.extrapolate(scales, y_vals, method='damped_oscillator')['zero_noise_value'])

        errs['Unmitigated'].append(abs(y_vals[0] - E_ideal))
        errs['Global_Quadratic_ZNE'].append(abs(float(p_quad[2]) - E_ideal))
        errs['Local_Linear_ZNE'].append(abs(float(p_lin[1]) - E_ideal))
        errs['Exponential_ZNE'].append(abs(e_exp - E_ideal))
        errs['Damped_Harmonic_ZNE'].append(abs(e_harm - E_ideal))

    out = {}
    for k, arr in errs.items():
        a = np.array(arr)
        out[k] = {
            'mean_abs_error': float(np.mean(a)),
            'se_abs_error': float(np.std(a, ddof=1) / np.sqrt(len(a))),
            'median_abs_error': float(np.median(a))
        }
        print(f"  {k:<25s}: MAE = {out[k]['mean_abs_error']:.4f} +/- {out[k]['se_abs_error']:.4f}")
    return out


def run_fair_head_to_head_study():
    print("=" * 84)
    print("FAIR PRE-REGISTERED QUANTUM ZNE BENCHMARK: FRACTIONAL FOLDING & PAULI TWIRLING")
    print("=" * 84)

    echo_1q_summary = run_1q_echo_re_audit(n_seeds=25)

    base_qc = build_tfim_trotter_circuit(n_qubits=4, n_steps=3, J=0.35, h=0.45)
    E_ideal = ideal_expectation_z0(base_qc)
    print(f"\n4-Qubit TFIM (3 Trotter steps, {base_qc.count_ops().get('cx', 0)} CX gates) Ideal <Z_0> = {E_ideal:+.6f}")

    # Non-uniform fractional target scales (using 6 scale points for standard & spectral ZNE)
    target_scales = [1.0, 1.35, 1.75, 2.25, 2.85, 3.50]
    folded_circuits = []
    exact_scales = []
    for ts in target_scales:
        f_qc, lam = fold_circuit_fractional(base_qc, ts, two_qubit_weight=10.0, use_barriers=True)
        folded_circuits.append(f_qc)
        exact_scales.append(lam)
    exact_scales = np.array(exact_scales, dtype=float)
    print(f"Non-uniform fractional noise scale factors lambda: {np.round(exact_scales, 4).tolist()}")

    epsilons = [0.00, 0.04, 0.08, 0.14]
    n_seeds = 20
    total_shot_budget = 24000
    shots_per_scale = total_shot_budget // len(exact_scales)  # 4000 shots per scale point
    n_twirls_per_scale = 8
    shots_per_twirl = shots_per_scale // n_twirls_per_scale  # 500 shots * 8 twirls = 4000 shots per scale!

    zne_solver = LombScargleZNE()
    study_results = {
        'echo_1q_re_audit': echo_1q_summary,
        'ideal_expectation': E_ideal,
        'exact_fractional_scales': exact_scales.tolist(),
        'total_shot_budget': total_shot_budget,
        'n_seeds': n_seeds,
        'configs': {}
    }

    methods_list = [
        'Unmitigated',
        'Linear_ZNE',
        'Quadratic_ZNE',
        'Exponential_ZNE',
        'Damped_Harmonic_ZNE',
        'Pauli_Twirl_Linear_ZNE',
        'Pauli_Twirl_Exp_ZNE',
    ]

    for eps in epsilons:
        print(f"\n--- Running 4-Qubit TFIM Benchmark for Coherent Error epsilon = {eps:.2f} rad ({n_seeds} seeds) ---")
        sim = AerSimulator(noise_model=make_transmon_noise_model(eps))
        errs = {m: [] for m in methods_list}

        for seed_idx in range(n_seeds):
            rng = np.random.default_rng(10000 + int(eps * 1000) + seed_idx)

            # 1. Untwirled measurements across fractional scales (4000 shots per scale = 24,000 total)
            y_untwirled = np.array([
                measure_z0_expectation(sim, f_qc, shots=shots_per_scale, seed_sim=int(rng.integers(1, 10**8)))
                for f_qc in folded_circuits
            ])

            # Also measure unmitigated with full shot budget or base scale
            e_unmit = float(y_untwirled[0])

            # Local Linear ZNE (first 4 fractional scales: lambda in [1.0, 2.22])
            p_lin = np.polyfit(exact_scales[:4], y_untwirled[:4], 1)
            e_lin = float(p_lin[1])

            # Quadratic Richardson/Polynomial ZNE
            p_quad = np.polyfit(exact_scales, y_untwirled, 2)
            e_quad = float(p_quad[2])

            # Exponential ZNE
            e_exp = float(zne_solver.extrapolate(exact_scales, y_untwirled, method='exponential')['zero_noise_value'])

            # Damped Harmonic / Lomb-Scargle ZNE
            e_harm = float(zne_solver.extrapolate(exact_scales, y_untwirled, method='damped_oscillator')['zero_noise_value'])

            # 2. Pauli Twirled measurements across the first 4 fractional scales (equal total shots = 24,000:
            #    4 scales * 12 twirl instances * 500 shots = 24,000 shots!)
            n_scales_twirl = 4
            n_twirl_inst = 12
            shots_tw = total_shot_budget // (n_scales_twirl * n_twirl_inst)
            y_twirled = []
            for s_i in range(n_scales_twirl):
                vals_t = []
                for _ in range(n_twirl_inst):
                    tw_qc = pauli_twirl_circuit(folded_circuits[s_i], rng)
                    vals_t.append(measure_z0_expectation(sim, tw_qc, shots=shots_tw, seed_sim=int(rng.integers(1, 10**8))))
                y_twirled.append(np.mean(vals_t))
            y_twirled = np.array(y_twirled)
            scales_tw = exact_scales[:n_scales_twirl]

            p_tw_lin = np.polyfit(scales_tw, y_twirled, 1)
            e_tw_lin = float(p_tw_lin[1])
            e_tw_exp = float(zne_solver.extrapolate(scales_tw, y_twirled, method='exponential')['zero_noise_value'])

            estimates = {
                'Unmitigated': e_unmit,
                'Linear_ZNE': e_lin,
                'Quadratic_ZNE': e_quad,
                'Exponential_ZNE': e_exp,
                'Damped_Harmonic_ZNE': e_harm,
                'Pauli_Twirl_Linear_ZNE': e_tw_lin,
                'Pauli_Twirl_Exp_ZNE': e_tw_exp,
            }
            for m, val in estimates.items():
                errs[m].append(abs(val - E_ideal))

        cfg_summary = {}
        for m in methods_list:
            arr = np.array(errs[m])
            mean_e = float(np.mean(arr))
            se_e = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
            med_e = float(np.median(arr))
            cfg_summary[m] = {'mean_abs_error': mean_e, 'se_abs_error': se_e, 'median_abs_error': med_e}
            print(f"  {m:<25s}: MAE = {mean_e:.4f} +/- {se_e:.4f} (median = {med_e:.4f})")

        # Evaluate pre-registered decision rule: does Damped_Harmonic_ZNE beat the best standard baseline by > 2 SE?
        std_baselines = ['Linear_ZNE', 'Quadratic_ZNE', 'Exponential_ZNE', 'Pauli_Twirl_Linear_ZNE', 'Pauli_Twirl_Exp_ZNE']
        best_base = min(std_baselines, key=lambda k: cfg_summary[k]['mean_abs_error'])
        mae_harm = cfg_summary['Damped_Harmonic_ZNE']['mean_abs_error']
        se_harm = cfg_summary['Damped_Harmonic_ZNE']['se_abs_error']
        mae_best = cfg_summary[best_base]['mean_abs_error']
        se_best = cfg_summary[best_base]['se_abs_error']
        combined_se = float(np.sqrt(se_harm**2 + se_best**2))
        z_score_advantage = (mae_best - mae_harm) / combined_se if combined_se > 0 else 0.0
        niche_confirmed = bool(z_score_advantage > 2.0)

        cfg_summary['decision_rule'] = {
            'best_standard_method': best_base,
            'best_standard_mae': mae_best,
            'damped_harmonic_mae': mae_harm,
            'z_score_advantage_over_best_standard': float(z_score_advantage),
            'spectral_niche_confirmed_2SE': niche_confirmed,
        }
        print(f"  -> Best Standard Method: {best_base} (MAE={mae_best:.4f}) vs Damped Harmonic (MAE={mae_harm:.4f}) "
              f"| z = {z_score_advantage:+.2f} SE -> Niche Confirmed: {niche_confirmed}")

        study_results['configs'][f"eps_{eps:.2f}"] = cfg_summary

    out_json = os.path.join(RESULTS_DIR, "fair_zne_study.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(study_results, f, indent=2)

    # Generate Publication Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))
    eps_arr = np.array(epsilons)
    plot_styles = [
        ('Unmitigated',            'o--', 'gray',        'Unmitigated ($\\lambda=1$)'),
        ('Linear_ZNE',             's-',  'forestgreen', 'Local Linear ZNE (Fractional Folding)'),
        ('Quadratic_ZNE',          '^:',  'darkorange',  'Quadratic Richardson ZNE'),
        ('Exponential_ZNE',        'v-.', 'teal',        'Exponential ZNE'),
        ('Damped_Harmonic_ZNE',    'd-',  'royalblue',   'Damped Harmonic / Lomb-Scargle ZNE'),
        ('Pauli_Twirl_Linear_ZNE', 'P-',  'crimson',     'Pauli Twirling + Linear ZNE (Equal Shots)'),
        ('Pauli_Twirl_Exp_ZNE',    '*-',  'purple',      'Pauli Twirling + Exponential ZNE (Equal Shots)'),
    ]

    for m_key, marker, col, label in plot_styles:
        means = [study_results['configs'][f"eps_{e:.2f}"][m_key]['mean_abs_error'] for e in epsilons]
        ses = [study_results['configs'][f"eps_{e:.2f}"][m_key]['se_abs_error'] for e in epsilons]
        ax1.errorbar(eps_arr, means, yerr=ses, fmt=marker, color=col, lw=1.8, capsize=3, label=label)

    ax1.set_xlabel('Coherent Over-Rotation Angle $\\epsilon$ (rad)', fontsize=10.5)
    ax1.set_ylabel('Mean Absolute Error $|\\hat{E}(0) - E_{\\mathrm{ideal}}|$', fontsize=10.5)
    ax1.set_title('(a) 4-Qubit TFIM Trotter Error vs. Coherent Over-Rotation (20 Seeds)', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8.2, loc='upper left')
    ax1.grid(True, alpha=0.35)

    # Bar chart at high coherent error epsilon = 0.14 rad
    cfg_hi = study_results['configs']['eps_0.14']
    short_names = ['Unmit.', 'Linear\nZNE', 'Quad.\nZNE', 'Exp.\nZNE', 'Damped\nHarmonic', 'Twirl +\nLinear', 'Twirl +\nExp.']
    maes_hi = [cfg_hi[m]['mean_abs_error'] for m in methods_list]
    ses_hi = [cfg_hi[m]['se_abs_error'] for m in methods_list]
    cols_hi = ['gray', 'forestgreen', 'darkorange', 'teal', 'royalblue', 'crimson', 'purple']

    bars = ax2.bar(range(len(methods_list)), maes_hi, yerr=ses_hi, color=cols_hi, edgecolor='black', capsize=4, width=0.6)
    ax2.set_xticks(range(len(methods_list)))
    ax2.set_xticklabels(short_names, fontsize=8.8)
    ax2.set_ylabel('Mean Absolute Error ($\\epsilon = 0.14\\,$rad)', fontsize=10.5)
    ax2.set_title('(b) Head-to-Head Comparison Under Strong Coherent Drift ($\\epsilon=0.14\\,$rad)', fontsize=11, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.35)
    for b, v in zip(bars, maes_hi):
        ax2.text(b.get_x() + b.get_width() / 2.0, v + 0.008, f"{v:.3f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    for p_img in [
        os.path.join(RESULTS_DIR, "fair_zne_comparison.png"),
        os.path.join(BASE_DIR, "benchmarks", "benchmark_zne_comparison.png"),
    ]:
        plt.savefig(p_img, dpi=300)
    plt.close()
    print(f"\nSaved fair benchmark results to {out_json} and figures to results/fair_zne_comparison.png")
    return study_results


if __name__ == '__main__':
    run_fair_head_to_head_study()
