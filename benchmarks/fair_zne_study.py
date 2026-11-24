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
from typing import List
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, amplitude_damping_error, coherent_unitary_error, depolarizing_error
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


def make_transmon_noise_model(
    epsilon_coh: float,
    p1_dep: float = 0.001,
    p2_dep: float = 0.012,
    noisy_twirl_frames: bool = False,
) -> NoiseModel:
    noise = NoiseModel()
    # 1-qubit error: small coherent RZ(0.5*epsilon_coh) + depolarizing p1_dep
    err_1q_coh = coherent_unitary_error(RZGate(0.5 * epsilon_coh).to_matrix())
    err_1q_dep = depolarizing_error(p1_dep, 1)
    noise.add_all_qubit_quantum_error(err_1q_coh.compose(err_1q_dep), ['rx', 'rz', 'h'])

    if noisy_twirl_frames:
        # Realistic finite error on Pauli twirl frame gates (['x', 'y', 'z']) inserted around CX
        err_twirl = coherent_unitary_error(RZGate(0.25 * epsilon_coh).to_matrix()).compose(
            depolarizing_error(p1_dep, 1)
        )
        noise.add_all_qubit_quantum_error(err_twirl, ['x', 'y', 'z'])

    # 2-qubit CX error: coherent RZZ(epsilon_coh) + local RZ(epsilon_coh) on control + depolarizing p2_dep
    rzz_mat = RZZGate(epsilon_coh).to_matrix()
    rz_c_mat = np.kron(np.eye(2), RZGate(epsilon_coh).to_matrix())
    err_2q_coh = coherent_unitary_error(rzz_mat @ rz_c_mat)
    err_2q_dep = depolarizing_error(p2_dep, 2)
    noise.add_all_qubit_quantum_error(err_2q_coh.compose(err_2q_dep), ['cx'])
    return noise


def _build_1q_echo_circuit(n_id: int) -> QuantumCircuit:
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    for _ in range(int(n_id)):
        qc.id(0)
        qc.barrier(0)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def _build_1q_bichromatic_batch(n_id: int, rng: np.random.Generator, n_batches: int = 32) -> List[QuantumCircuit]:
    """
    Constructs an out-of-family bi-chromatic + quasi-static Gaussian dephasing ensemble:
    20/32 shots evolve at mode 1 (theta1 = 0.22 + delta) and 12/32 at mode 2 (theta2 = 0.53 + delta),
    with quasi-static Gaussian frequency drift delta ~ N(0, 0.07^2) across batches.
    """
    circuits = []
    for b in range(n_batches):
        base_theta = 0.22 if b < 20 else 0.53
        delta = float(rng.normal(0.0, 0.07))
        theta_eff = base_theta + delta
        qc = QuantumCircuit(1, 1)
        qc.h(0)
        for _ in range(int(n_id)):
            qc.id(0)
            qc.rz(theta_eff, 0)
            qc.barrier(0)
        qc.h(0)
        qc.measure(0, 0)
        circuits.append(qc)
    return circuits


def _measure_batched_1q_z(sim: AerSimulator, circuits: List[QuantumCircuit], total_shots: int, rng: np.random.Generator) -> float:
    shots_each = max(1, total_shots // len(circuits))
    res = sim.run(circuits, shots=shots_each, seed_simulator=int(rng.integers(1, 10**8))).result()
    vals = []
    for idx in range(len(circuits)):
        counts = res.get_counts(idx)
        n0 = counts.get('0', 0)
        n1 = counts.get('1', 0)
        vals.append((n0 - n1) / shots_each)
    return float(np.mean(vals))


def _run_1q_scenario(
    scenario_name: str,
    scales: np.ndarray,
    sim: AerSimulator,
    zne_solver: LombScargleZNE,
    n_seeds: int = 25,
    shots_per_scale: int = 4096,
    bichromatic_detuning: bool = False,
) -> dict:
    E_ideal = 1.0
    total_budget = len(scales) * shots_per_scale
    n_local = min(4, len(scales))
    methods = [
        'Unmitigated',
        'Unmitigated_Equal_Budget',
        'Global_Quadratic_ZNE',
        'Local_Linear_ZNE',
        'Exponential_ZNE',
        'Damped_Harmonic_ZNE',
        'Auto_LombScargle_ZNE',
    ]
    circuit_counts = {
        'Unmitigated': 1,
        'Unmitigated_Equal_Budget': 1,
        'Global_Quadratic_ZNE': len(scales),
        'Local_Linear_ZNE': n_local,
        'Exponential_ZNE': len(scales),
        'Damped_Harmonic_ZNE': len(scales),
        'Auto_LombScargle_ZNE': len(scales),
    }
    shot_counts = {
        'Unmitigated': shots_per_scale,
        'Unmitigated_Equal_Budget': total_budget,
        'Global_Quadratic_ZNE': total_budget,
        'Local_Linear_ZNE': n_local * shots_per_scale,
        'Exponential_ZNE': total_budget,
        'Damped_Harmonic_ZNE': total_budget,
        'Auto_LombScargle_ZNE': total_budget,
    }

    errs = {m: [] for m in methods}
    unstable_flags = {m: [] for m in ['Exponential_ZNE', 'Damped_Harmonic_ZNE', 'Auto_LombScargle_ZNE']}
    auto_osc_selected = []

    print(f"\n--- Regime A [{scenario_name}]: {len(scales)} scales ({total_budget} eq-budget shots, {n_seeds} seeds) ---")
    for seed in range(n_seeds):
        rng = np.random.default_rng(5000 + sum(ord(c) for c in scenario_name) + seed)
        y_vals = []
        for s in scales:
            if bichromatic_detuning:
                circs = _build_1q_bichromatic_batch(int(s), rng, n_batches=32)
                y_vals.append(_measure_batched_1q_z(sim, circs, shots_per_scale, rng))
            else:
                qc = _build_1q_echo_circuit(int(s))
                y_vals.append(_measure_batched_1q_z(sim, [qc], shots_per_scale, rng))
        y_vals = np.array(y_vals, dtype=float)

        # Equal-budget unmitigated baseline: full total_budget shots spent at base scale s[0] = 1
        if bichromatic_detuning:
            eq_circs = _build_1q_bichromatic_batch(int(scales[0]), rng, n_batches=32)
            e_unmit_eq = _measure_batched_1q_z(sim, eq_circs, total_budget, rng)
        else:
            e_unmit_eq = _measure_batched_1q_z(sim, [_build_1q_echo_circuit(int(scales[0]))], total_budget, rng)

        p_quad = np.polyfit(scales, y_vals, 2)
        p_lin = np.polyfit(scales[:n_local], y_vals[:n_local], 1)
        res_exp = zne_solver.extrapolate(scales, y_vals, method='exponential')
        res_harm = zne_solver.extrapolate(scales, y_vals, method='damped_oscillator')
        res_auto = zne_solver.extrapolate(scales, y_vals, method='auto')

        estimates = {
            'Unmitigated': float(y_vals[0]),
            'Unmitigated_Equal_Budget': float(e_unmit_eq),
            'Global_Quadratic_ZNE': float(p_quad[2]),
            'Local_Linear_ZNE': float(p_lin[1]),
            'Exponential_ZNE': float(res_exp['zero_noise_value']),
            'Damped_Harmonic_ZNE': float(res_harm['zero_noise_value']),
            'Auto_LombScargle_ZNE': float(res_auto['zero_noise_value']),
        }
        for m, val in estimates.items():
            errs[m].append(abs(val - E_ideal))

        unstable_flags['Exponential_ZNE'].append(float(bool(res_exp.get('unstable_fit', False))))
        unstable_flags['Damped_Harmonic_ZNE'].append(float(bool(res_harm.get('unstable_fit', False))))
        unstable_flags['Auto_LombScargle_ZNE'].append(float(bool(res_auto.get('unstable_fit', False))))
        auto_osc_selected.append(float(res_auto.get('model_type') == 'damped_oscillator'))

    summary = {
        'scales': scales.tolist(),
        'total_shot_budget': int(total_budget),
        'n_seeds': int(n_seeds),
    }
    for m in methods:
        a = np.array(errs[m], dtype=float)
        inst_rate = float(np.mean(unstable_flags[m])) if m in unstable_flags else 0.0
        entry = {
            'mean_abs_error': float(np.mean(a)),
            'se_abs_error': float(np.std(a, ddof=1) / np.sqrt(len(a))),
            'median_abs_error': float(np.median(a)),
            'instability_rate': inst_rate,
            'n_circuits_per_seed': int(circuit_counts[m]),
            'total_shots_per_seed': int(shot_counts[m]),
        }
        if m == 'Auto_LombScargle_ZNE':
            entry['oscillator_selected_rate'] = float(np.mean(auto_osc_selected))
        summary[m] = entry
        extra = f" | inst={inst_rate:.2f}" if m in unstable_flags else ""
        if m == 'Auto_LombScargle_ZNE':
            extra += f" | osc_sel={entry['oscillator_selected_rate']:.2f}"
        print(f"  {m:<28s}: MAE = {entry['mean_abs_error']:.4f} +/- {entry['se_abs_error']:.4f}{extra}")

    std_baselines = ['Unmitigated_Equal_Budget', 'Global_Quadratic_ZNE', 'Local_Linear_ZNE', 'Exponential_ZNE']
    best_std = min(std_baselines, key=lambda k: summary[k]['mean_abs_error'])
    mae_harm = summary['Damped_Harmonic_ZNE']['mean_abs_error']
    se_harm = summary['Damped_Harmonic_ZNE']['se_abs_error']
    mae_auto = summary['Auto_LombScargle_ZNE']['mean_abs_error']
    se_auto = summary['Auto_LombScargle_ZNE']['se_abs_error']
    mae_eq = summary['Unmitigated_Equal_Budget']['mean_abs_error']
    se_eq = summary['Unmitigated_Equal_Budget']['se_abs_error']
    mae_best = summary[best_std]['mean_abs_error']
    se_best = summary[best_std]['se_abs_error']

    comb_best_harm = float(np.sqrt(se_harm**2 + se_best**2))
    comb_best_auto = float(np.sqrt(se_auto**2 + se_best**2))
    comb_eq_auto = float(np.sqrt(se_auto**2 + se_eq**2))
    z_harm_best = (mae_best - mae_harm) / comb_best_harm if comb_best_harm > 0 else 0.0
    z_auto_best = (mae_best - mae_auto) / comb_best_auto if comb_best_auto > 0 else 0.0
    z_auto_eq = (mae_eq - mae_auto) / comb_eq_auto if comb_eq_auto > 0 else 0.0

    summary['decision_rule'] = {
        'best_standard_method': best_std,
        'best_standard_mae': mae_best,
        'unmitigated_equal_budget_mae': mae_eq,
        'damped_harmonic_mae': mae_harm,
        'auto_lombscargle_mae': mae_auto,
        'z_harm_over_best_standard': float(z_harm_best),
        'z_auto_over_best_standard': float(z_auto_best),
        'z_auto_over_equal_budget': float(z_auto_eq),
        'auto_beats_equal_budget_2SE': bool(z_auto_eq > 2.0),
        'spectral_niche_confirmed_2SE': bool(z_harm_best > 2.0),
    }
    return summary


def run_1q_echo_re_audit(n_seeds: int = 25) -> dict:
    """
    Audits 1-qubit echo ZNE across 4 noise/scale regimes with strictly matched equal-shot baselines:
      - A1_deep_monomode_in_family: theta=0.22 rad, p_dep=0.025, scales {1,2,3,5,7,10,13,17}
      - A2_short_scale_coherent: theta=0.22 rad, p_dep=0.025, short scales {1,2,3,4,5,6}
      - A3_monotonic_depol_amp_damping: theta=0.0, amplitude_damping(0.015)+depol(0.025), scales {1..17}
      - A4_bichromatic_gaussian_1f_drift: bi-chromatic theta in {0.22,0.53} + N(0,0.07) quasi-static drift + depol(0.022)
    """
    zne_solver = LombScargleZNE()
    deep_scales = np.array([1, 2, 3, 5, 7, 10, 13, 17], dtype=float)
    short_scales = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    # A1: Deep in-family mono-mode coherent + depolarizing
    noise_a1 = NoiseModel()
    noise_a1.add_all_qubit_quantum_error(
        coherent_unitary_error(RZGate(0.22).to_matrix()).compose(depolarizing_error(0.025, 1)), ['id']
    )
    a1 = _run_1q_scenario(
        'A1_deep_monomode_in_family', deep_scales, AerSimulator(noise_model=noise_a1), zne_solver, n_seeds=n_seeds
    )

    # A2: Short-scale coherent (restricted folding horizon)
    a2 = _run_1q_scenario(
        'A2_short_scale_coherent', short_scales, AerSimulator(noise_model=noise_a1), zne_solver, n_seeds=n_seeds
    )

    # A3: Out-of-family purely monotonic amplitude damping + depolarizing
    noise_a3 = NoiseModel()
    noise_a3.add_all_qubit_quantum_error(
        amplitude_damping_error(0.015).compose(depolarizing_error(0.025, 1)), ['id']
    )
    a3 = _run_1q_scenario(
        'A3_monotonic_depol_amp_damping', deep_scales, AerSimulator(noise_model=noise_a3), zne_solver, n_seeds=n_seeds
    )

    # A4: Out-of-family bi-chromatic + quasi-static Gaussian frequency drift + depolarizing
    noise_a4 = NoiseModel()
    noise_a4.add_all_qubit_quantum_error(depolarizing_error(0.022, 1), ['id'])
    a4 = _run_1q_scenario(
        'A4_bichromatic_gaussian_1f_drift',
        deep_scales,
        AerSimulator(noise_model=noise_a4),
        zne_solver,
        n_seeds=n_seeds,
        bichromatic_detuning=True,
    )

    # Preserve top-level A1 keys for backward schema compatibility alongside full scenarios dict
    out = dict(a1)
    out['scenarios'] = {
        'A1_deep_monomode_in_family': a1,
        'A2_short_scale_coherent': a2,
        'A3_monotonic_depol_amp_damping': a3,
        'A4_bichromatic_gaussian_1f_drift': a4,
    }
    return out


def _eval_tfim_noise_config(
    label: str,
    sim: AerSimulator,
    folded_circuits: List[QuantumCircuit],
    exact_scales: np.ndarray,
    E_ideal: float,
    eps: float,
    n_seeds: int,
    total_shot_budget: int,
    zne_solver: LombScargleZNE,
) -> dict:
    shots_per_scale = total_shot_budget // len(exact_scales)  # 4000 * 6 = 24000
    n_scales_twirl = 4
    n_twirl_inst = 12
    shots_tw = total_shot_budget // (n_scales_twirl * n_twirl_inst)  # 4 * 12 * 500 = 24000
    shots_tw_only = total_shot_budget // n_twirl_inst  # 12 * 2000 = 24000

    methods_list = [
        'Unmitigated',
        'Unmitigated_Equal_Budget',
        'Pauli_Twirl_Only_Equal_Budget',
        'Linear_ZNE',
        'Quadratic_ZNE',
        'Exponential_ZNE',
        'Damped_Harmonic_ZNE',
        'Auto_LombScargle_ZNE',
        'Pauli_Twirl_Linear_ZNE',
        'Pauli_Twirl_Exp_ZNE',
    ]
    circuit_counts = {
        'Unmitigated': 1,
        'Unmitigated_Equal_Budget': 1,
        'Pauli_Twirl_Only_Equal_Budget': n_twirl_inst,
        'Linear_ZNE': 4,
        'Quadratic_ZNE': len(exact_scales),
        'Exponential_ZNE': len(exact_scales),
        'Damped_Harmonic_ZNE': len(exact_scales),
        'Auto_LombScargle_ZNE': len(exact_scales),
        'Pauli_Twirl_Linear_ZNE': n_scales_twirl * n_twirl_inst,
        'Pauli_Twirl_Exp_ZNE': n_scales_twirl * n_twirl_inst,
    }
    shot_counts = {
        'Unmitigated': shots_per_scale,
        'Unmitigated_Equal_Budget': total_shot_budget,
        'Pauli_Twirl_Only_Equal_Budget': n_twirl_inst * shots_tw_only,
        'Linear_ZNE': 4 * shots_per_scale,
        'Quadratic_ZNE': len(exact_scales) * shots_per_scale,
        'Exponential_ZNE': len(exact_scales) * shots_per_scale,
        'Damped_Harmonic_ZNE': len(exact_scales) * shots_per_scale,
        'Auto_LombScargle_ZNE': len(exact_scales) * shots_per_scale,
        'Pauli_Twirl_Linear_ZNE': n_scales_twirl * n_twirl_inst * shots_tw,
        'Pauli_Twirl_Exp_ZNE': n_scales_twirl * n_twirl_inst * shots_tw,
    }

    errs = {m: [] for m in methods_list}
    unstable_flags = {m: [] for m in ['Exponential_ZNE', 'Damped_Harmonic_ZNE', 'Auto_LombScargle_ZNE', 'Pauli_Twirl_Exp_ZNE']}
    auto_osc_selected = []

    print(f"\n--- Running 4-Qubit TFIM [{label}] for eps = {eps:.2f} rad ({n_seeds} seeds) ---")
    for seed_idx in range(n_seeds):
        rng = np.random.default_rng(10000 + sum(ord(c) for c in label) + int(eps * 1000) + seed_idx)

        # 1. Untwirled measurements across fractional scales (4000 shots per scale = 24,000 total)
        y_untwirled = np.array([
            measure_z0_expectation(sim, f_qc, shots=shots_per_scale, seed_sim=int(rng.integers(1, 10**8)))
            for f_qc in folded_circuits
        ])

        e_unmit = float(y_untwirled[0])
        # Equal-budget unmitigated baseline at lambda=1 with full 24,000 shots
        e_unmit_eq = float(measure_z0_expectation(
            sim, folded_circuits[0], shots=total_shot_budget, seed_sim=int(rng.integers(1, 10**8))
        ))

        # Pauli Twirling only (no extrapolation, lambda=1, 12 twirls * 2000 shots = 24,000 shots)
        tw_only_vals = [
            measure_z0_expectation(
                sim, pauli_twirl_circuit(folded_circuits[0], rng), shots=shots_tw_only, seed_sim=int(rng.integers(1, 10**8))
            )
            for _ in range(n_twirl_inst)
        ]
        e_tw_only = float(np.mean(tw_only_vals))

        p_lin = np.polyfit(exact_scales[:4], y_untwirled[:4], 1)
        p_quad = np.polyfit(exact_scales, y_untwirled, 2)
        res_exp = zne_solver.extrapolate(exact_scales, y_untwirled, method='exponential')
        res_harm = zne_solver.extrapolate(exact_scales, y_untwirled, method='damped_oscillator')
        res_auto = zne_solver.extrapolate(exact_scales, y_untwirled, method='auto')

        # 2. Pauli Twirled + ZNE across first 4 fractional scales (4 * 12 * 500 = 24,000 shots)
        y_twirled = []
        for s_i in range(n_scales_twirl):
            vals_t = [
                measure_z0_expectation(
                    sim, pauli_twirl_circuit(folded_circuits[s_i], rng), shots=shots_tw, seed_sim=int(rng.integers(1, 10**8))
                )
                for _ in range(n_twirl_inst)
            ]
            y_twirled.append(np.mean(vals_t))
        y_twirled = np.array(y_twirled)
        scales_tw = exact_scales[:n_scales_twirl]

        p_tw_lin = np.polyfit(scales_tw, y_twirled, 1)
        res_tw_exp = zne_solver.extrapolate(scales_tw, y_twirled, method='exponential')

        estimates = {
            'Unmitigated': e_unmit,
            'Unmitigated_Equal_Budget': e_unmit_eq,
            'Pauli_Twirl_Only_Equal_Budget': e_tw_only,
            'Linear_ZNE': float(p_lin[1]),
            'Quadratic_ZNE': float(p_quad[2]),
            'Exponential_ZNE': float(res_exp['zero_noise_value']),
            'Damped_Harmonic_ZNE': float(res_harm['zero_noise_value']),
            'Auto_LombScargle_ZNE': float(res_auto['zero_noise_value']),
            'Pauli_Twirl_Linear_ZNE': float(p_tw_lin[1]),
            'Pauli_Twirl_Exp_ZNE': float(res_tw_exp['zero_noise_value']),
        }
        for m, val in estimates.items():
            errs[m].append(abs(val - E_ideal))

        unstable_flags['Exponential_ZNE'].append(float(bool(res_exp.get('unstable_fit', False))))
        unstable_flags['Damped_Harmonic_ZNE'].append(float(bool(res_harm.get('unstable_fit', False))))
        unstable_flags['Auto_LombScargle_ZNE'].append(float(bool(res_auto.get('unstable_fit', False))))
        unstable_flags['Pauli_Twirl_Exp_ZNE'].append(float(bool(res_tw_exp.get('unstable_fit', False))))
        auto_osc_selected.append(float(res_auto.get('model_type') == 'damped_oscillator'))

    cfg_summary = {}
    for m in methods_list:
        arr = np.array(errs[m], dtype=float)
        mean_e = float(np.mean(arr))
        se_e = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        med_e = float(np.median(arr))
        inst_rate = float(np.mean(unstable_flags[m])) if m in unstable_flags else 0.0
        entry = {
            'mean_abs_error': mean_e,
            'se_abs_error': se_e,
            'median_abs_error': med_e,
            'instability_rate': inst_rate,
            'n_circuits_per_seed': int(circuit_counts[m]),
            'total_shots_per_seed': int(shot_counts[m]),
        }
        if m == 'Auto_LombScargle_ZNE':
            entry['oscillator_selected_rate'] = float(np.mean(auto_osc_selected))
        cfg_summary[m] = entry
        print(f"  {m:<30s}: MAE = {mean_e:.4f} +/- {se_e:.4f} (med = {med_e:.4f}, inst={inst_rate:.2f})")

    std_baselines = [
        'Unmitigated_Equal_Budget',
        'Pauli_Twirl_Only_Equal_Budget',
        'Linear_ZNE',
        'Quadratic_ZNE',
        'Exponential_ZNE',
        'Pauli_Twirl_Linear_ZNE',
        'Pauli_Twirl_Exp_ZNE',
    ]
    best_base = min(std_baselines, key=lambda k: cfg_summary[k]['mean_abs_error'])
    mae_harm = cfg_summary['Damped_Harmonic_ZNE']['mean_abs_error']
    se_harm = cfg_summary['Damped_Harmonic_ZNE']['se_abs_error']
    mae_best = cfg_summary[best_base]['mean_abs_error']
    se_best = cfg_summary[best_base]['se_abs_error']
    mae_eq = cfg_summary['Unmitigated_Equal_Budget']['mean_abs_error']
    se_eq = cfg_summary['Unmitigated_Equal_Budget']['se_abs_error']
    combined_se = float(np.sqrt(se_harm**2 + se_best**2))
    z_score_advantage = (mae_best - mae_harm) / combined_se if combined_se > 0 else 0.0
    combined_eq_best = float(np.sqrt(se_eq**2 + se_best**2))
    z_best_over_eq = (mae_eq - mae_best) / combined_eq_best if combined_eq_best > 0 else 0.0
    niche_confirmed = bool(z_score_advantage > 2.0)

    cfg_summary['decision_rule'] = {
        'best_standard_method': best_base,
        'best_standard_mae': mae_best,
        'unmitigated_equal_budget_mae': mae_eq,
        'damped_harmonic_mae': mae_harm,
        'auto_lombscargle_mae': cfg_summary['Auto_LombScargle_ZNE']['mean_abs_error'],
        'z_score_advantage_over_best_standard': float(z_score_advantage),
        'z_best_over_equal_budget': float(z_best_over_eq),
        'spectral_niche_confirmed_2SE': niche_confirmed,
    }
    print(f"  -> Best Standard: {best_base} (MAE={mae_best:.4f}) vs Unmit Eq={mae_eq:.4f} (z={z_best_over_eq:+.2f}) "
          f"| Damped Harm={mae_harm:.4f} (z={z_score_advantage:+.2f}) -> Niche Confirmed: {niche_confirmed}")
    return cfg_summary


def run_fair_head_to_head_study():
    print("=" * 84)
    print("FAIR PRE-REGISTERED QUANTUM ZNE BENCHMARK: EQUAL-BUDGET & OUT-OF-FAMILY AUDIT")
    print("=" * 84)

    echo_1q_summary = run_1q_echo_re_audit(n_seeds=25)

    base_qc = build_tfim_trotter_circuit(n_qubits=4, n_steps=3, J=0.35, h=0.45)
    E_ideal = ideal_expectation_z0(base_qc)
    print(f"\n4-Qubit TFIM (3 Trotter steps, {base_qc.count_ops().get('cx', 0)} CX gates) Ideal <Z_0> = {E_ideal:+.6f}")

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
    zne_solver = LombScargleZNE()

    study_results = {
        'echo_1q_re_audit': echo_1q_summary,
        'ideal_expectation': E_ideal,
        'exact_fractional_scales': exact_scales.tolist(),
        'total_shot_budget': total_shot_budget,
        'n_seeds': n_seeds,
        'configs': {},
    }

    for eps in epsilons:
        sim = AerSimulator(noise_model=make_transmon_noise_model(eps, noisy_twirl_frames=False))
        study_results['configs'][f"eps_{eps:.2f}"] = _eval_tfim_noise_config(
            label='ideal_twirl_frame',
            sim=sim,
            folded_circuits=folded_circuits,
            exact_scales=exact_scales,
            E_ideal=E_ideal,
            eps=eps,
            n_seeds=n_seeds,
            total_shot_budget=total_shot_budget,
            zne_solver=zne_solver,
        )

    # Noisy Pauli-twirl frame audit at eps = 0.08 rad
    sim_noisy_frame = AerSimulator(noise_model=make_transmon_noise_model(0.08, noisy_twirl_frames=True))
    study_results['noisy_twirl_frame_eps_0.08'] = _eval_tfim_noise_config(
        label='noisy_twirl_frame_x_y_z',
        sim=sim_noisy_frame,
        folded_circuits=folded_circuits,
        exact_scales=exact_scales,
        E_ideal=E_ideal,
        eps=0.08,
        n_seeds=n_seeds,
        total_shot_budget=total_shot_budget,
        zne_solver=zne_solver,
    )

    out_json = os.path.join(RESULTS_DIR, "fair_zne_study.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(study_results, f, indent=2)

    # Generate 3-panel Publication Figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # Panel (a): Regime A1-A4 Out-of-Family Stress Test
    scen_keys = [
        'A1_deep_monomode_in_family',
        'A2_short_scale_coherent',
        'A3_monotonic_depol_amp_damping',
        'A4_bichromatic_gaussian_1f_drift',
    ]
    scen_labels = ['A1: Deep\nMono-Mode', 'A2: Short\nCoherent', 'A3: Monotonic\nAmp+Depol', 'A4: Bichrom.\n+Gauss Drift']
    a_methods = [
        ('Unmitigated_Equal_Budget', 'gray', 'Unmitigated (Eq. Shots)'),
        ('Local_Linear_ZNE', 'forestgreen', 'Local Linear ZNE'),
        ('Exponential_ZNE', 'teal', 'Exponential ZNE'),
        ('Damped_Harmonic_ZNE', 'royalblue', 'Forced Harmonic ZNE'),
        ('Auto_LombScargle_ZNE', 'goldenrod', 'Auto Lomb-Scargle ZNE'),
    ]
    x_idx = np.arange(len(scen_keys))
    bar_w = 0.16
    scenarios_dict = echo_1q_summary['scenarios']
    for offset_i, (m_key, col, label) in enumerate(a_methods):
        maes = [scenarios_dict[sk][m_key]['mean_abs_error'] for sk in scen_keys]
        ses = [scenarios_dict[sk][m_key]['se_abs_error'] for sk in scen_keys]
        ax1.bar(x_idx + (offset_i - 2) * bar_w, maes, yerr=ses, width=bar_w, color=col, edgecolor='black', capsize=2.5, label=label)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(scen_labels, fontsize=8.5)
    ax1.set_ylabel('Mean Absolute Error $|\\hat{E}(0) - E_{\\mathrm{ideal}}|$', fontsize=10)
    ax1.set_title('(a) 1Q Out-of-Family & Equal-Budget Audit', fontsize=10.5, fontweight='bold')
    ax1.legend(fontsize=7.6, loc='upper right')
    ax1.grid(True, axis='y', alpha=0.35)

    # Panel (b): 4Q TFIM vs epsilon (including Equal-Budget & Twirl-Only & Auto)
    eps_arr = np.array(epsilons)
    plot_styles = [
        ('Unmitigated_Equal_Budget', 'o--', 'gray', 'Unmitigated (Eq. 24k Shots)'),
        ('Pauli_Twirl_Only_Equal_Budget', 'X:', 'slategray', 'Pauli Twirl Only (24k Shots)'),
        ('Linear_ZNE', 's-', 'forestgreen', 'Local Linear ZNE'),
        ('Auto_LombScargle_ZNE', 'd-', 'goldenrod', 'Auto Lomb-Scargle ZNE'),
        ('Pauli_Twirl_Linear_ZNE', 'P-', 'crimson', 'Pauli Twirl + Linear ZNE'),
        ('Pauli_Twirl_Exp_ZNE', '*-', 'purple', 'Pauli Twirl + Exp ZNE'),
    ]
    for m_key, marker, col, label in plot_styles:
        means = [study_results['configs'][f"eps_{e:.2f}"][m_key]['mean_abs_error'] for e in epsilons]
        ses = [study_results['configs'][f"eps_{e:.2f}"][m_key]['se_abs_error'] for e in epsilons]
        ax2.errorbar(eps_arr, means, yerr=ses, fmt=marker, color=col, lw=1.8, capsize=3, label=label)
    ax2.set_xlabel('Coherent Over-Rotation Angle $\\epsilon$ (rad)', fontsize=10)
    ax2.set_ylabel('Mean Absolute Error $|\\hat{E}(0) - E_{\\mathrm{ideal}}|$', fontsize=10)
    ax2.set_title('(b) 4Q TFIM vs. Coherent Drift (20 Seeds)', fontsize=10.5, fontweight='bold')
    ax2.legend(fontsize=7.5, loc='upper left')
    ax2.grid(True, alpha=0.35)

    # Panel (c): Ideal vs Noisy Twirl Frame at eps = 0.08 rad
    comp_methods = [
        ('Unmitigated_Equal_Budget', 'Unmit.\n(Eq.)'),
        ('Pauli_Twirl_Only_Equal_Budget', 'Twirl\nOnly'),
        ('Linear_ZNE', 'Linear\nZNE'),
        ('Damped_Harmonic_ZNE', 'Forced\nHarm.'),
        ('Auto_LombScargle_ZNE', 'Auto\nL-S'),
        ('Pauli_Twirl_Linear_ZNE', 'Twirl+\nLinear'),
        ('Pauli_Twirl_Exp_ZNE', 'Twirl+\nExp'),
    ]
    xc = np.arange(len(comp_methods))
    w3 = 0.36
    ideal_08 = study_results['configs']['eps_0.08']
    noisy_08 = study_results['noisy_twirl_frame_eps_0.08']
    m_id = [min(ideal_08[mk]['mean_abs_error'], 0.85) for mk, _ in comp_methods]
    s_id = [ideal_08[mk]['se_abs_error'] for mk, _ in comp_methods]
    m_ny = [min(noisy_08[mk]['mean_abs_error'], 0.85) for mk, _ in comp_methods]
    s_ny = [noisy_08[mk]['se_abs_error'] for mk, _ in comp_methods]
    ax3.bar(xc - w3 / 2, m_id, yerr=s_id, width=w3, color='steelblue', edgecolor='black', capsize=2.5, label='Ideal Twirl Frames')
    ax3.bar(xc + w3 / 2, m_ny, yerr=s_ny, width=w3, color='coral', edgecolor='black', capsize=2.5, label='Noisy Twirl Frames (x,y,z)')
    for idx_b, (mk, _) in enumerate(comp_methods):
        raw_val = ideal_08[mk]['mean_abs_error']
        if raw_val > 0.85:
            ax3.text(idx_b, 0.82, f">{0.85:.2f}\n({raw_val:.1f})", ha='center', va='top', fontsize=7.2, fontweight='bold')
    ax3.set_xticks(xc)
    ax3.set_xticklabels([lbl for _, lbl in comp_methods], fontsize=8.0)
    ax3.set_ylabel('Mean Absolute Error ($\\epsilon = 0.08\\,$rad)', fontsize=10)
    ax3.set_title('(c) Ideal vs. Noisy Twirl Frames ($\\epsilon=0.08$)', fontsize=10.5, fontweight='bold')
    ax3.legend(fontsize=7.8, loc='upper right')
    ax3.grid(True, axis='y', alpha=0.35)

    plt.tight_layout()
    for p_img in [
        os.path.join(RESULTS_DIR, "fair_zne_comparison.png"),
        os.path.join(BASE_DIR, "benchmarks", "benchmark_zne_comparison.png"),
        os.path.join(BASE_DIR, "manuscript", "fig_fair_zne_comparison.png"),
    ]:
        plt.savefig(p_img, dpi=300)
    plt.close()
    print(f"\nSaved fair benchmark results to {out_json} and figures to results/, benchmarks/, and manuscript/")
    return study_results


if __name__ == '__main__':
    run_fair_head_to_head_study()
