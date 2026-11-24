"""
Unit tests for LombScargleZNE, fractional unitary gate folding, and Clifford Pauli Twirling.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from src.lombscargle_zne import (
    LombScargleZNE,
    fold_quantum_circuit,
    fold_circuit_fractional,
    pauli_twirl_circuit,
)


def test_circuit_folding_unitary():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    f1 = fold_quantum_circuit(qc, 1)
    assert len(f1.data) == len(qc.data)

    f3 = fold_quantum_circuit(qc, 3)
    assert len(f3.data) == 3 * len(qc.data)
    assert Operator(f3).equiv(Operator(qc))

    f5 = fold_quantum_circuit(qc, 5)
    assert len(f5.data) == 5 * len(qc.data)
    assert Operator(f5).equiv(Operator(qc))


def test_fractional_folding_unitary_equivalence_and_nonuniform_scales():
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.rx(0.4, 2)
    qc.cx(1, 2)
    op_orig = Operator(qc)

    scales = []
    for target in [1.0, 1.4, 1.9, 2.5, 3.0]:
        f_qc, lam = fold_circuit_fractional(qc, target, two_qubit_weight=10.0, use_barriers=True)
        assert Operator(f_qc).equiv(op_orig)
        scales.append(lam)

    # Verify monotonically increasing fractional scale factors
    assert scales[0] == 1.0
    assert all(scales[i + 1] >= scales[i] for i in range(len(scales) - 1))
    assert any(not float(s).is_integer() for s in scales)


def test_pauli_twirl_preserves_exact_logical_operator():
    qc = QuantumCircuit(3)
    qc.rx(0.7, 0)
    qc.cx(0, 1)
    qc.rz(-0.3, 1)
    qc.cx(1, 2)
    op_orig = Operator(qc)

    rng = np.random.default_rng(12345)
    for _ in range(16):
        tw = pauli_twirl_circuit(qc, rng)
        assert Operator(tw).equiv(op_orig)


def test_lomb_scargle_frequency_detection():
    np.random.seed(42)
    true_omega = 0.80
    period = 2 * np.pi / true_omega
    x = np.sort(np.random.uniform(0.0, 4.0 * period, 60))
    y = np.cos(true_omega * x) + 0.05 * np.random.normal(size=len(x))

    zne = LombScargleZNE(min_freq=0.1, max_freq=2.0)
    freqs, pgram, max_z, fap = zne.compute_spectrum(x, y)
    dom_freq = freqs[np.argmax(pgram)]

    assert abs(dom_freq - true_omega) < 0.05
    assert fap < 0.01


def test_linear_and_damped_harmonic_extrapolation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    y_lin = 1.0 - 0.08 * x[:4]
    zne = LombScargleZNE()
    res_lin = zne.extrapolate(x[:4], y_lin, method='linear')
    assert abs(res_lin['zero_noise_value'] - 1.0) < 1e-4
    assert res_lin['unstable_fit'] is False

    omega = 0.4
    gamma = 0.02
    y_osc = 1.0 * np.exp(-gamma * x) * np.cos(omega * x)
    res_osc = zne.extrapolate(x, y_osc, method='damped_oscillator')
    assert abs(res_osc['zero_noise_value'] - 1.0) < 0.05
    assert abs(res_osc['dominant_freq'] - omega) < 0.05
    assert res_osc['at_bounds'] is False
    assert res_osc['unstable_fit'] is False
    assert res_osc['cond_number'] < 1e4


def test_stability_diagnostics_and_auto_fallback():
    zne = LombScargleZNE()
    # 1. Clean in-family damped oscillator should pass auto stability gate
    x_deep = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 13.0, 17.0])
    y_deep = np.exp(-0.025 * x_deep) * np.cos(0.22 * x_deep)
    res_auto_ok = zne.extrapolate(x_deep, y_deep, method='auto')
    assert res_auto_ok['model_type'] == 'damped_oscillator'
    assert res_auto_ok['unstable_fit'] is False
    assert abs(res_auto_ok['zero_noise_value'] - 1.0) < 0.02

    # 2. Purely monotonic short-scale data triggers auto fallback to linear_local
    x_short = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y_mono = np.exp(-0.06 * x_short)
    res_mono_auto = zne.extrapolate(x_short, y_mono, method='auto')
    assert res_mono_auto['model_type'] == 'linear_local'
    assert res_mono_auto['unstable_fit'] is False

    # 3. Pathological upward/saturating short data triggers bound saturation & instability in forced oscillator
    y_bad = np.array([0.92, 0.90, 0.91, 0.95, 1.02, 1.15])
    res_bad_forced = zne.extrapolate(x_short, y_bad, method='damped_oscillator')
    assert 'cond_number' in res_bad_forced
    assert 'at_bounds' in res_bad_forced
    assert res_bad_forced['unstable_fit'] is True

    # Auto rejects an unstable harmonic fit and falls back to linear_local
    res_bad_auto = zne.extrapolate(x_short, y_bad, method='auto')
    assert res_bad_auto['model_type'] == 'linear_local'
    assert res_bad_auto['unstable_fit'] is False


def test_fair_zne_study_json_completeness():
    import json
    from pathlib import Path

    json_path = Path(__file__).resolve().parents[1] / "results" / "fair_zne_study.json"
    assert json_path.exists(), f"Expected benchmark output at {json_path}"
    data = json.loads(json_path.read_text(encoding="utf-8"))

    scenarios = data["echo_1q_re_audit"]["scenarios"]
    expected_scenarios = {
        "A1_deep_monomode_in_family",
        "A2_short_scale_coherent",
        "A3_monotonic_depol_amp_damping",
        "A4_bichromatic_gaussian_1f_drift",
    }
    assert expected_scenarios.issubset(set(scenarios.keys()))
    for sc in expected_scenarios:
        assert "Unmitigated_Equal_Budget" in scenarios[sc]
        assert "Auto_LombScargle_ZNE" in scenarios[sc]
        assert "instability_rate" in scenarios[sc]["Damped_Harmonic_ZNE"]
        assert "n_circuits_per_seed" in scenarios[sc]["Unmitigated_Equal_Budget"]

    assert "noisy_twirl_frame_eps_0.08" in data
    for cfg in list(data["configs"].values()) + [data["noisy_twirl_frame_eps_0.08"]]:
        assert "Unmitigated_Equal_Budget" in cfg
        assert "Pauli_Twirl_Only_Equal_Budget" in cfg
        assert "Auto_LombScargle_ZNE" in cfg
        assert "instability_rate" in cfg["Damped_Harmonic_ZNE"]
        assert "n_circuits_per_seed" in cfg["Pauli_Twirl_Linear_ZNE"]
