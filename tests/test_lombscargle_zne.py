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

    omega = 0.4
    gamma = 0.02
    y_osc = 1.0 * np.exp(-gamma * x) * np.cos(omega * x)
    res_osc = zne.extrapolate(x, y_osc, method='damped_oscillator')
    assert abs(res_osc['zero_noise_value'] - 1.0) < 0.05
    assert abs(res_osc['dominant_freq'] - omega) < 0.05
