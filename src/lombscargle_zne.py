r"""
LombScargleZNE: Spectral & Harmonic Zero-Noise Extrapolation and Pauli Twirling Benchmarks.

Provides:
1. Non-uniform fractional unitary gate folding (`fold_circuit_fractional`) with barriers
   to prevent transpiler cancellation while preserving exact logical unitary equivalence.
2. Exact 2-qubit Clifford Pauli Twirling (`pauli_twirl_circuit`) over the 16 two-qubit
   Pauli group conjugations of CX gates (converting coherent unitary over-rotations
   into stochastic depolarizing channels).
3. Extrapolation estimators: Local Linear ZNE, Richardson/Quadratic ZNE, Exponential ZNE,
   and Damped Harmonic / Lomb-Scargle Spectral ZNE (`LombScargleZNE`).
"""

import numpy as np
from scipy.signal import lombscargle
from scipy.optimize import curve_fit
from typing import Tuple, List, Dict, Union, Optional
import qiskit
from qiskit import QuantumCircuit


# Exact Pauli propagation table across CX(control, target):
# Given pre-gate Paulis (P_c, P_t) in {'I','X','Y','Z'}^2, returns post-gate Paulis (P'_c, P'_t)
# such that (P'_c \otimes P'_t) CX (P_c \otimes P_t) = +/- CX (logical operation invariant up to global phase).
CX_PAULI_TWIRL_TABLE = {
    ('I', 'I'): ('I', 'I'),
    ('I', 'X'): ('I', 'X'),
    ('I', 'Y'): ('Z', 'Y'),
    ('I', 'Z'): ('Z', 'Z'),
    ('X', 'I'): ('X', 'X'),
    ('X', 'X'): ('X', 'I'),
    ('X', 'Y'): ('Y', 'Z'),
    ('X', 'Z'): ('Y', 'Y'),
    ('Y', 'I'): ('Y', 'X'),
    ('Y', 'X'): ('Y', 'I'),
    ('Y', 'Y'): ('X', 'Z'),
    ('Y', 'Z'): ('X', 'Y'),
    ('Z', 'I'): ('Z', 'I'),
    ('Z', 'X'): ('Z', 'X'),
    ('Z', 'Y'): ('I', 'Y'),
    ('Z', 'Z'): ('I', 'Z'),
}
PAULI_PAIRS = list(CX_PAULI_TWIRL_TABLE.keys())


def _apply_pauli_1q(qc: QuantumCircuit, qubit, label: str):
    if label == 'X':
        qc.x(qubit)
    elif label == 'Y':
        qc.y(qubit)
    elif label == 'Z':
        qc.z(qubit)


class LombScargleZNE:
    """
    Zero-Noise Extrapolator utilizing spectral frequency analysis
    and damped harmonic models, alongside standard polynomial/exponential ZNE baselines.
    """

    def __init__(self, fap_threshold: float = 0.05, min_freq: float = 0.05, max_freq: float = 2.0):
        self.fap_threshold = fap_threshold
        self.min_freq = min_freq
        self.max_freq = max_freq

    def compute_spectrum(self, scale_factors: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        x = np.asarray(scale_factors, dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)

        poly = np.polyfit(x, y, 1)
        detrended_y = y - np.polyval(poly, x)
        detrended_y = detrended_y - np.mean(detrended_y)
        var_y = np.var(detrended_y)
        if var_y < 1e-14:
            var_y = 1e-14

        angular_freqs = np.linspace(self.min_freq, self.max_freq, 600)
        # Note: scipy.signal.lombscargle without normalize=True returns unnormalized power;
        # dividing by var_y approximates Scargle variance-normalized power.
        pgram = lombscargle(x, detrended_y, angular_freqs)
        norm_pgram = pgram / var_y

        max_z = float(np.max(norm_pgram))
        n_eff = len(x)
        fap = 1.0 - (1.0 - np.exp(-np.clip(max_z, 0.0, 50.0))) ** n_eff
        fap = float(np.clip(fap, 0.0, 1.0))

        return angular_freqs, norm_pgram, max_z, fap

    def extrapolate(
        self,
        scale_factors: Union[List[float], np.ndarray],
        values: Union[List[float], np.ndarray],
        method: str = 'auto'
    ) -> Dict[str, Union[float, str, np.ndarray]]:
        x = np.asarray(scale_factors, dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)

        freqs, pgram, max_z, fap = self.compute_spectrum(x, y)
        dom_freq = float(freqs[np.argmax(pgram)])

        def damped_osc(s, e0, gamma, omega, phi):
            return e0 * np.exp(-gamma * s) * np.cos(omega * s + phi)

        if method == 'damped_oscillator' or (method == 'auto' and fap < self.fap_threshold):
            try:
                e0_guess = float(y[0]) if abs(y[0]) > 0.05 else 1.0
                # At s -> 0, coherent over-rotation vanishes, so phase offset |phi| at s=0 is small
                popt, _ = curve_fit(
                    damped_osc, x, y,
                    p0=[e0_guess, 0.08, dom_freq, 0.0],
                    bounds=([-2.0, 0.0, 0.01, -0.35], [2.0, 1.5, 3.0, 0.35]),
                    maxfev=5000
                )
                e0_fit, gamma_fit, omega_fit, phi_fit = popt
                zero_noise_val = float(e0_fit * np.cos(phi_fit))
                return {
                    'zero_noise_value': zero_noise_val,
                    'model_type': 'damped_oscillator',
                    'fap': float(fap),
                    'dominant_freq': float(omega_fit),
                    'gamma': float(gamma_fit),
                    'params': popt,
                    'scale_factors': x,
                    'values': y
                }
            except Exception:
                pass

        if method == 'exponential':
            try:
                def exp_model(s, a, b):
                    return a * np.exp(-b * s)
                a0 = float(y[0]) if abs(y[0]) > 0.05 else 1.0
                popt, _ = curve_fit(exp_model, x, y, p0=[a0, 0.05], bounds=([-2.0, 0.0], [2.0, 2.0]), maxfev=3000)
                return {
                    'zero_noise_value': float(popt[0]),
                    'model_type': 'exponential',
                    'fap': float(fap),
                    'dominant_freq': dom_freq,
                    'params': popt,
                    'scale_factors': x,
                    'values': y
                }
            except Exception:
                pass

        if method == 'polynomial':
            p2 = np.polyfit(x, y, min(2, len(x) - 1))
            return {
                'zero_noise_value': float(p2[-1]),
                'model_type': 'polynomial_quad',
                'fap': float(fap),
                'dominant_freq': dom_freq,
                'params': p2,
                'scale_factors': x,
                'values': y
            }

        sub_len = min(4, len(x))
        p1 = np.polyfit(x[:sub_len], y[:sub_len], 1)
        return {
            'zero_noise_value': float(p1[1]),
            'model_type': 'linear_local',
            'fap': float(fap),
            'dominant_freq': dom_freq,
            'params': p1,
            'scale_factors': x,
            'values': y
        }


def fold_quantum_circuit(circuit: QuantumCircuit, scale_factor: int) -> QuantumCircuit:
    r"""
    Unitary digital gate folding: G -> G (G^\dagger G)^k
    For odd integer scale_factor = 2k + 1 (1, 3, 5, 7, ...).
    """
    if scale_factor <= 1:
        return circuit.copy()

    k = (scale_factor - 1) // 2
    folded = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)

    for _ in range(k):
        folded.compose(circuit, inplace=True)
        folded.compose(circuit.inverse(), inplace=True)
    folded.compose(circuit, inplace=True)

    return folded


def fold_circuit_fractional(
    circuit: QuantumCircuit,
    target_scale: float,
    two_qubit_weight: float = 10.0,
    use_barriers: bool = True
) -> Tuple[QuantumCircuit, float]:
    r"""
    Gate-level fractional unitary folding:
    Replaces full circuit layers and/or a subset of individual unitary instructions
    G with G G^\dagger G (separated by optional barriers so transpilers do not cancel them),
    returning `(folded_circuit, exact_effective_scale_lambda)`.

    Effective noise scale factor `lambda` is computed by weighting 2-qubit gates by
    `two_qubit_weight` relative to 1-qubit gates (reflecting typical transmon hardware
    where 2-qubit CX error ~10x 1-qubit error).
    """
    if target_scale <= 1.0 + 1e-9:
        return circuit.copy(), 1.0

    # Separate unitary instructions from final measurements/barriers
    unitary_insts = []
    meas_insts = []
    for inst in circuit.data:
        name = inst.operation.name
        if name in ('measure', 'reset'):
            meas_insts.append(inst)
        elif name != 'barrier':
            unitary_insts.append(inst)

    weights = np.array([
        two_qubit_weight if len(inst.qubits) >= 2 else 1.0
        for inst in unitary_insts
    ], dtype=float)
    base_weight = float(np.sum(weights))
    if base_weight == 0.0:
        return circuit.copy(), 1.0

    # Determine integer global folds k_full (each adds +2 * base_weight) and partial gate folds
    k_full = int(max(0, np.floor((target_scale - 1.0) / 2.0)))
    rem_scale = target_scale - (1.0 + 2.0 * k_full)
    target_extra_weight = (rem_scale / 2.0) * base_weight

    # Prioritize folding 2-qubit gates first, then 1-qubit gates from the end of the circuit
    order = np.argsort(-weights)
    fold_counts = np.full(len(unitary_insts), k_full, dtype=int)
    acc_w = 0.0
    for idx in order:
        if acc_w + 0.5 * weights[idx] <= target_extra_weight:
            fold_counts[idx] += 1
            acc_w += weights[idx]

    qc_out = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    total_weight = 0.0
    for idx, inst in enumerate(unitary_insts):
        op = inst.operation
        qargs = inst.qubits
        cargs = inst.clbits
        k_i = int(fold_counts[idx])
        qc_out.append(op, qargs, cargs)
        total_weight += weights[idx]
        if k_i > 0:
            op_inv = op.inverse()
            for _ in range(k_i):
                if use_barriers:
                    qc_out.barrier(*qargs)
                qc_out.append(op_inv, qargs, cargs)
                if use_barriers:
                    qc_out.barrier(*qargs)
                qc_out.append(op, qargs, cargs)
                total_weight += 2.0 * weights[idx]

    for inst in meas_insts:
        qc_out.append(inst.operation, inst.qubits, inst.clbits)

    exact_lambda = total_weight / base_weight
    return qc_out, float(exact_lambda)


def pauli_twirl_circuit(circuit: QuantumCircuit, rng: np.random.Generator) -> QuantumCircuit:
    r"""
    Applies Randomized Compiling (Pauli Twirling) around every 2-qubit `cx` gate in `circuit`
    by inserting a uniformly sampled Pauli pair (P_c, P_t) before `cx` and its exact
    Clifford-propagated partner (P'_c, P'_t) after `cx`.
    """
    twirled = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    for inst in circuit.data:
        op = inst.operation
        qargs = inst.qubits
        cargs = inst.clbits
        if op.name == 'cx' and len(qargs) == 2:
            pc, pt = PAULI_PAIRS[int(rng.integers(0, len(PAULI_PAIRS)))]
            pc_post, pt_post = CX_PAULI_TWIRL_TABLE[(pc, pt)]
            _apply_pauli_1q(twirled, qargs[0], pc)
            _apply_pauli_1q(twirled, qargs[1], pt)
            twirled.barrier(qargs[0], qargs[1])
            twirled.append(op, qargs, cargs)
            twirled.barrier(qargs[0], qargs[1])
            _apply_pauli_1q(twirled, qargs[0], pc_post)
            _apply_pauli_1q(twirled, qargs[1], pt_post)
        else:
            twirled.append(op, qargs, cargs)
    return twirled
