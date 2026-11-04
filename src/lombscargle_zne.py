r"""
LombScargleZNE: Spectral Zero-Noise Extrapolation for Quantum Error Mitigation.

This module provides a noise mitigation framework that utilizes the generalized
Lomb-Scargle periodogram to decouple coherent oscillatory phase errors (e.g.,
from AC Stark shifts, microwave over-rotation, and TLS defects in superconducting transmons)
from secular decay on non-uniformly sampled noise scaling grids.
"""

import numpy as np
from scipy.signal import lombscargle
from scipy.optimize import curve_fit
from typing import Tuple, List, Dict, Optional, Union
import qiskit
from qiskit import QuantumCircuit


class LombScargleZNE:
    """
    Zero-Noise Extrapolator utilizing Lomb-Scargle spectral analysis
    for non-uniform noise grids and coherent error decoupling.
    """

    def __init__(self, fap_threshold: float = 0.05, min_freq: float = 0.5, max_freq: float = 15.0):
        """
        Args:
            fap_threshold: Significance threshold for False Alarm Probability to identify
                           coherent harmonic noise components.
            min_freq: Minimum angular frequency to probe in Lomb-Scargle spectrum.
            max_freq: Maximum angular frequency to probe in Lomb-Scargle spectrum.
        """
        self.fap_threshold = fap_threshold
        self.min_freq = min_freq
        self.max_freq = max_freq

    def compute_spectrum(self, scale_factors: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Compute normalized Lomb-Scargle periodogram on un-interpolated coordinates.

        Returns:
            angular_freqs, norm_pgram, max_z, fap
        """
        x = np.asarray(scale_factors, dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)

        # Remove secular linear trend for pure harmonic spectral audit
        poly = np.polyfit(x, y, 1)
        detrended_y = y - np.polyval(poly, x)
        var_y = np.var(detrended_y)
        if var_y < 1e-14:
            var_y = 1e-14

        angular_freqs = np.linspace(self.min_freq, self.max_freq, 600)
        pgram = lombscargle(x, detrended_y, angular_freqs, precenter=True)
        # Normalized power z(omega) = P(omega) / var(y)
        norm_pgram = pgram / var_y

        max_z = np.max(norm_pgram)
        # False Alarm Probability under standard Scargle distribution: FAP = 1 - (1 - exp(-z))^N
        n_eff = len(x)
        fap = 1.0 - (1.0 - np.exp(-np.clip(max_z, 0, 50))) ** n_eff
        fap = float(np.clip(fap, 0.0, 1.0))

        return angular_freqs, norm_pgram, max_z, fap

    def extrapolate(
        self,
        scale_factors: Union[List[float], np.ndarray],
        values: Union[List[float], np.ndarray],
        sigmas: Optional[Union[List[float], np.ndarray]] = None
    ) -> Dict[str, Union[float, str, np.ndarray]]:
        """
        Perform zero-noise extrapolation to lambda -> 0.

        Evaluates whether coherent harmonic modes exist. If detected (FAP < threshold),
        fits a decoupled spectral model: E(lambda) = E0 * exp(-gamma * lambda) + A * cos(omega * lambda + phi).
        Here E0 represents the true unperturbed zero-noise expectation value at lambda = 0.
        Otherwise, fits a robust exponential / polynomial model.

        Returns:
            Dictionary with extrapolated 'zero_noise_value', 'model_type', 'fap', and details.
        """
        x = np.asarray(scale_factors, dtype=np.float64)
        y = np.asarray(values, dtype=np.float64)

        if sigmas is None:
            w = np.ones_like(y)
        else:
            w = 1.0 / (np.asarray(sigmas, dtype=np.float64) + 1e-12)

        # 1. Spectral Analysis
        freqs, pgram, max_z, fap = self.compute_spectrum(x, y)
        dominant_freq = freqs[np.argmax(pgram)]

        has_coherent_mode = fap < self.fap_threshold

        if has_coherent_mode:
            # Model: E(lambda) = E0 * exp(-gamma * scale) + A * cos(omega * scale + phi)
            # Notice that at scale = 0, the secular value is exactly E0.
            def spectral_model(scale, e0, gamma, amp, phi):
                return e0 * np.exp(-gamma * scale) + amp * np.cos(dominant_freq * scale + phi)

            try:
                # Estimate initial parameters from data
                poly = np.polyfit(x, y, 1)
                e0_init = poly[1]
                gamma_init = 0.15
                amp_init = np.std(y) * 0.7
                phi_init = 0.0

                popt, _ = curve_fit(
                    spectral_model, x, y,
                    p0=[e0_init, gamma_init, amp_init, phi_init],
                    bounds=([-np.inf, 0.0, -np.inf, -np.pi], [np.inf, 5.0, np.inf, np.pi]),
                    maxfev=6000
                )
                e0_val = popt[0]

                return {
                    'zero_noise_value': float(e0_val),
                    'model_type': 'spectral_lomb_scargle',
                    'fap': float(fap),
                    'dominant_freq': float(dominant_freq),
                    'params': popt,
                    'scale_factors': x,
                    'values': y
                }
            except Exception:
                # Fallback if non-linear curve_fit encounters singularities
                pass

        # Robust Exponential Fallback: E(scale) = E0 * exp(-gamma * scale)
        def exp_model(scale, e0, gamma):
            return e0 * np.exp(-gamma * scale)

        try:
            poly = np.polyfit(x, y, 1)
            popt, _ = curve_fit(exp_model, x, y, p0=[poly[1], 0.15], maxfev=3000)
            return {
                'zero_noise_value': float(popt[0]),
                'model_type': 'exponential_fallback',
                'fap': float(fap),
                'dominant_freq': float(dominant_freq),
                'params': popt,
                'scale_factors': x,
                'values': y
            }
        except Exception:
            # Robust Linear Fallback
            poly = np.polyfit(x, y, 1, w=w)
            return {
                'zero_noise_value': float(poly[1]),
                'model_type': 'linear_fallback',
                'fap': float(fap),
                'dominant_freq': float(dominant_freq),
                'params': poly,
                'scale_factors': x,
                'values': y
            }


def fold_quantum_circuit(circuit: QuantumCircuit, scale_factor: float) -> QuantumCircuit:
    r"""
    Perform unitary digital gate folding (G -> G G^\dagger G) to scale noise by scale_factor >= 1.0.
    For non-integer scale factor lambda, folds a random fraction of two-qubit / entangling gates.
    """
    if scale_factor <= 1.0001:
        return circuit.copy()

    ops = [op for op in circuit.data]
    total_gates = len(ops)
    if total_gates == 0:
        return circuit.copy()

    target_extra = int(np.round((scale_factor - 1.0) * total_gates / 2.0))
    if target_extra <= 0:
        return circuit.copy()

    folded_circuit = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
    indices_to_fold = set(np.random.choice(total_gates, size=min(target_extra, total_gates), replace=False))

    for idx, instruction in enumerate(ops):
        folded_circuit.append(instruction)
        if idx in indices_to_fold:
            try:
                inv_op = instruction.operation.inverse()
                folded_circuit.append(inv_op, instruction.qubits, instruction.clbits)
                folded_circuit.append(instruction)
            except Exception:
                pass

    return folded_circuit
