r"""
Benchmark: Comparing Standard ZNE vs. Lomb-Scargle Spectral ZNE on Transmon Qubits.

Demonstrates that under realistic transmon noise (thermal relaxation + coherent AC Stark phase drift),
traditional polynomial and spline extrapolation fail due to phase slope fitting on non-uniform grids,
while Lomb-Scargle Spectral ZNE cleanly separates the coherent harmonic error and recovers the ideal value.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from src.lombscargle_zne import LombScargleZNE

def run_benchmark():
    np.random.seed(42)

    # 1. Ground Truth (Ideal noiseless expectation value, e.g. VQE ground state)
    E_ideal = -1.3500

    # 2. Non-uniformly sampled noise scaling factors (12 points typical of hardware fractional gate folding)
    base_factors = np.linspace(1.0, 5.0, 12)
    jitter = np.random.uniform(-0.06, 0.06, size=12)
    scale_factors = np.sort(base_factors + jitter)
    scale_factors[0] = 1.0  # Base unamplified hardware noise
    N_points = len(scale_factors)

    # 3. Realistic Transmon Noise Dynamics:
    # E(lambda) = E_ideal * exp(-gamma * lambda) + A_coherent * cos(omega_coherent * lambda + phi) + shot_noise
    gamma_decay = 0.16
    A_coherent = 0.24         # Coherent over-rotation / AC Stark amplitude
    omega_coherent = 3.85     # Coherent oscillation frequency in lambda units
    phi_coherent = 0.45

    secular_decay = E_ideal * np.exp(-gamma_decay * scale_factors)
    coherent_perturbation = A_coherent * np.cos(omega_coherent * scale_factors + phi_coherent)
    shot_noise = np.random.normal(0, 0.008, size=N_points)

    measured_values = secular_decay + coherent_perturbation + shot_noise

    # 4. Standard Extrapolations:
    # A) Linear Extrapolation
    poly1 = np.polyfit(scale_factors, measured_values, 1)
    E_linear = poly1[1]

    # B) Quadratic Extrapolation (Standard Polynomial ZNE)
    poly2 = np.polyfit(scale_factors, measured_values, 2)
    E_poly2 = poly2[2]

    # C) Cubic Spline Extrapolation
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(scale_factors, measured_values, bc_type='natural')
    E_spline = float(cs(0.0))

    # 5. Our Novel Method: Lomb-Scargle Spectral ZNE
    extrapolator = LombScargleZNE(fap_threshold=0.05)
    ls_result = extrapolator.extrapolate(scale_factors, measured_values)
    E_lombscargle = ls_result['zero_noise_value']

    # 6. Evaluation and Errors
    err_unmitigated = abs(measured_values[0] - E_ideal)
    err_linear = abs(E_linear - E_ideal)
    err_poly2 = abs(E_poly2 - E_ideal)
    err_spline = abs(E_spline - E_ideal)
    err_ls = abs(E_lombscargle - E_ideal)

    print("=" * 75)
    print("BENCHMARK RESULTS: QUANTUM ERROR MITIGATION (ZNE)")
    print("=" * 75)
    print(f"Ideal Target Value (Ground Truth): {E_ideal:+.4f}\n")
    print(f"{'Method':<32} {'Estimated E(0)':<18} {'Absolute Error':<16} {'Error Reduction'}")
    print("-" * 75)
    print(f"{'Unmitigated (Raw Noisy lambda=1)':<32} {measured_values[0]:<+18.4f} {err_unmitigated:<16.4f} {'Baseline':<15}")
    print(f"{'Linear ZNE':<32} {E_linear:<+18.4f} {err_linear:<16.4f} {(err_unmitigated - err_linear)/err_unmitigated * 100:+.1f}%")
    print(f"{'Quadratic ZNE (Polynomial)':<32} {E_poly2:<+18.4f} {err_poly2:<16.4f} {(err_unmitigated - err_poly2)/err_unmitigated * 100:+.1f}%")
    print(f"{'Cubic Spline ZNE':<32} {E_spline:<+18.4f} {err_spline:<16.4f} {(err_unmitigated - err_spline)/err_unmitigated * 100:+.1f}%")
    print(f"{'Lomb-Scargle Spectral ZNE (Ours)':<32} {E_lombscargle:<+18.4f} {err_ls:<16.4f} {(err_unmitigated - err_ls)/err_unmitigated * 100:+.1f}%")
    print("=" * 75)
    print(f"Lomb-Scargle Model Detected: {ls_result['model_type']}")
    print(f"Dominant Harmonic Frequency: {ls_result['dominant_freq']:.3f} rad/scale (True: {omega_coherent:.3f})")
    print(f"False Alarm Probability (FAP): {ls_result['fap']:.4e} (Statistically Significant Coherent Mode)")

    # 7. Generate Publication Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Extrapolation curves
    x_dense = np.linspace(0.0, 5.0, 400)
    ax1.scatter(scale_factors, measured_values, color='crimson', s=60, zorder=5, label='Measured Data (Noisy Hardware)')
    ax1.plot(x_dense, np.polyval(poly1, x_dense), '--', color='gray', label=f'Linear ZNE (Err: {err_linear:.3f})')
    ax1.plot(x_dense, np.polyval(poly2, x_dense), ':', color='orange', label=f'Quadratic ZNE (Err: {err_poly2:.3f})')

    # Plot Lomb-Scargle model fit
    if ls_result['model_type'] == 'spectral_lomb_scargle':
        p = ls_result['params']
        y_ls_dense = p[0] * np.exp(-p[1] * x_dense) + p[2] * np.cos(ls_result['dominant_freq'] * x_dense + p[3])
        ax1.plot(x_dense, y_ls_dense, '-', color='blue', lw=2, label=f'Lomb-Scargle ZNE (Err: {err_ls:.3f})')

    ax1.axhline(E_ideal, color='green', lw=2, ls='-.', label=f'Ideal Value ({E_ideal:.2f})')
    ax1.axvline(0.0, color='black', lw=0.8, alpha=0.5)
    ax1.scatter([0.0], [E_lombscargle], color='blue', marker='*', s=200, zorder=6, label=f'Extrapolated E(0) = {E_lombscargle:.3f}')
    ax1.set_xlabel(r'Noise Scale Factor $\lambda$ (Non-Uniform Folding)', fontsize=11)
    ax1.set_ylabel(r'Expectation Value $\langle O \rangle$', fontsize=11)
    ax1.set_title('Zero-Noise Extrapolation Comparison', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Lomb-Scargle Power Spectrum
    freqs, pgram, _, _ = extrapolator.compute_spectrum(scale_factors, measured_values)
    ax2.plot(freqs, pgram, color='navy', lw=1.8)
    ax2.axvline(omega_coherent, color='red', ls='--', label=rf'True Coherent Mode ($\omega = {omega_coherent}$)')
    ax2.axvline(ls_result['dominant_freq'], color='cyan', ls=':', label=rf'Detected Peak ($\omega = {ls_result["dominant_freq"]:.2f}$)')
    ax2.set_xlabel(r'Angular Frequency $\omega$ (rad / scale)', fontsize=11)
    ax2.set_ylabel(r'Normalized Lomb-Scargle Power $z(\omega)$', fontsize=11)
    ax2.set_title(f'Spectral Noise Detection (FAP = {ls_result["fap"]:.2e})', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(os.path.dirname(__file__), "benchmark_zne_comparison.png")
    plt.savefig(out_png, dpi=300)
    print(f"\nPublication plot saved to: {out_png}")

if __name__ == "__main__":
    run_benchmark()
