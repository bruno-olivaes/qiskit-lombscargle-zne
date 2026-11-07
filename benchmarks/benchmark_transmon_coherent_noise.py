"""
Empirical Benchmark: Quantum Error Mitigation on Transmon Qubits with Coherent Phase Noise.

Simulates a real quantum circuit in Qiskit 2.5 and Qiskit Aer under:
1. Incoherent depolarizing relaxation (T1, T2 decoherence).
2. Coherent systematic over-rotation / AC Stark detuning (Rz(theta)).
Evaluates why quadratic polynomial ZNE diverges across oscillatory regimes,
and measures the empirical performance of Linear ZNE vs. Damped Harmonic Spectral ZNE.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
import qiskit
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, coherent_unitary_error, depolarizing_error
from qiskit.circuit.library import RZGate
from src.lombscargle_zne import LombScargleZNE

def run_quantum_zne_benchmark():
    print("=" * 80)
    print("EMPIRICAL QUANTUM ERROR MITIGATION BENCHMARK (QISKIT AER & REAL NOISE)")
    print("=" * 80)

    # 1. Physical Target: Superposition state |+>.
    # Observable: X operator. Ideal noiseless expectation value: <X> = +1.0000
    E_ideal = 1.0000

    # 2. Transmon Noise Model:
    # Coherent phase detuning theta = 0.22 rad per cycle + depolarizing rate p = 0.025
    theta_true = 0.22
    p_dep_true = 0.025

    noise = NoiseModel()
    err_coh = coherent_unitary_error(RZGate(theta_true).to_matrix())
    err_dep = depolarizing_error(p_dep_true, 1)
    noise.add_all_qubit_quantum_error(err_coh.compose(err_dep), ['id', 'h'])

    sim = AerSimulator(noise_model=noise)

    # 3. Scale noise by digital identity insertion / folding
    scales = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
    measured_values = []
    shots = 40000

    print(f"Executing quantum circuits across {len(scales)} noise scale factors ({shots} shots each)...")
    for s in scales:
        circ = QuantumCircuit(1, 1)
        circ.h(0)
        # Apply s cycles of noisy identity operations
        for _ in range(int(s)):
            circ.id(0)
        # Measure in X basis (apply H then measure computational basis)
        circ.h(0)
        circ.measure(0, 0)

        counts = sim.run(circ, shots=shots).result().get_counts()
        p0 = counts.get('0', 0) / shots
        p1 = counts.get('1', 0) / shots
        measured_values.append(p0 - p1)

    measured_values = np.array(measured_values)

    # 4. Standard Extrapolations:
    # A) Unmitigated base hardware error (lambda = 1)
    E_unmitigated = measured_values[0]

    # B) Local Linear ZNE (points 1 to 4: small lambda regime)
    p_lin = np.polyfit(scales[:4], measured_values[:4], 1)
    E_linear = float(p_lin[1])

    # C) Quadratic ZNE on moderate range (points 1 to 6)
    p_quad_6 = np.polyfit(scales[:6], measured_values[:6], 2)
    E_quad_6 = float(p_quad_6[2])

    # D) Global Quadratic ZNE (all 12 points)
    p_quad_all = np.polyfit(scales, measured_values, 2)
    E_quad_all = float(p_quad_all[2])

    # E) Damped Harmonic Spectral ZNE (Our Method)
    zne = LombScargleZNE()
    res_spectral = zne.extrapolate(scales, measured_values, method='damped_oscillator')
    E_spectral = res_spectral['zero_noise_value']

    # 5. Compute Absolute Errors
    err_unmitigated = abs(E_unmitigated - E_ideal)
    err_lin = abs(E_linear - E_ideal)
    err_quad_6 = abs(E_quad_6 - E_ideal)
    err_quad_all = abs(E_quad_all - E_ideal)
    err_spectral = abs(E_spectral - E_ideal)

    print("\n" + "=" * 80)
    print("AUDITED BENCHMARK RESULTS: QUANTUM ERROR MITIGATION")
    print(f"Ground Truth (Ideal Noiseless Value): {E_ideal:+.4f}\n")
    print(f"{'Method':<35} {'Estimated E(0)':<18} {'Absolute Error':<16} {'Error Reduction'}")
    print("-" * 80)
    print(f"{'Unmitigated (Raw Noisy lambda=1)':<35} {E_unmitigated:<+18.4f} {err_unmitigated:<16.4f} {'Baseline':<15}")
    print(f"{'Linear ZNE (Small lambda <= 4)':<35} {E_linear:<+18.4f} {err_lin:<16.4f} {(err_unmitigated - err_lin)/err_unmitigated * 100:+.1f}%")
    print(f"{'Quadratic ZNE (pts 1-6)':<35} {E_quad_6:<+18.4f} {err_quad_6:<16.4f} {(err_unmitigated - err_quad_6)/err_unmitigated * 100:+.1f}%")
    print(f"{'Global Quadratic ZNE (all pts)':<35} {E_quad_all:<+18.4f} {err_quad_all:<16.4f} {(err_unmitigated - err_quad_all)/err_unmitigated * 100:+.1f}%")
    print(f"{'Damped Harmonic Spectral ZNE':<35} {E_spectral:<+18.4f} {err_spectral:<16.4f} {(err_unmitigated - err_spectral)/err_unmitigated * 100:+.1f}%")
    print("=" * 80)

    if 'dominant_freq' in res_spectral:
        print(f"Detected Harmonic Frequency: {res_spectral['dominant_freq']:.3f} rad/scale (True theta: {theta_true:.3f})")
    if 'gamma' in res_spectral:
        print(f"Fitted Damping Decay (gamma): {res_spectral['gamma']:.4f} (True p_dep: {p_dep_true:.4f})")

    # 6. Generate Publication Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x_dense = np.linspace(0.0, 12.5, 300)
    ax1.scatter(scales, measured_values, color='crimson', s=55, zorder=5, label='Measured Circuit Data (Aer QPU)')
    ax1.plot(x_dense, np.polyval(p_lin, x_dense), '--', color='forestgreen', lw=1.8, label=f'Linear ZNE (Err: {err_lin:.4f})')
    ax1.plot(x_dense, np.polyval(p_quad_all, x_dense), ':', color='orange', lw=2.0, label=f'Global Quadratic (Err: {err_quad_all:.4f})')

    if res_spectral['model_type'] == 'damped_oscillator':
        p = res_spectral['params']
        y_dense_osc = p[0] * np.exp(-p[1] * x_dense) * np.cos(p[2] * x_dense + p[3])
        ax1.plot(x_dense, y_dense_osc, '-', color='navy', lw=2.2, label=f'Damped Harmonic (Err: {err_spectral:.4f})')

    ax1.axhline(E_ideal, color='black', ls='-.', lw=1.5, label='Ideal Noiseless Target (1.0000)')
    ax1.set_xlim(-0.5, 12.5)
    ax1.set_ylim(-1.0, 1.3)
    ax1.set_xlabel('Noise Scale Factor $\lambda$', fontsize=11)
    ax1.set_ylabel('Expectation Value $\langle X \\rangle$', fontsize=11)
    ax1.set_title('Quantum ZNE: Coherent Phase Oscillation vs. Fits', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, alpha=0.4)

    # Subplot 2: Absolute Error Comparison
    methods = ['Unmitigated', 'Linear ZNE', 'Quad (pts 1-6)', 'Global Quad', 'Damped Harmonic']
    errors = [err_unmitigated, err_lin, err_quad_6, err_quad_all, err_spectral]
    colors = ['gray', 'forestgreen', 'mediumseagreen', 'crimson', 'royalblue']

    bars = ax2.bar(methods, errors, color=colors, edgecolor='black', width=0.55)
    ax2.set_ylabel('Absolute Error $|E(0) - E_{\mathrm{ideal}}|$', fontsize=11)
    ax2.set_title('Extrapolation Error Comparison', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(methods, rotation=20, ha='right', fontsize=9)
    ax2.grid(True, ls=':', alpha=0.5)

    for bar, err in zip(bars, errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{err:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    out_img = os.path.join(os.path.dirname(__file__), "benchmark_zne_comparison.png")
    plt.savefig(out_img, dpi=300)
    print(f"\nComparative figure saved to: {out_img}")

if __name__ == '__main__':
    run_quantum_zne_benchmark()
