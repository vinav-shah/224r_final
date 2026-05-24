"""Noise model definitions for Qiskit Aer simulation."""
import numpy as np
from qiskit_aer.noise import (
    NoiseModel, depolarizing_error, amplitude_damping_error, thermal_relaxation_error
)


def make_depolarizing_model(p1: float = 0.01, p2: float = 0.05) -> NoiseModel:
    """Depolarizing noise: p1 per single-qubit gate, p2 per two-qubit gate."""
    model = NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"])
    model.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    return model


def make_amplitude_damping_model(gamma: float = 0.02) -> NoiseModel:
    """Amplitude damping (T1 decay) noise applied after every gate."""
    model = NoiseModel()
    error1 = amplitude_damping_error(gamma)
    error2 = error1.expand(error1)
    model.add_all_qubit_quantum_error(error1, ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"])
    model.add_all_qubit_quantum_error(error2, ["cx"])
    return model


def make_combined_model(p1: float = 0.005, p2: float = 0.02, gamma: float = 0.01) -> NoiseModel:
    """Combined depolarizing + amplitude damping noise."""
    from qiskit_aer.noise import kraus_error
    from qiskit.quantum_info import Kraus

    dep1 = depolarizing_error(p1, 1)
    dep2 = depolarizing_error(p2, 2)
    amp1 = amplitude_damping_error(gamma)

    # Compose errors for single-qubit gates
    composed1 = dep1.compose(amp1)
    composed2 = dep2.compose(amp1.expand(amp1))

    model = NoiseModel()
    model.add_all_qubit_quantum_error(composed1, ["h", "x", "y", "z", "s", "t", "rx", "ry", "rz"])
    model.add_all_qubit_quantum_error(composed2, ["cx"])
    return model


def make_strong_noise_model(p1: float = 0.03, p2: float = 0.1, gamma: float = 0.03) -> NoiseModel:
    """Stronger noise for testing robustness."""
    return make_combined_model(p1, p2, gamma)


NOISE_MODELS = {
    "depolarizing_weak":  lambda: make_depolarizing_model(0.005, 0.02),
    "depolarizing_med":   lambda: make_depolarizing_model(0.01,  0.05),
    "depolarizing_strong":lambda: make_depolarizing_model(0.03,  0.10),
    "amplitude_damping":  lambda: make_amplitude_damping_model(0.02),
    "combined_weak":      lambda: make_combined_model(0.003, 0.01, 0.005),
    "combined_med":       lambda: make_combined_model(0.005, 0.02, 0.01),
    "combined_strong":    lambda: make_strong_noise_model(),
}
