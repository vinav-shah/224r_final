"""
Baseline methods for quantum circuit compilation comparison.

  1. Random agent       — uniformly samples gates until STOP or max_depth
  2. Qiskit transpiler  — uses Qiskit's standard transpiler on the ideal circuit
  3. Greedy fidelity    — myopic greedy: each step picks the gate with highest
                          immediate fidelity improvement (expensive, upper-ish bound)
"""

import numpy as np
from typing import Optional
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from quantum_env import QuantumCircuitEnv, TARGET_UNITARIES, _apply_gate


# ----- helpers --------------------------------------------------------------

def _eval_fidelity(env: QuantumCircuitEnv) -> float:
    """Public wrapper around env's private fidelity computation."""
    return env._compute_fidelity()


# ----- Random agent ---------------------------------------------------------

def run_random(
    env: QuantumCircuitEnv,
    n_episodes: int = 100,
    stop_prob: float = 0.1,
    seed: int = 0,
) -> dict:
    """
    At each step sample uniformly over gate actions.  With probability
    `stop_prob` issue a STOP instead.  Returns aggregated metrics.
    """
    rng = np.random.default_rng(seed)
    fidelities, depths = [], []

    for _ in range(n_episodes):
        env.reset()
        for step in range(env.max_depth):
            if rng.random() < stop_prob:
                action = env.STOP
            else:
                action = int(rng.integers(env.n_gate_actions))
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                fidelities.append(info.get("fidelity", env._compute_fidelity()))
                depths.append(env._depth)
                break
        else:
            fidelities.append(env._compute_fidelity())
            depths.append(env._depth)

    return {
        "mean_fidelity": float(np.mean(fidelities)),
        "std_fidelity":  float(np.std(fidelities)),
        "mean_depth":    float(np.mean(depths)),
        "fidelities":    fidelities,
    }


# ----- Qiskit transpiler baseline -------------------------------------------

def run_qiskit_transpiler(
    env: QuantumCircuitEnv,
    optimization_level: int = 3,
    seed_transpiler: int = 42,
) -> dict:
    """
    Build the ideal target circuit, transpile it to the native gate set,
    then evaluate fidelity under the noise model.
    """
    n_qubits  = env.n_qubits
    simulator = AerSimulator(method="density_matrix")

    # Reconstruct ideal target circuit from unitary
    ideal_qc = QuantumCircuit(n_qubits)
    # Build the circuit we know corresponds to the target
    from quantum_env import TARGET_UNITARIES
    target_fn = TARGET_UNITARIES[env.target_name]

    # We build the circuit by name since we know the target
    if env.target_name == "bell":
        ideal_qc.h(0); ideal_qc.cx(0, 1)
    elif env.target_name == "swap":
        ideal_qc.swap(0, 1)
    elif env.target_name == "ghz":
        ideal_qc.h(0)
        for i in range(n_qubits - 1):
            ideal_qc.cx(i, i + 1)
    elif env.target_name == "qft":
        from qiskit.circuit.library import QFT
        ideal_qc = QFT(n_qubits).decompose()

    # Transpile
    basis_gates = ["cx", "h", "x", "y", "z", "s", "t", "rx", "ry", "rz"]
    transpiled = transpile(
        ideal_qc,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        seed_transpiler=seed_transpiler,
    )
    depth = transpiled.depth()

    # Evaluate fidelity under noise by injecting into env
    # We can't directly use env._compute_fidelity because the circuit isn't in env._circuit.
    # Replicate the calculation here.
    dim = 2 ** n_qubits
    ideal_outputs = env._ideal_outputs
    total = 0.0

    for i in range(dim):
        init_qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            if (i >> q) & 1:
                init_qc.x(q)
        full_qc = init_qc.compose(transpiled)
        full_qc.save_density_matrix()

        job = simulator.run(full_qc, noise_model=env.noise_model, shots=1)
        dm  = DensityMatrix(job.result().data()["density_matrix"])
        total += float(state_fidelity(Statevector(ideal_outputs[i]), dm))

    fidelity = total / dim

    return {
        "mean_fidelity": fidelity,
        "std_fidelity":  0.0,
        "mean_depth":    float(depth),
        "fidelities":    [fidelity],
        "circuit":       transpiled,
    }


# ----- Greedy myopic baseline -----------------------------------------------

def run_greedy(
    env: QuantumCircuitEnv,
    n_episodes: int = 10,
) -> dict:
    """
    At each step, try every possible gate action, keep the one that maximises
    fidelity, then commit it.  Expensive but a strong sequential greedy upper bound.
    """
    import copy

    fidelities, depths = [], []

    for _ in range(n_episodes):
        env.reset()
        for step in range(env.max_depth):
            best_fid   = -1.0
            best_action = env.STOP

            for a in range(env.n_gate_actions):
                # Speculatively apply gate
                name, kwargs, qubits = env.catalogue[a]
                _apply_gate(env._circuit, name, kwargs, qubits)
                env._depth += 1
                fid = env._compute_fidelity()
                # Undo
                env._circuit.data.pop()
                env._depth -= 1

                if fid > best_fid:
                    best_fid    = fid
                    best_action = a

            _, _, terminated, truncated, info = env.step(best_action)
            if terminated or truncated:
                break

        fidelity = env._compute_fidelity()
        fidelities.append(fidelity)
        depths.append(env._depth)

    return {
        "mean_fidelity": float(np.mean(fidelities)),
        "std_fidelity":  float(np.std(fidelities)),
        "mean_depth":    float(np.mean(depths)),
        "fidelities":    fidelities,
    }
