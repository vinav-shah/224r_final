"""
Gymnasium environment for noise-aware quantum circuit compilation.

MDP formulation:
  State  : noiseless unitary of partial circuit + target unitary + depth fraction
  Action : add one gate from a fixed gate set, or STOP
  Reward : terminal fidelity under noise model (0 on non-terminal steps)
  Done   : STOP action OR max_depth reached
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional

import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, DensityMatrix, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

# ----- Gate catalogue -------------------------------------------------------

def _build_gate_catalogue(n_qubits: int):
    """
    Returns a list of (name, kwargs, qubit_indices) tuples defining every
    discrete action the agent can take (excluding STOP).
    """
    catalogue = []
    angles = [np.pi / 4, np.pi / 2, np.pi]

    for q in range(n_qubits):
        for gate in ("h", "x", "y", "z", "s", "t"):
            catalogue.append((gate, {}, [q]))
        for theta in angles:
            catalogue.append(("rx", {"theta": theta}, [q]))
            catalogue.append(("ry", {"theta": theta}, [q]))
            catalogue.append(("rz", {"theta": theta}, [q]))

    # Two-qubit entangling gates
    for ctrl in range(n_qubits):
        for tgt in range(n_qubits):
            if ctrl != tgt:
                catalogue.append(("cx", {}, [ctrl, tgt]))

    return catalogue  # STOP is implicitly action index len(catalogue)


def _apply_gate(circuit: QuantumCircuit, name: str, kwargs: dict, qubits: list):
    gate_fn = getattr(circuit, name)
    gate_fn(*kwargs.values(), *qubits)


# ----- Target unitaries -----------------------------------------------------

def bell_unitary(n_qubits: int = 2) -> np.ndarray:
    """Unitary for Bell-state preparation: H on q0 then CNOT(0,1)."""
    assert n_qubits >= 2
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    qc.cx(0, 1)
    return Operator(qc).data


def swap_unitary(n_qubits: int = 2) -> np.ndarray:
    assert n_qubits >= 2
    qc = QuantumCircuit(n_qubits)
    qc.swap(0, 1)
    return Operator(qc).data


def ghz_unitary(n_qubits: int = 3) -> np.ndarray:
    assert n_qubits >= 2
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return Operator(qc).data


def qft_unitary(n_qubits: int = 2) -> np.ndarray:
    """Quantum Fourier Transform unitary."""
    from qiskit.circuit.library import QFT
    return Operator(QFT(n_qubits)).data


TARGET_UNITARIES = {
    "bell":  bell_unitary,
    "swap":  swap_unitary,
    "ghz":   ghz_unitary,
    "qft":   qft_unitary,
}


# ----- Environment ----------------------------------------------------------

class QuantumCircuitEnv(gym.Env):
    """
    Gymnasium environment for RL-based noise-aware quantum circuit compilation.

    Parameters
    ----------
    n_qubits      : number of qubits (2 or 3)
    target_name   : key from TARGET_UNITARIES dict
    noise_model   : Qiskit NoiseModel (None = noiseless, for debugging)
    max_depth     : maximum number of gates before forced termination
    depth_penalty : coefficient for depth penalty added to terminal reward
    reward_shaping: if True, add intermediate reward proportional to fidelity gain
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_qubits: int = 2,
        target_name: str = "bell",
        noise_model: Optional[NoiseModel] = None,
        max_depth: int = 20,
        depth_penalty: float = 0.01,
        reward_shaping: bool = False,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.target_name = target_name
        self.noise_model = noise_model
        self.max_depth = max_depth
        self.depth_penalty = depth_penalty
        self.reward_shaping = reward_shaping

        # Build target unitary
        self.target_unitary = TARGET_UNITARIES[target_name](n_qubits)
        # Pre-compute ideal output states for each basis input
        self._ideal_outputs = self._precompute_ideal_outputs()

        # Gate catalogue
        self.catalogue = _build_gate_catalogue(n_qubits)
        self.n_gate_actions = len(self.catalogue)
        self.STOP = self.n_gate_actions  # last action index

        self.action_space = spaces.Discrete(self.n_gate_actions + 1)

        # Observation: flatten [real(U_current), imag(U_current),
        #                        real(U_target),  imag(U_target), depth_frac]
        obs_dim = 4 * self.dim * self.dim + 1
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        # Qiskit Aer density-matrix simulator (noiseless unitary is computed
        # analytically; noisy eval uses Aer)
        self.simulator = AerSimulator(method="density_matrix")

        self._circuit: Optional[QuantumCircuit] = None
        self._depth: int = 0
        self._prev_fidelity: float = 0.0

    # ------------------------------------------------------------------
    def _precompute_ideal_outputs(self):
        """Compute U_target |i⟩ for each computational basis state |i⟩."""
        outputs = []
        for i in range(self.dim):
            sv = np.zeros(self.dim, dtype=complex)
            sv[i] = 1.0
            outputs.append(self.target_unitary @ sv)
        return outputs

    def _circuit_unitary(self) -> np.ndarray:
        """Return the ideal (noiseless) unitary of the current circuit."""
        if self._depth == 0:
            return np.eye(self.dim, dtype=complex)
        return Operator(self._circuit).data

    def _obs(self) -> np.ndarray:
        U_cur = self._circuit_unitary()
        U_tgt = self.target_unitary
        depth_frac = np.float32(self._depth / self.max_depth)
        obs = np.concatenate([
            U_cur.real.ravel(),
            U_cur.imag.ravel(),
            U_tgt.real.ravel(),
            U_tgt.imag.ravel(),
            [depth_frac],
        ]).astype(np.float32)
        return obs

    def _compute_fidelity(self) -> float:
        """
        Average state fidelity over all computational basis states.

        All basis-state circuits are submitted in a single batched Aer job.
        Aer dispatches them across its internal thread pool (controlled by
        OMP_NUM_THREADS / the container's CPU count), giving near-linear
        speedup with core count without any Python-level thread overhead.
        """
        circuits = []
        for i in range(self.dim):
            init_qc = QuantumCircuit(self.n_qubits)
            for q in range(self.n_qubits):
                if (i >> q) & 1:
                    init_qc.x(q)
            full_qc = init_qc.compose(self._circuit) if self._depth > 0 else init_qc
            full_qc.save_density_matrix()
            circuits.append(full_qc)

        job     = self.simulator.run(circuits, noise_model=self.noise_model, shots=1)
        result  = job.result()
        total   = 0.0
        for i, circ in enumerate(circuits):
            dm    = DensityMatrix(result.data(i)["density_matrix"])
            ideal = Statevector(self._ideal_outputs[i])
            total += float(state_fidelity(ideal, dm))
        return total / self.dim

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._circuit = QuantumCircuit(self.n_qubits)
        self._depth = 0
        self._prev_fidelity = 0.0
        return self._obs(), {}

    def step(self, action: int):
        terminated = False
        truncated = False
        reward = 0.0

        if action == self.STOP or self._depth >= self.max_depth:
            # Terminal: compute noisy fidelity
            fidelity = self._compute_fidelity()
            penalty = self.depth_penalty * self._depth
            reward = float(fidelity) - penalty
            terminated = True
        else:
            name, kwargs, qubits = self.catalogue[action]
            _apply_gate(self._circuit, name, kwargs, qubits)
            self._depth += 1

            if self.reward_shaping:
                fidelity = self._compute_fidelity()
                reward = fidelity - self._prev_fidelity
                self._prev_fidelity = fidelity

            if self._depth >= self.max_depth:
                fidelity = self._compute_fidelity()
                penalty = self.depth_penalty * self._depth
                reward = float(fidelity) - penalty
                truncated = True

        obs = self._obs()
        info = {"depth": self._depth}
        if terminated or truncated:
            info["fidelity"] = self._compute_fidelity() if not (terminated or truncated) else (
                fidelity if "fidelity" in dir() else self._compute_fidelity()
            )

        return obs, reward, terminated, truncated, info

    def action_name(self, action: int) -> str:
        if action == self.STOP:
            return "STOP"
        name, kwargs, qubits = self.catalogue[action]
        kv = ",".join(f"{v:.3f}" for v in kwargs.values())
        qs = ",".join(str(q) for q in qubits)
        return f"{name}({kv})@[{qs}]" if kv else f"{name}@[{qs}]"
