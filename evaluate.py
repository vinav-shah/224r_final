"""
Evaluation script: compare trained PPO agent against baselines.

Usage:
    python evaluate.py --model results/bell_n2_combined_med_d20/final_model.pt
    python evaluate.py --model results/.../final_model.pt --noise depolarizing_strong
    python evaluate.py --baselines_only --target bell --noise combined_med
"""

import argparse
import json
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from noise_models import NOISE_MODELS
from quantum_env  import QuantumCircuitEnv
from ppo_agent    import PPOAgent
from baselines    import run_random, run_qiskit_transpiler, run_greedy


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",           default=None,   help="Path to saved PPO model (.pt)")
    p.add_argument("--target",          default="bell", choices=["bell", "swap", "ghz", "qft"])
    p.add_argument("--n_qubits",        type=int, default=2)
    p.add_argument("--noise",           default="combined_med", choices=list(NOISE_MODELS.keys()))
    p.add_argument("--noiseless",       action="store_true")
    p.add_argument("--max_depth",       type=int, default=20)
    p.add_argument("--depth_penalty",   type=float, default=0.005)
    p.add_argument("--n_eval_episodes", type=int, default=50)
    p.add_argument("--baselines_only",  action="store_true")
    p.add_argument("--run_greedy",      action="store_true", help="Include slow greedy baseline")
    p.add_argument("--out_dir",         default="results")
    p.add_argument("--vary_noise",      action="store_true", help="Eval across multiple noise strengths")
    return p.parse_args()


def rollout_ppo(agent: PPOAgent, env: QuantumCircuitEnv, n_episodes: int) -> dict:
    """Run the trained PPO agent greedily (argmax policy)."""
    fidelities, depths, circuits_desc = [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_circuit = []
        for _ in range(env.max_depth + 1):
            with torch.no_grad():
                obs_t  = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                logits, _ = agent.net(obs_t)
                action = logits.argmax(dim=-1).item()

            ep_circuit.append(env.action_name(action))
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                fidelities.append(info.get("fidelity", env._compute_fidelity()))
                depths.append(env._depth)
                circuits_desc.append(ep_circuit)
                break
        else:
            fidelities.append(env._compute_fidelity())
            depths.append(env._depth)
            circuits_desc.append(ep_circuit)

    return {
        "mean_fidelity": float(np.mean(fidelities)),
        "std_fidelity":  float(np.std(fidelities)),
        "mean_depth":    float(np.mean(depths)),
        "fidelities":    fidelities,
        "circuits":      circuits_desc,
    }


def print_results(name: str, res: dict):
    print(f"  {name:<28} fidelity={res['mean_fidelity']:.4f} ± {res['std_fidelity']:.4f}  "
          f"depth={res['mean_depth']:.1f}")


def plot_fidelity_bars(results: dict, out_path: str, title: str):
    names  = list(results.keys())
    means  = [results[n]["mean_fidelity"] for n in names]
    stds   = [results[n]["std_fidelity"]  for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors[:len(names)], alpha=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average Fidelity")
    ax.set_title(title)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved bar chart: {out_path}")


def plot_learning_curve(log_path: str, out_path: str):
    if not os.path.exists(log_path):
        return
    with open(log_path) as f:
        log = json.load(f)

    steps = [e["step"] for e in log]
    fids  = [e["mean_fidelity"] for e in log]
    rets  = [e["mean_return"]   for e in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(steps, fids, color="#4C72B0")
    ax1.set_xlabel("Environment steps")
    ax1.set_ylabel("Mean fidelity (last 20 ep)")
    ax1.set_title("Fidelity during training")

    ax2.plot(steps, rets, color="#DD8452")
    ax2.set_xlabel("Environment steps")
    ax2.set_ylabel("Mean return (last 20 ep)")
    ax2.set_title("Return during training")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved learning curve: {out_path}")


def vary_noise_eval(agent, target, n_qubits, max_depth, depth_penalty, n_ep, out_dir):
    """Evaluate PPO agent and random baseline across multiple noise strengths."""
    noise_keys = [
        "depolarizing_weak", "depolarizing_med", "depolarizing_strong",
        "combined_weak", "combined_med", "combined_strong",
    ]
    ppo_fids, rand_fids = [], []

    for nk in noise_keys:
        nm  = NOISE_MODELS[nk]()
        env = QuantumCircuitEnv(n_qubits=n_qubits, target_name=target,
                                noise_model=nm, max_depth=max_depth,
                                depth_penalty=depth_penalty)
        if agent:
            pr = rollout_ppo(agent, env, n_ep)
            ppo_fids.append(pr["mean_fidelity"])
        rr = run_random(env, n_ep)
        rand_fids.append(rr["mean_fidelity"])

    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(noise_keys))
    if agent:
        ax.plot(x, ppo_fids,  "o-", label="PPO",   color="#4C72B0")
    ax.plot(x, rand_fids, "s--", label="Random", color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels(noise_keys, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean Fidelity")
    ax.set_title(f"Fidelity vs Noise Strength — {target} ({n_qubits}q)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(out_dir, "noise_robustness.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved noise robustness plot: {path}")


def main():
    args = parse_args()

    noise_model = None if args.noiseless else NOISE_MODELS[args.noise]()
    env = QuantumCircuitEnv(
        n_qubits=args.n_qubits,
        target_name=args.target,
        noise_model=noise_model,
        max_depth=args.max_depth,
        depth_penalty=args.depth_penalty,
    )

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Determine output dir
    if args.model:
        out_dir = os.path.dirname(args.model)
    else:
        out_dir = os.path.join(args.out_dir, f"eval_{args.target}_n{args.n_qubits}")
    os.makedirs(out_dir, exist_ok=True)

    # Load PPO agent if provided
    agent = None
    if args.model and not args.baselines_only:
        agent = PPOAgent(obs_dim=obs_dim, n_actions=n_actions)
        agent.load(args.model)
        print(f"Loaded model: {args.model}\n")

    print(f"=== Evaluation: {args.target} ({args.n_qubits}q) | noise={args.noise if not args.noiseless else 'none'} ===\n")

    results = {}

    # Run baselines
    print("Running baselines...")
    results["Random"]   = run_random(env, args.n_eval_episodes)
    results["Qiskit"]   = run_qiskit_transpiler(env)
    if args.run_greedy:
        print("  (greedy baseline — this is slow)")
        results["Greedy"] = run_greedy(env, min(10, args.n_eval_episodes))

    # Run PPO agent
    if agent:
        print("Running PPO agent...")
        results["PPO"] = rollout_ppo(agent, env, args.n_eval_episodes)
        # Print sample circuit
        sample = results["PPO"]["circuits"][0] if results["PPO"]["circuits"] else []
        print(f"\n  Sample PPO circuit: {' → '.join(sample)}\n")

    print("\nResults:")
    for name, res in results.items():
        print_results(name, res)

    # Save results
    save = {k: {kk: vv for kk, vv in v.items() if kk not in ("circuits", "circuit")} for k, v in results.items()}
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(save, f, indent=2)

    # Plots
    noise_tag  = "noiseless" if args.noiseless else args.noise
    title      = f"Fidelity Comparison — {args.target} ({args.n_qubits}q, {noise_tag})"
    plot_fidelity_bars(results, os.path.join(out_dir, "fidelity_comparison.png"), title)

    log_path = os.path.join(out_dir, "log.json")
    plot_learning_curve(log_path, os.path.join(out_dir, "learning_curve.png"))

    if args.vary_noise:
        vary_noise_eval(agent, args.target, args.n_qubits, args.max_depth,
                        args.depth_penalty, args.n_eval_episodes, out_dir)


if __name__ == "__main__":
    main()
