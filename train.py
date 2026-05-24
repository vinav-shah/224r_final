"""
Training script for RL-based noise-aware quantum circuit compilation.

Usage:
    python train.py                         # default: bell target, combined_med noise
    python train.py --target ghz --n_qubits 3
    python train.py --noise depolarizing_strong --depth_penalty 0.005
    python train.py --noiseless              # debug: no noise
"""

import argparse
import json
import os
import time
import numpy as np

from noise_models import NOISE_MODELS
from quantum_env import QuantumCircuitEnv
from ppo_agent import PPOAgent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target",         default="bell",         choices=["bell", "swap", "ghz", "qft"])
    p.add_argument("--n_qubits",       type=int, default=2)
    p.add_argument("--noise",          default="combined_med", choices=list(NOISE_MODELS.keys()))
    p.add_argument("--noiseless",      action="store_true")
    p.add_argument("--max_depth",      type=int,   default=20)
    p.add_argument("--depth_penalty",  type=float, default=0.005)
    p.add_argument("--reward_shaping", action="store_true")
    p.add_argument("--total_steps",    type=int,   default=200_000)
    p.add_argument("--rollout_steps",  type=int,   default=256)
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--hidden",         type=int,   default=256)
    p.add_argument("--n_epochs",       type=int,   default=10)
    p.add_argument("--batch_size",     type=int,   default=64)
    p.add_argument("--clip_eps",       type=float, default=0.2)
    p.add_argument("--ent_coef",       type=float, default=0.01)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--out_dir",        default="results")
    p.add_argument("--run_name",       default=None)
    return p.parse_args()


def make_run_name(args) -> str:
    noise_tag = "noiseless" if args.noiseless else args.noise
    return f"{args.target}_n{args.n_qubits}_{noise_tag}_d{args.max_depth}"


def main():
    args = parse_args()
    np.random.seed(args.seed)

    run_name = args.run_name or make_run_name(args)
    out_dir  = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    # Save config
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Build environment
    noise_model = None if args.noiseless else NOISE_MODELS[args.noise]()
    env = QuantumCircuitEnv(
        n_qubits=args.n_qubits,
        target_name=args.target,
        noise_model=noise_model,
        max_depth=args.max_depth,
        depth_penalty=args.depth_penalty,
        reward_shaping=args.reward_shaping,
    )

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPOAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden=args.hidden,
        lr=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=args.clip_eps,
        vf_coef=0.5,
        ent_coef=args.ent_coef,
        max_grad_norm=0.5,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        rollout_steps=args.rollout_steps,
        device="cpu",
    )

    print(f"\n{'='*60}")
    print(f"Run : {run_name}")
    print(f"Obs : {obs_dim}  |  Actions: {n_actions}")
    print(f"Target: {args.target} ({args.n_qubits} qubits)")
    print(f"Noise : {'none' if args.noiseless else args.noise}")
    print(f"Total steps: {args.total_steps:,}")
    print(f"{'='*60}\n")

    # -------- Training loop -------------------------------------------------
    log = []          # list of dicts for each update
    ep_rewards = []   # per-episode returns
    ep_fidelities = []

    obs, _ = env.reset()
    ep_ret = 0.0
    global_step = 0
    update_num  = 0
    t0 = time.time()

    while global_step < args.total_steps:
        # Collect rollout
        for _ in range(args.rollout_steps):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, log_prob, reward, value, float(done))
            obs = next_obs
            ep_ret += reward
            global_step += 1

            if done:
                ep_rewards.append(ep_ret)
                if "fidelity" in info:
                    ep_fidelities.append(info["fidelity"])
                ep_ret = 0.0
                obs, _ = env.reset()

            if global_step >= args.total_steps:
                break

        # PPO update
        last_done = terminated or truncated
        metrics = agent.learn(obs, last_done)
        update_num += 1

        # Logging
        recent_ret = np.mean(ep_rewards[-20:]) if ep_rewards else float("nan")
        recent_fid = np.mean(ep_fidelities[-20:]) if ep_fidelities else float("nan")
        elapsed    = time.time() - t0
        sps        = global_step / elapsed

        log_entry = {
            "step":        global_step,
            "update":      update_num,
            "mean_return": recent_ret,
            "mean_fidelity": recent_fid,
            "policy_loss": metrics["policy_loss"],
            "value_loss":  metrics["value_loss"],
            "entropy":     metrics["entropy"],
            "kl":          metrics["kl"],
            "sps":         sps,
        }
        log.append(log_entry)

        if update_num % 5 == 0 or update_num == 1:
            print(
                f"[step {global_step:>7,}] "
                f"ret={recent_ret:+.4f}  "
                f"fid={recent_fid:.4f}  "
                f"pol={metrics['policy_loss']:+.4f}  "
                f"val={metrics['value_loss']:.4f}  "
                f"ent={metrics['entropy']:.3f}  "
                f"sps={sps:.0f}"
            )

        # Checkpoint
        if update_num % 50 == 0:
            agent.save(os.path.join(out_dir, f"ckpt_{update_num}.pt"))

    # Final save
    agent.save(os.path.join(out_dir, "final_model.pt"))
    with open(os.path.join(out_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete. Results in: {out_dir}")
    print(f"Final mean fidelity (last 20 ep): {np.mean(ep_fidelities[-20:]):.4f}")


if __name__ == "__main__":
    main()
