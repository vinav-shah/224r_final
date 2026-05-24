"""
Modal training + evaluation for RL noise-aware quantum circuit compilation.

All compute runs on Modal. W&B tracks every run.

Usage:
    modal run modal_train.py --dry-run              # preview job matrix
    modal run modal_train.py                        # full 16-run matrix
    modal run modal_train.py --target bell          # single target, all noise levels
    modal run modal_train.py --target bell --noise combined_med
    modal run modal_train.py --skip-eval            # train only
    modal run modal_train.py --download             # pull results volume locally
"""

import os
from pathlib import Path

import modal

# ── Modal primitives ──────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).parent

app = modal.App("quantum-rl-compiler")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "qiskit==2.4.1",
        "qiskit-aer==0.17.2",
        "torch==2.9.1",
        "gymnasium==1.2.3",
        "numpy",
        "matplotlib",
        "wandb",
    )
    .add_local_dir(str(PROJECT_DIR), remote_path="/src")
)

volume = modal.Volume.from_name("quantum-rl-results", create_if_missing=True)
RESULTS_DIR = "/results"

WANDB_PROJECT = "quantum-rl-compiler"
WANDB_SECRET  = modal.Secret.from_name("wandb")

# CPUs matched to Aer's internal parallelism over batched circuits:
#   2-qubit → dim=4 basis states batched → 4 CPUs
#   3-qubit → dim=8 basis states batched → 8 CPUs
# Modal sets OMP_NUM_THREADS automatically from the cpu= request,
# which Aer picks up so its thread pool matches available cores.
_CPU_FOR_NQUBITS = {2: 4, 3: 8}


# ── Training ──────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={RESULTS_DIR: volume},
    secrets=[WANDB_SECRET],
    cpu=8,          # upper bound; actual parallelism set per-job via n_fidelity_workers
    timeout=60 * 60,
)
def train_run(cfg: dict) -> dict:
    import sys, os, json, time
    sys.path.insert(0, "/src")

    import numpy as np
    import wandb
    from noise_models import NOISE_MODELS
    from quantum_env  import QuantumCircuitEnv
    from ppo_agent    import PPOAgent

    run_name = cfg["run_name"]
    out_dir  = f"{RESULTS_DIR}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config=cfg,
        dir=out_dir,
        resume="allow",
    )

    noise_model = None if cfg.get("noiseless") else NOISE_MODELS[cfg["noise"]]()
    env = QuantumCircuitEnv(
        n_qubits=cfg["n_qubits"],
        target_name=cfg["target"],
        noise_model=noise_model,
        max_depth=cfg["max_depth"],
        depth_penalty=cfg["depth_penalty"],
        reward_shaping=cfg.get("reward_shaping", False),
    )

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = PPOAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden=cfg.get("hidden", 256),
        lr=cfg.get("lr", 3e-4),
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=cfg.get("clip_eps", 0.2),
        vf_coef=0.5,
        ent_coef=cfg.get("ent_coef", 0.03),
        max_grad_norm=0.5,
        n_epochs=cfg.get("n_epochs", 10),
        batch_size=cfg.get("batch_size", 64),
        rollout_steps=cfg.get("rollout_steps", 256),
        device="cpu",
    )

    np.random.seed(cfg.get("seed", 42))
    log_entries  = []
    ep_rewards   = []
    ep_fidelities = []
    obs, _ = env.reset()
    ep_ret      = 0.0
    global_step = 0
    update_num  = 0
    done        = False
    total_steps = cfg["total_steps"]
    t0 = time.time()

    while global_step < total_steps:
        for _ in range(cfg.get("rollout_steps", 256)):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, log_prob, reward, value, float(done))
            obs     = next_obs
            ep_ret += reward
            global_step += 1

            if done:
                ep_rewards.append(ep_ret)
                if "fidelity" in info:
                    ep_fidelities.append(info["fidelity"])
                ep_ret = 0.0
                obs, _ = env.reset()

            if global_step >= total_steps:
                break

        metrics    = agent.learn(obs, done)
        update_num += 1
        elapsed    = time.time() - t0
        recent_ret = float(np.mean(ep_rewards[-20:]))   if ep_rewards    else float("nan")
        recent_fid = float(np.mean(ep_fidelities[-20:])) if ep_fidelities else float("nan")

        wandb_log = {
            "train/mean_fidelity": recent_fid,
            "train/mean_return":   recent_ret,
            "train/policy_loss":   metrics["policy_loss"],
            "train/value_loss":    metrics["value_loss"],
            "train/entropy":       metrics["entropy"],
            "train/kl":            metrics["kl"],
            "train/sps":           global_step / elapsed,
            "train/update":        update_num,
        }
        wandb.log(wandb_log, step=global_step)

        log_entries.append({"step": global_step, "update": update_num,
                            "mean_return": recent_ret, "mean_fidelity": recent_fid,
                            **metrics})

        if update_num % 25 == 0:
            print(f"[{run_name}] step={global_step:,}  fid={recent_fid:.4f}  "
                  f"ret={recent_ret:.4f}  ent={metrics['entropy']:.3f}")

        if update_num % 100 == 0:
            agent.save(f"{out_dir}/ckpt_{update_num}.pt")

    agent.save(f"{out_dir}/final_model.pt")
    with open(f"{out_dir}/log.json", "w") as f:
        json.dump(log_entries, f)

    final_fid = float(np.mean(ep_fidelities[-20:])) if ep_fidelities else float("nan")
    wandb.summary["final_fidelity"] = final_fid
    wandb.finish()

    volume.commit()
    print(f"[{run_name}] DONE — final fidelity={final_fid:.4f}")
    return {"run_name": run_name, "final_fidelity": final_fid, "steps": global_step}


# ── Evaluation ────────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={RESULTS_DIR: volume},
    secrets=[WANDB_SECRET],
    cpu=8,
    timeout=30 * 60,
)
def eval_run(cfg: dict) -> dict:
    import sys, os, json
    sys.path.insert(0, "/src")

    import numpy as np
    import torch
    import wandb
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from noise_models import NOISE_MODELS
    from quantum_env  import QuantumCircuitEnv
    from ppo_agent    import PPOAgent
    from baselines    import run_random, run_qiskit_transpiler

    run_name   = cfg["run_name"]
    out_dir    = f"{RESULTS_DIR}/{run_name}"
    model_path = f"{out_dir}/final_model.pt"

    wandb.init(
        project=WANDB_PROJECT,
        name=f"{run_name}_eval",
        config=cfg,
        dir=out_dir,
        resume="allow",
    )

    noise_model = None if cfg.get("noiseless") else NOISE_MODELS[cfg["noise"]]()
    env = QuantumCircuitEnv(
        n_qubits=cfg["n_qubits"],
        target_name=cfg["target"],
        noise_model=noise_model,
        max_depth=cfg["max_depth"],
        depth_penalty=cfg["depth_penalty"],
    )

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n
    n_ep      = cfg.get("n_eval_episodes", 50)

    algorithm = cfg.get("algorithm", "ppo")
    if algorithm == "sac":
        from sac_agent import DiscreteSACAgent
        agent = DiscreteSACAgent(obs_dim=obs_dim, n_actions=n_actions,
                                 hidden=cfg.get("hidden", 256))
        agent.load(model_path)
        def greedy_action(obs):
            return agent.select_action(obs, deterministic=True)
        agent_label = "SAC"
    else:
        agent = PPOAgent(obs_dim=obs_dim, n_actions=n_actions,
                         hidden=cfg.get("hidden", 256))
        agent.load(model_path)
        def greedy_action(obs):
            with torch.no_grad():
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                return agent.net(obs_t)[0].argmax(dim=-1).item()
        agent_label = "PPO"

    # ── Greedy rollout ──
    fidelities, depths, circuits_desc = [], [], []
    for _ in range(n_ep):
        obs, _ = env.reset()
        ep_circ = []
        for _ in range(env.max_depth + 1):
            action = greedy_action(obs)
            ep_circ.append(env.action_name(action))
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                fidelities.append(info.get("fidelity", env._compute_fidelity()))
                depths.append(env._depth)
                circuits_desc.append(ep_circ)
                break

    ppo_res    = {"mean_fidelity": float(np.mean(fidelities)),  # labelled generically; agent_label tracks algorithm
                  "std_fidelity":  float(np.std(fidelities)),
                  "mean_depth":    float(np.mean(depths))}
    random_res = run_random(env, n_ep)
    qiskit_res = run_qiskit_transpiler(env)
    results    = {agent_label: ppo_res, "Random": random_res, "Qiskit": qiskit_res}

    # ── W&B: scalar summary ──
    wandb.summary[f"eval/{algorithm}_fidelity"] = ppo_res["mean_fidelity"]
    wandb.summary["eval/ppo_fidelity"]    = ppo_res["mean_fidelity"]
    wandb.summary["eval/random_fidelity"] = random_res["mean_fidelity"]
    wandb.summary["eval/qiskit_fidelity"] = qiskit_res["mean_fidelity"]
    wandb.summary["eval/ppo_depth"]       = ppo_res["mean_depth"]
    wandb.summary["eval/gain_vs_random"]  = ppo_res["mean_fidelity"] - random_res["mean_fidelity"]

    # ── W&B: fidelity distribution table ──
    fid_table = wandb.Table(columns=["episode", "fidelity", "depth"])
    for i, (f, d) in enumerate(zip(fidelities, depths)):
        fid_table.add_data(i, f, d)
    wandb.log({"eval/fidelity_distribution": fid_table})

    # ── W&B: sample circuits table ──
    circ_table = wandb.Table(columns=["episode", "circuit", "fidelity"])
    for i, (circ, fid) in enumerate(zip(circuits_desc[:10], fidelities[:10])):
        circ_table.add_data(i, " → ".join(circ), fid)
    wandb.log({"eval/sample_circuits": circ_table})

    # ── W&B: noise robustness sweep ──
    noise_keys = ["depolarizing_weak", "depolarizing_med", "depolarizing_strong",
                  "combined_weak", "combined_med", "combined_strong"]
    rob_table  = wandb.Table(columns=["noise_model", "ppo_fidelity", "random_fidelity", "gain"])
    ppo_rob, rand_rob = [], []

    for nk in noise_keys:
        nm   = NOISE_MODELS[nk]()
        env2 = QuantumCircuitEnv(n_qubits=cfg["n_qubits"], target_name=cfg["target"],
                                 noise_model=nm, max_depth=cfg["max_depth"],
                                 depth_penalty=cfg["depth_penalty"])
        fids2 = []
        for _ in range(20):
            obs2, _ = env2.reset()
            for _ in range(env2.max_depth + 1):
                with torch.no_grad():
                    obs_t2 = torch.as_tensor(obs2, dtype=torch.float32).unsqueeze(0)
                    action2 = agent.net(obs_t2)[0].argmax(dim=-1).item()
                obs2, _, term2, trunc2, info2 = env2.step(action2)
                if term2 or trunc2:
                    fids2.append(info2.get("fidelity", env2._compute_fidelity()))
                    break
        pf = float(np.mean(fids2))
        rf = run_random(env2, 20)["mean_fidelity"]
        ppo_rob.append(pf);  rand_rob.append(rf)
        rob_table.add_data(nk, pf, rf, pf - rf)

    wandb.log({"eval/noise_robustness": rob_table})

    # ── Plots (also uploaded to W&B) ──
    noise_tag = cfg.get("noise", "noiseless")

    # Bar chart
    names  = list(results.keys())
    means  = [results[n]["mean_fidelity"] for n in names]
    stds   = [results[n]["std_fidelity"]  for n in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, means, yerr=stds, capsize=5,
                  color=["#4C72B0", "#DD8452", "#55A868"], alpha=0.85)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Average Fidelity")
    ax.set_title(f"{cfg['target']} ({cfg['n_qubits']}q, {noise_tag})")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    bar_path = f"{out_dir}/fidelity_comparison.png"
    plt.savefig(bar_path, dpi=150); plt.close()
    wandb.log({"eval/fidelity_comparison": wandb.Image(bar_path)})

    # Learning curve (from training log)
    log_path = f"{out_dir}/log.json"
    if os.path.exists(log_path):
        with open(log_path) as f:
            train_log = json.load(f)
        steps = [e["step"]         for e in train_log]
        fids  = [e["mean_fidelity"] for e in train_log]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, fids, color="#4C72B0", linewidth=1.5)
        ax.set_xlabel("Environment steps"); ax.set_ylabel("Mean fidelity (last 20 ep)")
        ax.set_title(f"Learning curve — {cfg['target']} ({cfg['n_qubits']}q, {noise_tag})")
        ax.set_ylim(0, 1.05); plt.tight_layout()
        lc_path = f"{out_dir}/learning_curve.png"
        plt.savefig(lc_path, dpi=150); plt.close()
        wandb.log({"eval/learning_curve": wandb.Image(lc_path)})

    # Noise robustness line chart
    x = list(range(len(noise_keys)))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, ppo_rob,  "o-",  label="PPO",    color="#4C72B0")
    ax.plot(x, rand_rob, "s--", label="Random",  color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(noise_keys, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Mean Fidelity")
    ax.set_title(f"Robustness — {cfg['target']} ({cfg['n_qubits']}q)")
    ax.legend(); ax.set_ylim(0, 1.05); plt.tight_layout()
    rob_path = f"{out_dir}/noise_robustness.png"
    plt.savefig(rob_path, dpi=150); plt.close()
    wandb.log({"eval/noise_robustness_plot": wandb.Image(rob_path)})

    # Save sample circuits
    with open(f"{out_dir}/sample_circuits.txt", "w") as f:
        for i, (circ, fid) in enumerate(zip(circuits_desc[:10], fidelities[:10])):
            f.write(f"Episode {i+1} (fid={fid:.4f}): {' → '.join(circ)}\n")

    save = {k: {kk: vv for kk, vv in v.items() if kk not in ("circuits", "circuit")}
            for k, v in results.items()}
    with open(f"{out_dir}/eval_results.json", "w") as f:
        json.dump(save, f, indent=2)

    wandb.finish()
    volume.commit()

    print(f"[{run_name}] eval — PPO={ppo_res['mean_fidelity']:.4f}  "
          f"Random={random_res['mean_fidelity']:.4f}  Qiskit={qiskit_res['mean_fidelity']:.4f}")
    return results


# ── SAC Training ─────────────────────────────────────────────────────────────

@app.function(
    image=image,
    volumes={RESULTS_DIR: volume},
    secrets=[WANDB_SECRET],
    cpu=8,
    timeout=60 * 60,
)
def train_run_sac(cfg: dict) -> dict:
    """Discrete SAC training. Off-policy replay buffer prevents catastrophic forgetting."""
    import sys, os, json, time
    sys.path.insert(0, "/src")

    import numpy as np
    import wandb
    from noise_models  import NOISE_MODELS
    from quantum_env   import QuantumCircuitEnv
    from sac_agent     import DiscreteSACAgent

    run_name = cfg["run_name"]
    out_dir  = f"{RESULTS_DIR}/{run_name}"
    os.makedirs(out_dir, exist_ok=True)

    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        config=cfg,
        dir=out_dir,
        resume="allow",
    )

    noise_model = None if cfg.get("noiseless") else NOISE_MODELS[cfg["noise"]]()
    env = QuantumCircuitEnv(
        n_qubits=cfg["n_qubits"],
        target_name=cfg["target"],
        noise_model=noise_model,
        max_depth=cfg["max_depth"],
        depth_penalty=cfg["depth_penalty"],
    )

    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DiscreteSACAgent(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden=cfg.get("hidden", 256),
        lr_actor=cfg.get("lr", 3e-4),
        lr_critic=cfg.get("lr", 3e-4),
        lr_alpha=cfg.get("lr", 3e-4),
        gamma=0.99,
        tau=cfg.get("tau", 0.005),
        batch_size=cfg.get("batch_size", 256),
        buffer_size=cfg.get("buffer_size", 100_000),
        learning_starts=cfg.get("learning_starts", 1_000),
        target_entropy_ratio=cfg.get("target_entropy_ratio", 0.6),
        device="cpu",
    )

    np.random.seed(cfg.get("seed", 42))
    log_entries   = []
    ep_rewards    = []
    ep_fidelities = []
    obs, _ = env.reset()
    ep_ret      = 0.0
    global_step = 0
    total_steps = cfg["total_steps"]
    log_interval = cfg.get("log_interval", 1_000)
    ckpt_interval = cfg.get("ckpt_interval", 50_000)
    t0 = time.time()

    # Running metric windows for logging
    recent_metrics = {"critic_loss": [], "actor_loss": [], "alpha_loss": [],
                      "alpha": [], "entropy": []}

    while global_step < total_steps:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store(obs, action, reward, next_obs, done)
        obs     = next_obs
        ep_ret += reward
        global_step += 1

        if done:
            ep_rewards.append(ep_ret)
            if "fidelity" in info:
                ep_fidelities.append(info["fidelity"])
            ep_ret = 0.0
            obs, _ = env.reset()

        # One gradient update per env step
        metrics = agent.update()
        for k, v in metrics.items():
            recent_metrics[k].append(v)

        if global_step % log_interval == 0:
            elapsed    = time.time() - t0
            recent_ret = float(np.mean(ep_rewards[-20:]))    if ep_rewards    else float("nan")
            recent_fid = float(np.mean(ep_fidelities[-20:])) if ep_fidelities else float("nan")
            avg = {k: float(np.mean(v)) if v else float("nan") for k, v in recent_metrics.items()}

            wandb.log({
                "train/mean_fidelity": recent_fid,
                "train/mean_return":   recent_ret,
                "train/critic_loss":   avg["critic_loss"],
                "train/actor_loss":    avg["actor_loss"],
                "train/alpha":         avg["alpha"],
                "train/entropy":       avg["entropy"],
                "train/sps":           global_step / elapsed,
            }, step=global_step)

            log_entries.append({"step": global_step, "mean_return": recent_ret,
                                "mean_fidelity": recent_fid, **avg})

            print(f"[{run_name}] step={global_step:,}  fid={recent_fid:.4f}  "
                  f"α={avg['alpha']:.4f}  ent={avg['entropy']:.3f}  "
                  f"sps={global_step/elapsed:.0f}")

            for k in recent_metrics:
                recent_metrics[k].clear()

        if global_step % ckpt_interval == 0:
            agent.save(f"{out_dir}/ckpt_{global_step}.pt")

    agent.save(f"{out_dir}/final_model.pt")
    with open(f"{out_dir}/log.json", "w") as f:
        json.dump(log_entries, f)

    final_fid = float(np.mean(ep_fidelities[-20:])) if ep_fidelities else float("nan")
    wandb.summary["final_fidelity"] = final_fid
    wandb.finish()
    volume.commit()

    print(f"[{run_name}] DONE — final fidelity={final_fid:.4f}")
    return {"run_name": run_name, "final_fidelity": final_fid, "steps": global_step}


# ── Summary helper ────────────────────────────────────────────────────────────

@app.function(image=image, volumes={RESULTS_DIR: volume})
def list_results() -> list:
    import os, json
    runs = []
    for run_name in sorted(os.listdir(RESULTS_DIR)):
        p = f"{RESULTS_DIR}/{run_name}/eval_results.json"
        if os.path.exists(p):
            with open(p) as f:
                ev = json.load(f)
            runs.append({"run": run_name, **ev})
    return runs


# ── Experiment matrix ─────────────────────────────────────────────────────────

def make_configs(target_filter=None, noise_filter=None, seeds=(42,)) -> list:
    experiments = [
        dict(target="bell", n_qubits=2, max_depth=15, total_steps=300_000,
             depth_penalty=0.005, ent_coef=0.03),
        dict(target="swap", n_qubits=2, max_depth=20, total_steps=300_000,
             depth_penalty=0.005, ent_coef=0.05),
        dict(target="qft",  n_qubits=2, max_depth=20, total_steps=300_000,
             depth_penalty=0.005, ent_coef=0.03),
        dict(target="ghz",  n_qubits=3, max_depth=25, total_steps=400_000,
             depth_penalty=0.003, ent_coef=0.05, hidden=512, rollout_steps=512),
    ]
    noise_models = ["combined_weak", "combined_med", "combined_strong", "depolarizing_strong"]

    cfgs = []
    for exp in experiments:
        if target_filter and exp["target"] not in target_filter:
            continue
        for noise in noise_models:
            if noise_filter and noise not in noise_filter:
                continue
            for seed in seeds:
                n_qubits = exp["n_qubits"]
                cfg = {
                    "noise": noise, "seed": seed,
                    "lr": 3e-4, "n_epochs": 10, "batch_size": 64, "clip_eps": 0.2,
                    "n_eval_episodes": 50,
                    "hidden":        exp.get("hidden", 256),
                    "rollout_steps": exp.get("rollout_steps", 256),
                    **{k: v for k, v in exp.items() if k not in ("hidden", "rollout_steps")},
                }
                cfg["run_name"] = f"{cfg['target']}_n{cfg['n_qubits']}_{noise}_s{seed}"
                cfgs.append(cfg)
    return cfgs


# ── SAC experiment matrix ─────────────────────────────────────────────────────

def make_sac_configs(target_filter=None, noise_filter=None, seeds=(42,)) -> list:
    experiments = [
        dict(target="bell", n_qubits=2, max_depth=15, total_steps=300_000,
             depth_penalty=0.005),
        dict(target="swap", n_qubits=2, max_depth=20, total_steps=300_000,
             depth_penalty=0.005),
        dict(target="qft",  n_qubits=2, max_depth=20, total_steps=300_000,
             depth_penalty=0.005),
        dict(target="ghz",  n_qubits=3, max_depth=25, total_steps=400_000,
             depth_penalty=0.003, hidden=512),
    ]
    noise_models = ["combined_weak", "combined_med", "combined_strong", "depolarizing_strong"]

    cfgs = []
    for exp in experiments:
        if target_filter and exp["target"] not in target_filter:
            continue
        for noise in noise_models:
            if noise_filter and noise not in noise_filter:
                continue
            for seed in seeds:
                cfg = {
                    "algorithm": "sac",
                    "noise": noise, "seed": seed,
                    "lr": 3e-4,
                    "batch_size": 256,
                    "buffer_size": 100_000,
                    "learning_starts": 1_000,
                    "target_entropy_ratio": 0.6,
                    "tau": 0.005,
                    "n_eval_episodes": 50,
                    "hidden": exp.get("hidden", 256),
                    **{k: v for k, v in exp.items() if k != "hidden"},
                }
                cfg["run_name"] = f"{cfg['target']}_n{cfg['n_qubits']}_{noise}_sac_s{seed}"
                cfgs.append(cfg)
    return cfgs


# ── Entrypoint ────────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(
    target:    str  = "",
    noise:     str  = "",
    seeds:     str  = "42",
    dry_run:   bool = False,
    skip_eval: bool = False,
    download:  bool = False,
    sac:       bool = False,   # launch SAC runs (can combine with PPO)
    sac_only:  bool = False,   # launch only SAC, no PPO
):
    """
    --target bell,swap,ghz,qft   filter targets (comma-separated)
    --noise  combined_med,...     filter noise models (comma-separated)
    --seeds  42,1,2               comma-separated seeds
    --dry-run                     preview jobs, don't submit
    --skip-eval                   train only
    --download                    pull results volume to ./modal_results/
    --sac                         also launch 16 SAC runs in parallel with PPO
    --sac-only                    launch only the 16 SAC runs
    """
    if download:
        import subprocess
        subprocess.run(["modal", "volume", "get", "quantum-rl-results", "/", "./modal_results/"])
        return

    target_filter = [t.strip() for t in target.split(",") if t.strip()] or None
    noise_filter  = [n.strip() for n in noise.split(",")  if n.strip()] or None
    seed_list     = [int(s.strip()) for s in seeds.split(",")]

    ppo_cfgs = [] if sac_only else make_configs(target_filter, noise_filter, seed_list)
    sac_cfgs = make_sac_configs(target_filter, noise_filter, seed_list) if (sac or sac_only) else []

    print(f"\n{'='*60}")
    if ppo_cfgs:
        print(f"PPO runs: {len(ppo_cfgs)}")
        for c in ppo_cfgs:
            print(f"  {c['run_name']:55s}  steps={c['total_steps']:,}")
    if sac_cfgs:
        print(f"SAC runs: {len(sac_cfgs)}")
        for c in sac_cfgs:
            print(f"  {c['run_name']:55s}  steps={c['total_steps']:,}")
    print(f"{'='*60}\n")

    if dry_run:
        print("Dry run — not submitting.")
        return

    # Launch PPO and SAC in parallel (Modal maps run concurrently)
    import concurrent.futures
    futures = {}
    with concurrent.futures.ThreadPoolExecutor() as pool:
        if ppo_cfgs:
            futures["ppo"] = pool.submit(lambda: list(train_run.map(ppo_cfgs, order_outputs=False)))
        if sac_cfgs:
            futures["sac"] = pool.submit(lambda: list(train_run_sac.map(sac_cfgs, order_outputs=False)))

    all_train = []
    for algo, fut in futures.items():
        results = fut.result()
        print(f"\n=== {algo.upper()} Training complete ===")
        for r in sorted(results, key=lambda x: x["run_name"]):
            print(f"  {r['run_name']:55s}  fidelity={r['final_fidelity']:.4f}")
        all_train.extend(results)

    if skip_eval:
        return

    # Evaluate all trained runs
    all_cfgs = ppo_cfgs + sac_cfgs
    print("\nSubmitting evaluation jobs (parallel)...")
    list(eval_run.map(all_cfgs, order_outputs=False))

    all_results = list(list_results.remote())
    if all_results:
        print("\n=== Summary ===")
        print(f"{'Run':<57} {'Agent':>6} {'Random':>8} {'Qiskit':>8}")
        print("-" * 85)
        for r in sorted(all_results, key=lambda x: x["run"]):
            algo   = "SAC" if "SAC" in r else "PPO"
            agent  = r.get("SAC", r.get("PPO", {})).get("mean_fidelity", float("nan"))
            rand   = r.get("Random", {}).get("mean_fidelity", float("nan"))
            qiskit = r.get("Qiskit", {}).get("mean_fidelity", float("nan"))
            print(f"  {r['run']:<55} {agent:>6.4f} {rand:>8.4f} {qiskit:>8.4f}")
