"""
Custom PPO implementation using PyTorch.

Architecture:
  - Shared MLP backbone
  - Separate policy head (categorical) and value head (scalar)
  - GAE advantage estimation
  - Clipped surrogate objective + entropy bonus + value loss
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Tuple, Optional


# ----- Network --------------------------------------------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head  = nn.Linear(hidden, 1)

        # Orthogonal init (standard for PPO)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        logits = self.policy_head(h)
        value  = self.value_head(h).squeeze(-1)
        return logits, value

    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy, value


# ----- Rollout buffer -------------------------------------------------------

class RolloutBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self.obs      = torch.zeros(capacity, obs_dim,  device=device)
        self.actions  = torch.zeros(capacity,           device=device, dtype=torch.long)
        self.log_probs= torch.zeros(capacity,           device=device)
        self.rewards  = torch.zeros(capacity,           device=device)
        self.values   = torch.zeros(capacity,           device=device)
        self.dones    = torch.zeros(capacity,           device=device)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i]       = torch.as_tensor(obs,      device=self.device, dtype=torch.float32)
        self.actions[i]   = torch.as_tensor(action,   device=self.device)
        self.log_probs[i] = torch.as_tensor(log_prob, device=self.device)
        self.rewards[i]   = torch.as_tensor(reward,   device=self.device, dtype=torch.float32)
        self.values[i]    = torch.as_tensor(value,    device=self.device)
        self.dones[i]     = torch.as_tensor(done,     device=self.device, dtype=torch.float32)
        self.ptr += 1

    def full(self):
        return self.ptr >= self.capacity

    def reset(self):
        self.ptr = 0

    def compute_returns_and_advantages(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = self.ptr
        advantages = torch.zeros(T, device=self.device)
        last_gae = 0.0
        last_val  = torch.tensor(last_value, device=self.device)

        for t in reversed(range(T)):
            next_val  = last_val if t == T - 1 else self.values[t + 1]
            next_done = self.dones[t]
            delta     = self.rewards[t] + gamma * next_val * (1 - next_done) - self.values[t]
            last_gae  = delta + gamma * gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values[:T]
        return returns, advantages


# ----- PPO agent ------------------------------------------------------------

class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        rollout_steps: int = 512,
        device: str = "cpu",
    ):
        self.gamma        = gamma
        self.gae_lambda   = gae_lambda
        self.clip_eps     = clip_eps
        self.vf_coef      = vf_coef
        self.ent_coef     = ent_coef
        self.max_grad_norm= max_grad_norm
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.rollout_steps= rollout_steps

        self.device = torch.device(device)
        self.net    = ActorCritic(obs_dim, n_actions, hidden).to(self.device)
        self.opt    = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer(rollout_steps, obs_dim, self.device)

    def select_action(self, obs: np.ndarray):
        """Sample action; return (action, log_prob, value) as Python scalars."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
            action, log_prob, _, value = self.net.get_action_and_value(obs_t)
        return action.item(), log_prob.item(), value.item()

    def store(self, obs, action, log_prob, reward, value, done):
        self.buffer.add(obs, action, log_prob, reward, value, done)

    def learn(self, last_obs: np.ndarray, last_done: bool) -> dict:
        """Run PPO update on collected rollout. Returns dict of loss metrics."""
        with torch.no_grad():
            obs_t = torch.as_tensor(last_obs, device=self.device, dtype=torch.float32).unsqueeze(0)
            _, _, _, last_val = self.net.get_action_and_value(obs_t)
            last_val = last_val.item() * (1.0 - float(last_done))

        T = self.buffer.ptr
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_val, self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_b    = self.buffer.obs[:T]
        act_b    = self.buffer.actions[:T]
        logp_b   = self.buffer.log_probs[:T]
        ret_b    = returns
        adv_b    = advantages

        metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": []}

        for _ in range(self.n_epochs):
            perm = torch.randperm(T, device=self.device)
            for start in range(0, T, self.batch_size):
                idx = perm[start: start + self.batch_size]
                _, new_logp, entropy, new_val = self.net.get_action_and_value(obs_b[idx], act_b[idx])

                ratio    = (new_logp - logp_b[idx]).exp()
                surr1    = ratio * adv_b[idx]
                surr2    = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv_b[idx]
                pol_loss = -torch.min(surr1, surr2).mean()
                val_loss = 0.5 * (new_val - ret_b[idx]).pow(2).mean()
                ent_loss = -entropy.mean()

                loss = pol_loss + self.vf_coef * val_loss + self.ent_coef * ent_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = (logp_b[idx] - new_logp).mean().item()

                metrics["policy_loss"].append(pol_loss.item())
                metrics["value_loss"].append(val_loss.item())
                metrics["entropy"].append(-ent_loss.item())
                metrics["kl"].append(kl)

        self.buffer.reset()
        return {k: float(np.mean(v)) for k, v in metrics.items()}

    def save(self, path: str):
        torch.save({
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "arch": {"obs_dim": self.net.backbone[0].in_features,
                     "n_actions": self.net.policy_head.out_features,
                     "hidden": self.net.backbone[0].out_features},
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        # Rebuild network from saved arch if it differs from current
        if "arch" in ckpt:
            arch = ckpt["arch"]
            if self.net.backbone[0].out_features != arch["hidden"]:
                self.net = ActorCritic(
                    arch["obs_dim"], arch["n_actions"], arch["hidden"]
                ).to(self.device)
        self.net.load_state_dict(ckpt["net"])
        self.opt.load_state_dict(ckpt["opt"])
