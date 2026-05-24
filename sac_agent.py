"""
Discrete SAC (Soft Actor-Critic for Discrete Action Settings).
Christodoulou 2019: https://arxiv.org/abs/1910.07207

Key differences from PPO that fix the GHZ forgetting problem:
  - Off-policy replay buffer: good trajectories are never discarded
  - Two Q-networks: reduces Q overestimation bias
  - Automatic temperature α: adapts exploration/exploitation balance
    so entropy never collapses catastrophically
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# ── Networks ──────────────────────────────────────────────────────────────────

class Actor(nn.Module):
    """Policy network: obs → categorical distribution over actions."""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        logits   = self.net(obs)
        probs    = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


class Critic(nn.Module):
    """Q-network: obs → Q-values for every action simultaneously."""
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, hidden),  nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, device: torch.device):
        self.capacity = capacity
        self.device   = device
        self.ptr = 0
        self.size = 0
        self.obs      = torch.zeros(capacity, obs_dim,  device=device)
        self.next_obs = torch.zeros(capacity, obs_dim,  device=device)
        self.actions  = torch.zeros(capacity,           device=device, dtype=torch.long)
        self.rewards  = torch.zeros(capacity,           device=device)
        self.dones    = torch.zeros(capacity,           device=device)

    def add(self, obs, action, reward, next_obs, done):
        i = self.ptr
        self.obs[i]      = torch.as_tensor(obs,      dtype=torch.float32, device=self.device)
        self.next_obs[i] = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        self.actions[i]  = int(action)
        self.rewards[i]  = float(reward)
        self.dones[i]    = float(done)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )

    def ready(self, min_size: int) -> bool:
        return self.size >= min_size


# ── Discrete SAC agent ────────────────────────────────────────────────────────

class DiscreteSACAgent:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,          # soft target update rate
        batch_size: int = 256,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        target_entropy_ratio: float = 0.6,  # fraction of max entropy as target
        device: str = "cpu",
    ):
        self.gamma          = gamma
        self.tau            = tau
        self.batch_size     = batch_size
        self.learning_starts = learning_starts
        self.n_actions      = n_actions

        self.device = torch.device(device)

        # Networks
        self.actor    = Actor(obs_dim, n_actions, hidden).to(self.device)
        self.critic1  = Critic(obs_dim, n_actions, hidden).to(self.device)
        self.critic2  = Critic(obs_dim, n_actions, hidden).to(self.device)
        self.target1  = Critic(obs_dim, n_actions, hidden).to(self.device)
        self.target2  = Critic(obs_dim, n_actions, hidden).to(self.device)
        self.target1.load_state_dict(self.critic1.state_dict())
        self.target2.load_state_dict(self.critic2.state_dict())
        for p in self.target1.parameters(): p.requires_grad_(False)
        for p in self.target2.parameters(): p.requires_grad_(False)

        # Automatic temperature: target entropy = ratio * log(n_actions)
        self.target_entropy = target_entropy_ratio * np.log(n_actions)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)

        # Optimisers
        self.actor_opt  = optim.Adam(self.actor.parameters(),   lr=lr_actor)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr_critic,
        )
        self.alpha_opt  = optim.Adam([self.log_alpha], lr=lr_alpha)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size, obs_dim, self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            probs, _ = self.actor(obs_t)
            if deterministic:
                return probs.argmax(dim=-1).item()
            return Categorical(probs).sample().item()

    def store(self, obs, action, reward, next_obs, done):
        self.buffer.add(obs, action, reward, next_obs, done)

    def update(self) -> dict:
        if not self.buffer.ready(self.learning_starts):
            return {}

        obs, actions, rewards, next_obs, dones = self.buffer.sample(self.batch_size)

        with torch.no_grad():
            next_probs, next_log_probs = self.actor(next_obs)
            q1_next = self.target1(next_obs)
            q2_next = self.target2(next_obs)
            min_q_next = torch.min(q1_next, q2_next)
            # V(s') = E_π[Q(s',a') - α*log π(a'|s')]
            v_next = (next_probs * (min_q_next - self.alpha * next_log_probs)).sum(dim=-1)
            q_target = rewards + self.gamma * (1 - dones) * v_next

        # ── Critic update ──
        q1_all = self.critic1(obs)
        q2_all = self.critic2(obs)
        q1 = q1_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2 = q2_all.gather(1, actions.unsqueeze(1)).squeeze(1)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 1.0
        )
        self.critic_opt.step()

        # ── Actor update ──
        probs, log_probs = self.actor(obs)
        with torch.no_grad():
            q1_pi = self.critic1(obs)
            q2_pi = self.critic2(obs)
            min_q_pi = torch.min(q1_pi, q2_pi)
        # E_π[α*log π - Q] over all actions
        actor_loss = (probs * (self.alpha.detach() * log_probs - min_q_pi)).sum(dim=-1).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Temperature update ──
        entropy = -(probs.detach() * log_probs.detach()).sum(dim=-1).mean()
        alpha_loss = self.log_alpha * (entropy - self.target_entropy).detach()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ── Soft target update ──
        for p, pt in zip(self.critic1.parameters(), self.target1.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)
        for p, pt in zip(self.critic2.parameters(), self.target2.parameters()):
            pt.data.copy_(self.tau * p.data + (1 - self.tau) * pt.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss.item(),
            "alpha":       self.alpha.item(),
            "entropy":     entropy.item(),
        }

    def save(self, path: str):
        torch.save({
            "actor":   self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target1": self.target1.state_dict(),
            "target2": self.target2.state_dict(),
            "log_alpha": self.log_alpha.data,
            "arch": {"obs_dim":   self.actor.net[0].in_features,
                     "n_actions": self.actor.net[-1].out_features,
                     "hidden":    self.actor.net[0].out_features},
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        if "arch" in ckpt:
            arch = ckpt["arch"]
            if self.actor.net[0].out_features != arch["hidden"]:
                self.actor   = Actor( arch["obs_dim"], arch["n_actions"], arch["hidden"]).to(self.device)
                self.critic1 = Critic(arch["obs_dim"], arch["n_actions"], arch["hidden"]).to(self.device)
                self.critic2 = Critic(arch["obs_dim"], arch["n_actions"], arch["hidden"]).to(self.device)
                self.target1 = Critic(arch["obs_dim"], arch["n_actions"], arch["hidden"]).to(self.device)
                self.target2 = Critic(arch["obs_dim"], arch["n_actions"], arch["hidden"]).to(self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.target1.load_state_dict(ckpt["target1"])
        self.target2.load_state_dict(ckpt["target2"])
        self.log_alpha.data.copy_(ckpt["log_alpha"])
