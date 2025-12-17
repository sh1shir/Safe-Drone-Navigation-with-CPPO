# cppo/ppo.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(128,128)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        self.net = nn.Sequential(*layers)
        self.mean = nn.Linear(prev, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = self.net(x)
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(128,128)):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# Rollout buffer with GAE (reward only, no cost)
class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device="cpu"):
        self.size = size
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.val = np.zeros(size, dtype=np.float32)

        self.ptr = 0
        self.gamma = gamma
        self.lam = lam
        self.device = device

        # arrays for computed advantages
        self.adv = np.zeros(size, dtype=np.float32)
        self.ret = np.zeros(size, dtype=np.float32)

    def store(self, obs, act, rew, logp, done, val):
        assert self.ptr < self.size
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.logp[self.ptr] = logp
        self.dones[self.ptr] = done
        self.val[self.ptr] = val
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        # compute GAE for reward
        path_slice = slice(0, self.ptr)
        rews = np.append(self.rews[path_slice], last_val)
        vals = np.append(self.val[path_slice], last_val)
        dones = np.append(self.dones[path_slice], 0)
        gae = 0
        for t in reversed(range(len(rews)-1)):
            delta = rews[t] + self.gamma * vals[t+1] * (1-dones[t]) - vals[t]
            gae = delta + self.gamma * self.lam * (1-dones[t]) * gae
            self.adv[t] = gae
        self.ret[path_slice] = self.adv[path_slice] + self.val[path_slice]

    def get(self):
        assert self.ptr == self.size, "buffer not full"
        self.ptr = 0
        data = dict(
            obs = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device),
            acts = torch.as_tensor(self.acts, dtype=torch.float32, device=self.device),
            logp = torch.as_tensor(self.logp, dtype=torch.float32, device=self.device),
            ret = torch.as_tensor(self.ret, dtype=torch.float32, device=self.device),
            adv = torch.as_tensor(self.adv, dtype=torch.float32, device=self.device),
        )
        # normalize advantages
        data["adv"] = (data["adv"] - data["adv"].mean()) / (data["adv"].std() + 1e-8)
        return data


# Standard PPO algorithm (no constraints)
class PPO:
    def __init__(self,
                 obs_dim,
                 act_dim,
                 device="cpu",
                 steps_per_epoch=4096,
                 epochs=50,
                 gamma=0.99,
                 lam=0.95,
                 pi_lr=3e-4,
                 vf_lr=1e-4,
                 clip=0.2,
                 target_kl=0.015):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.target_kl = target_kl

        # networks (only one value function)
        self.policy = GaussianPolicy(obs_dim, act_dim).to(device)
        self.value = Critic(obs_dim).to(device)

        # optimizers
        self.pi_opt = optim.Adam(self.policy.parameters(), lr=pi_lr)
        self.v_opt = optim.Adam(self.value.parameters(), lr=vf_lr)

        # buffer (no cost tracking)
        self.buffer = RolloutBuffer(obs_dim, act_dim, steps_per_epoch, gamma=gamma, lam=lam, device=device)

    def start_epoch(self):
        pass

    def select_action(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.policy(obs_t)
            val = self.value(obs_t).cpu().numpy().squeeze()
        
        dist = torch.distributions.Normal(mean, std)
        a_t = dist.sample()
        logp_t = dist.log_prob(a_t).sum(axis=-1)
        action = a_t.squeeze(0).cpu().numpy()
        
        return action, float(logp_t.cpu().numpy()), float(val)

    def compute_values(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            val = self.value(obs_t).cpu().numpy().squeeze()
        return float(val)

    def store_step(self, obs, act, rew, logp, done, val):
        self.buffer.store(obs, act, rew, logp, float(done), val)

    def finish_epoch(self, last_val=0.0):
        # Fill buffer if not full
        if self.buffer.ptr < self.buffer.size:
            fill = self.buffer.size - self.buffer.ptr
            for _ in range(fill):
                self.buffer.store(np.zeros(self.obs_dim, dtype=np.float32),
                                  np.zeros(self.act_dim, dtype=np.float32),
                                  0.0, 0.0, 1.0, 0.0)
        self.buffer.finish_path(last_val)

    def update(self):
        data = self.buffer.get()
        obs = data["obs"]
        acts = data["acts"]
        old_logp = data["logp"]
        adv = data["adv"]
        ret = data["ret"]
        
        # Policy update
        pi_loss = 0.0
        for epoch in range(10):
            mean, std = self.policy(obs)
            if torch.any(torch.isnan(mean)) or torch.any(torch.isnan(std)):
                print(f"NaN detected at epoch {epoch}, stopping policy update")
                break
            
            dist = torch.distributions.Normal(mean, std)
            new_logp = dist.log_prob(acts).sum(axis=-1)
            ratio = torch.exp(new_logp - old_logp)
            
            # KL divergence check for early stopping
            kl = (old_logp - new_logp).mean()
            if kl > self.target_kl:
                print(f"KL divergence {kl:.4f} too high, stopping at epoch {epoch}")
                break
            
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1-self.clip, 1+self.clip) * adv
            pi_loss = -torch.min(unclipped, clipped).mean()
            
            self.pi_opt.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.pi_opt.step()
        
        # Value function update
        loss_v = 0.0
        for _ in range(10):
            v_pred = self.value(obs)
            loss_v = ((v_pred - ret) ** 2).mean()
            
            self.v_opt.zero_grad()
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), max_norm=0.5)
            self.v_opt.step()
        
        return {
            "pi_loss": float(pi_loss.cpu().item()) if isinstance(pi_loss, torch.Tensor) else float(pi_loss),
            "loss_v": float(loss_v.cpu().item()) if isinstance(loss_v, torch.Tensor) else float(loss_v),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "value": self.value.state_dict(),
        }, path)

    def load(self, path, map_location=None):
        data = torch.load(path, map_location=map_location)
        self.policy.load_state_dict(data["policy"])
        self.value.load_state_dict(data["value"])
