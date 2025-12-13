# cppo/cppo.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

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


# small rollout buffer with GAE for reward and cost
class RolloutBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95, device="cpu"):
        self.size = size
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts = np.zeros((size, act_dim), dtype=np.float32)
        self.rews = np.zeros(size, dtype=np.float32)
        self.costs = np.zeros(size, dtype=np.float32)
        self.logp = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.val_r = np.zeros(size, dtype=np.float32)
        self.val_c = np.zeros(size, dtype=np.float32)

        self.ptr = 0
        self.gamma = gamma
        self.lam = lam
        self.device = device

        # arrays for computed advantages
        self.adv_r = np.zeros(size, dtype=np.float32)
        self.adv_c = np.zeros(size, dtype=np.float32)
        self.ret_r = np.zeros(size, dtype=np.float32)
        self.ret_c = np.zeros(size, dtype=np.float32)

    def store(self, obs, act, rew, cost, logp, done, v_r, v_c):
        assert self.ptr < self.size
        self.obs[self.ptr] = obs
        self.acts[self.ptr] = act
        self.rews[self.ptr] = rew
        self.costs[self.ptr] = cost
        self.logp[self.ptr] = logp
        self.dones[self.ptr] = done
        self.val_r[self.ptr] = v_r
        self.val_c[self.ptr] = v_c
        self.ptr += 1

    def finish_path(self, last_val_r=0.0, last_val_c=0.0):
        # compute GAE reward
        path_slice = slice(0, self.ptr)
        rews = np.append(self.rews[path_slice], last_val_r)
        vals_r = np.append(self.val_r[path_slice], last_val_r)
        dones = np.append(self.dones[path_slice], 0)
        gae = 0
        for t in reversed(range(len(rews)-1)):
            delta = rews[t] + self.gamma * vals_r[t+1] * (1-dones[t]) - vals_r[t]
            gae = delta + self.gamma * self.lam * (1-dones[t]) * gae
            self.adv_r[t] = gae
        self.ret_r[path_slice] = self.adv_r[path_slice] + self.val_r[path_slice]

        # compute GAE cost
        costs = np.append(self.costs[path_slice], last_val_c)
        vals_c = np.append(self.val_c[path_slice], last_val_c)
        gae = 0
        for t in reversed(range(len(costs)-1)):
            delta = costs[t] + self.gamma * vals_c[t+1] * (1-dones[t]) - vals_c[t]
            gae = delta + self.gamma * self.lam * (1-dones[t]) * gae
            self.adv_c[t] = gae
        self.ret_c[path_slice] = self.adv_c[path_slice] + self.val_c[path_slice]

    def get(self):
        assert self.ptr == self.size, "buffer not full"
        self.ptr = 0
        data = dict(
            obs = torch.as_tensor(self.obs, dtype=torch.float32, device=self.device),
            acts = torch.as_tensor(self.acts, dtype=torch.float32, device=self.device),
            logp = torch.as_tensor(self.logp, dtype=torch.float32, device=self.device),
            ret_r = torch.as_tensor(self.ret_r, dtype=torch.float32, device=self.device),
            ret_c = torch.as_tensor(self.ret_c, dtype=torch.float32, device=self.device),
            adv_r = torch.as_tensor(self.adv_r, dtype=torch.float32, device=self.device),
            adv_c = torch.as_tensor(self.adv_c, dtype=torch.float32, device=self.device),
            costs = torch.as_tensor(self.costs, dtype=torch.float32, device=self.device),
        )
        # normalize advantages
        data["adv_r_norm"] = (data["adv_r"] - data["adv_r"].mean()) / (data["adv_r"].std() + 1e-8)
        data["adv_c_norm"] = (data["adv_c"] - data["adv_c"].mean()) / (data["adv_c"].std() + 1e-8)
        return data


# Lagrangian
class Lagrangian:
    def __init__(self, init_lambda=0.1, lr=0.02, max_lambda=1e6, device="cpu"):
        self.device = device
        self.lambda_param = float(init_lambda)
        self.lr = float(lr)
        self.max_lambda = float(max_lambda)

    def update(self, observed_cost, cost_limit):
        viol = observed_cost - cost_limit
        self.lambda_param += self.lr * float(viol)
        self.lambda_param = max(0.0, min(self.lambda_param, self.max_lambda))
        return self.lambda_param

    def value(self):
        return float(self.lambda_param)


# CPPO algorithm
class CPPO:
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
                 target_kl=0.03,
                 cost_limit=0.05,
                 lagrangian_lr=0.02):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.target_kl = target_kl
        self.cost_limit = cost_limit

        # networks
        self.policy = GaussianPolicy(obs_dim, act_dim).to(device)
        self.v_r = Critic(obs_dim).to(device)
        self.v_c = Critic(obs_dim).to(device)

        # optimizers
        self.pi_opt = optim.Adam(self.policy.parameters(), lr=pi_lr)
        self.vr_opt = optim.Adam(self.v_r.parameters(), lr=vf_lr)
        self.vc_opt = optim.Adam(self.v_c.parameters(), lr=vf_lr)

        # buffer
        self.buffer = RolloutBuffer(obs_dim, act_dim, steps_per_epoch, gamma=gamma, lam=lam, device=device)

        # lagrangian
        self.lagrangian = Lagrangian(init_lambda=0.1, lr=lagrangian_lr, device=device)

    def start_epoch(self):
        pass

    def select_action(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            mean, std = self.policy(obs_t)
            v_r = self.v_r(obs_t).cpu().numpy().squeeze()
            v_c = self.v_c(obs_t).cpu().numpy().squeeze()
        
        dist = torch.distributions.Normal(mean, std)
        a_t = dist.sample()
        
        logp_t = dist.log_prob(a_t).sum(axis=-1)
        action = a_t.squeeze(0).cpu().numpy()
        return action, float(logp_t.cpu().numpy()), float(v_r), float(v_c)

    def compute_values(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            v_r = self.v_r(obs_t).cpu().numpy().squeeze()
            v_c = self.v_c(obs_t).cpu().numpy().squeeze()
        return float(v_r), float(v_c)

    def store_step(self, obs, act, rew, cost, logp, done, v_r, v_c):
        self.buffer.store(obs, act, rew, cost, logp, float(done), v_r, v_c)

    def finish_epoch(self, last_val_r=0.0, last_val_c=0.0):
        # compute GAE and prepare buffer (assumes buffer.ptr may be < size; pad with zeros if needed)
        # If buffer not full, fill remaining entries with zeros to get fixed-size batches
        if self.buffer.ptr < self.buffer.size:
            # pad with the last observation repeatedly (simple strategy)
            fill = self.buffer.size - self.buffer.ptr
            for _ in range(fill):
                # append zeros (no reward) - safer to ensure pointer reaches size
                self.buffer.store(np.zeros(self.obs_dim, dtype=np.float32),
                                  np.zeros(self.act_dim, dtype=np.float32),
                                  0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        self.buffer.finish_path(last_val_r, last_val_c)

    def update(self):
        data = self.buffer.get()
        obs = data["obs"]
        acts = data["acts"]
        old_logp = data["logp"]
        adv_r = data["adv_r"]
        adv_c = data["adv_c"]
        ret_r = data["ret_r"]
        ret_c = data["ret_c"]
        costs = data["costs"]
        lam_val = self.lagrangian.value()
        
        # combined advantage
        combined_adv = adv_r - lam_val * adv_c
        combined_adv = (combined_adv - combined_adv.mean()) / (combined_adv.std() + 1e-8)
        
        pi_loss = 0.0
        for epoch in range(10):
            mean, std = self.policy(obs)
            if torch.any(torch.isnan(mean)) or torch.any(torch.isnan(std)):
                print(f"NaN detected at epoch {epoch}, stopping policy update")
                break
            
            dist = torch.distributions.Normal(mean, std)
            new_logp = dist.log_prob(acts).sum(axis=-1)
            ratio = torch.exp(new_logp - old_logp)
            
            # KL divergence to check for early stopping
            kl = (old_logp - new_logp).mean()
            if kl > 0.015:  # Early stop if policy changes too much
                print(f"KL divergence {kl:.4f} too high, stopping at epoch {epoch}")
                break
            
            unclipped = ratio * combined_adv
            clipped = torch.clamp(ratio, 1-self.clip, 1+self.clip) * combined_adv
            pi_loss = -torch.min(unclipped, clipped).mean()
            
            self.pi_opt.zero_grad()
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.pi_opt.step()
        
        loss_vr = 0.0
        for _ in range(10):
            v_r_pred = self.v_r(obs)
            loss_vr = ((v_r_pred - ret_r) ** 2).mean()
            
            self.vr_opt.zero_grad()
            loss_vr.backward()
            torch.nn.utils.clip_grad_norm_(self.v_r.parameters(), max_norm=0.5)
            self.vr_opt.step()
        
        loss_vc = 0.0
        for _ in range(10):
            v_c_pred = self.v_c(obs)
            loss_vc = ((v_c_pred - ret_c) ** 2).mean()
            
            self.vc_opt.zero_grad()
            loss_vc.backward()
            torch.nn.utils.clip_grad_norm_(self.v_c.parameters(), max_norm=0.5)
            self.vc_opt.step()
        
        # update lambda with observed cost
        avg_cost = float(costs.mean().item())
        new_lambda = self.lagrangian.update(avg_cost, self.cost_limit)
        
        return {
            "pi_loss": float(pi_loss.cpu().item()) if isinstance(pi_loss, torch.Tensor) else float(pi_loss),
            "loss_vr": float(loss_vr.cpu().item()) if isinstance(loss_vr, torch.Tensor) else float(loss_vr),
            "loss_vc": float(loss_vc.cpu().item()) if isinstance(loss_vc, torch.Tensor) else float(loss_vc),
            "lambda": float(new_lambda),
            "avg_cost": float(avg_cost)
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "v_r": self.v_r.state_dict(),
            "v_c": self.v_c.state_dict(),
            "lambda": self.lagrangian.value()
        }, path)

    def load(self, path, map_location=None):
        data = torch.load(path, map_location=map_location)
        self.policy.load_state_dict(data["policy"])
        self.v_r.load_state_dict(data["v_r"])
        self.v_c.load_state_dict(data["v_c"])
        if "lambda" in data:
            self.lagrangian.lambda_param = float(data["lambda"])
