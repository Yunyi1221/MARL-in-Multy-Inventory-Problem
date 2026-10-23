"""IPPO, MAPPO and HAPPO with MLP actors, on the same environment and budget as train.py.

  ippo   shared actor + shared critic, both on the local observation (plus agent ID)
  mappo  shared actor on the local observation + centralized critic on all observations
  happo  one actor per agent + centralized critic; actors are updated one by one in a random
         order, each weighting its advantage by the ratio product of the agents updated before it

Examples:
  python -m ormat.ppo_marl --algo mappo --seed 0 --capacity-ratio 0.85

Results go to runs/<algo>_s<seed>/progress.csv, in the same format as train.py.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ormat.envs import EnvConfig
from ormat.mat import init_
from ormat.residual import bias_towards_zero, make_env

ALGOS = ("ippo", "mappo", "happo")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=ALGOS, default="mappo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--capacity-ratio", type=float, default=EnvConfig.capacity_ratio)
    p.add_argument("--fixed-cost", type=float, default=EnvConfig.fixed_cost,
                   help="fixed cost per warehouse order")
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--residual", action="store_true",
                   help="learn corrections to the priority-rationing heuristic instead of raw orders")
    p.add_argument("--residual-base", choices=("priority", "marginal"), default="priority",
                   help="heuristic the residual corrections are added to")
    p.add_argument("--max-delta", type=int, default=3, help="residual corrections lie in [-max_delta, max_delta]")
    p.add_argument("--residual-prior", type=float, default=2.0,
                   help="initial logit bonus for a zero correction, so training starts at the heuristic")
    return p.parse_args()


def mlp(i, h, o, out_gain=0.01):
    return nn.Sequential(nn.LayerNorm(i), init_(nn.Linear(i, h), relu=True), nn.ReLU(),
                         init_(nn.Linear(h, h), relu=True), nn.ReLU(), init_(nn.Linear(h, o), gain=out_gain))


class Agents(nn.Module):
    """Actors and critic for all three algorithms. Every input is (B, N, obs_dim)."""

    def __init__(self, algo, obs_dim, n_actions, n_agents, hidden):
        super().__init__()
        self.algo, self.N = algo, n_agents
        in_dim = obs_dim + n_agents  # local observation + one-hot agent ID
        if algo == "happo":
            self.actors = nn.ModuleList(mlp(obs_dim, hidden, n_actions) for _ in range(n_agents))
        else:
            self.actor = mlp(in_dim, hidden, n_actions)
        if algo == "ippo":
            self.critic = mlp(in_dim, hidden, 1, out_gain=1.0)
        else:
            self.critic = mlp(obs_dim * n_agents, hidden, 1, out_gain=1.0)

    def _with_id(self, obs):
        B = obs.shape[0]
        return torch.cat([obs, torch.eye(self.N).expand(B, self.N, self.N)], dim=-1)

    def logits(self, obs, agent=None):
        """All agents' logits (B, N, A), or one agent's (B, A) when agent is given (HAPPO)."""
        if self.algo == "happo":
            if agent is not None:
                return self.actors[agent](obs[:, agent])
            return torch.stack([a(obs[:, i]) for i, a in enumerate(self.actors)], dim=1)
        return self.actor(self._with_id(obs))

    def values(self, obs):
        """(B, N): one value per agent; the centralized critic repeats its scalar for every agent."""
        if self.algo == "ippo":
            return self.critic(self._with_id(obs)).squeeze(-1)
        return self.critic(obs.flatten(1)).expand(-1, self.N)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        dist = Categorical(logits=self.logits(obs))
        a = dist.probs.argmax(-1) if deterministic else dist.sample()
        return a, dist.log_prob(a), self.values(obs)


@torch.no_grad()
def evaluate(model, cfg, n_episodes, args, seed=12345):
    """Same evaluation seed and episode count as train.py and baselines.py."""
    env = make_env(cfg, n_episodes, seed, args.residual, args.max_delta, args.residual_base)
    obs = env.reset()
    total_cost = np.zeros(n_episodes)
    done = False
    while not done:
        a, _, _ = model.act(torch.as_tensor(obs), deterministic=True)
        obs, _, done, info = env.step(a.numpy())
        total_cost += info["cost"]
    return total_cost.mean()


def clipped(ratio, adv, clip):
    return torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv)


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = EnvConfig(capacity_ratio=args.capacity_ratio, fixed_cost=args.fixed_cost)
    env = make_env(cfg, args.n_envs, args.seed, args.residual, args.max_delta, args.residual_base)
    N, T, E = env.n_agents, cfg.horizon, args.n_envs
    model = Agents(args.algo, env.obs_dim, env.n_actions, N, args.hidden)
    if args.residual:
        for actor in (model.actors if args.algo == "happo" else [model.actor]):
            bias_towards_zero(actor[-1], args.max_delta, args.residual_prior)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    prefix = ("res-" if args.residual_base == "priority" else "resm-") if args.residual else ""
    run_dir = os.path.join(args.out_dir, f"{prefix}{args.algo}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    log_f = open(os.path.join(run_dir, "progress.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["update", "env_steps", "train_cost", "eval_cost"])

    t0 = time.time()
    for update in range(1, args.updates + 1):
        for g in opt.param_groups:
            g["lr"] = args.lr * (1 - (update - 1) / args.updates)

        # ---------------- collect one full episode ----------------
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew")}
        obs = env.reset()
        ep_cost = np.zeros(E)
        for _ in range(T):
            o = torch.as_tensor(obs)
            a, logp, v = model.act(o)
            obs, r, _, info = env.step(a.numpy())
            ep_cost += info["cost"]
            for k, x in zip(buf, (o, a, logp, v, torch.as_tensor(r, dtype=torch.float32))):
                buf[k].append(x)
        b = {k: torch.stack(v) for k, v in buf.items()}

        # ---------------- GAE (shared team reward) ----------------
        adv = torch.zeros(T, E, N)
        last = torch.zeros(E, N)
        for t in reversed(range(T)):
            next_v = b["val"][t + 1] if t + 1 < T else torch.zeros(E, N)
            delta = b["rew"][t][:, None] + args.gamma * next_v - b["val"][t]
            last = delta + args.gamma * args.gae_lambda * last
            adv[t] = last
        ret = adv + b["val"]

        flat = lambda x: x.reshape(T * E, *x.shape[2:])
        data = {k: flat(v) for k, v in b.items()}
        data["adv"], data["ret"] = flat(adv), flat(ret)
        batch = T * E
        mb_size = batch // args.minibatches

        def ppo_epochs(loss_fn):
            for _ in range(args.ppo_epochs):
                idx = torch.randperm(batch)
                for start in range(0, batch, mb_size):
                    mb = {k: v[idx[start:start + mb_size]] for k, v in data.items()}
                    opt.zero_grad()
                    loss_fn(mb).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    opt.step()

        def v_loss(mb):
            return F.huber_loss(model.values(mb["obs"]), mb["ret"], delta=10.0)

        if args.algo in ("ippo", "mappo"):
            def loss_fn(mb):
                dist = Categorical(logits=model.logits(mb["obs"]))
                a_n = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
                ratio = torch.exp(dist.log_prob(mb["act"]) - mb["logp"])
                pg = -clipped(ratio, a_n, args.clip).mean()
                return pg + v_loss(mb) - args.ent_coef * dist.entropy().mean()
            ppo_epochs(loss_fn)
        else:
            ppo_epochs(v_loss)  # critic first, then the actors one at a time
            team_adv = data["adv"][:, 0]  # the centralized critic gives every agent the same advantage
            team_adv = (team_adv - team_adv.mean()) / (team_adv.std() + 1e-8)
            data["m"] = torch.ones(batch)  # product of new/old ratios of the agents already updated
            for i in torch.randperm(N).tolist():
                def loss_fn(mb, i=i):
                    dist = Categorical(logits=model.logits(mb["obs"], agent=i))
                    ratio = torch.exp(dist.log_prob(mb["act"][:, i]) - mb["logp"][:, i])
                    return (-clipped(ratio, mb["m"] * mb["team_adv"], args.clip).mean()
                            - args.ent_coef * dist.entropy().mean())
                data["team_adv"] = team_adv
                ppo_epochs(loss_fn)
                with torch.no_grad():
                    new_logp = Categorical(logits=model.logits(data["obs"], agent=i)).log_prob(data["act"][:, i])
                    data["m"] = data["m"] * torch.exp(new_logp - data["logp"][:, i])

        if update % args.eval_every == 0 or update == args.updates:
            eval_cost = evaluate(model, cfg, args.eval_episodes, args)
            log.writerow([update, update * T * E, f"{ep_cost.mean():.2f}", f"{eval_cost:.2f}"])
            log_f.flush()
            print(f"[{args.algo} s{args.seed}] update {update:4d} | train cost {ep_cost.mean():7.1f} | "
                  f"eval cost {eval_cost:7.1f} | {time.time() - t0:5.0f}s", flush=True)

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    log_f.close()


if __name__ == "__main__":
    main()
