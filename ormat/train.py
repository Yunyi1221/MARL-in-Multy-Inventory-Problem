"""Train the minimal MAT with PPO.

Examples:
  python -m ormat.train --order-mode learned --seed 0
  python -m ormat.train --order-mode fixed --updates 300

Results go to runs/<order_mode>_s<seed>/: progress.csv (learning curve) and model.pt.
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from ormat.envs import EnvConfig, ReplenishmentEnv
from ormat.mat import ORDER_MODES, MultiAgentTransformer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--order-mode", choices=ORDER_MODES, default="fixed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--capacity-ratio", type=float, default=EnvConfig.capacity_ratio,
                   help="supplier capacity / mean total demand; smaller is tighter")
    p.add_argument("--n-envs", type=int, default=32)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=5)
    p.add_argument("--minibatches", type=int, default=4)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--order-ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=1.0)
    p.add_argument("--max-grad-norm", type=float, default=10.0)
    p.add_argument("--n-embd", type=int, default=64)
    p.add_argument("--n-block", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--threads", type=int, default=1)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, cfg, n_episodes, seed=12345):
    """Uses the same evaluation seed as baselines.py, so results are directly comparable.
    Returns the mean episode cost and each warehouse's mean decision position."""
    env = ReplenishmentEnv(cfg, n_envs=n_episodes, seed=seed)
    obs = env.reset()
    total_cost = np.zeros(n_episodes)
    rank_sum = np.zeros(cfg.n_agents)
    done, steps = False, 0
    while not done:
        actions, _, _, perm, _ = model.act(torch.as_tensor(obs), deterministic=True)
        # perm[b, k] = agent acting k-th -> position of each agent
        ranks = torch.argsort(perm, dim=1).float().mean(0).numpy()
        rank_sum += ranks
        obs, _, done, info = env.step(actions.numpy())
        total_cost += info["cost"]
        steps += 1
    return total_cost.mean(), rank_sum / steps


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = EnvConfig(capacity_ratio=args.capacity_ratio)
    env = ReplenishmentEnv(cfg, n_envs=args.n_envs, seed=args.seed)
    N, T, E = env.n_agents, cfg.horizon, args.n_envs
    model = MultiAgentTransformer(env.obs_dim, env.n_actions, N, n_embd=args.n_embd,
                                  n_block=args.n_block, order_mode=args.order_mode)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    run_dir = os.path.join(args.out_dir, f"{args.order_mode}_s{args.seed}")
    os.makedirs(run_dir, exist_ok=True)
    log_f = open(os.path.join(run_dir, "progress.csv"), "w", newline="")
    log = csv.writer(log_f)
    log.writerow(["update", "env_steps", "train_cost", "eval_cost"] + [f"rank_{i}" for i in range(N)])

    t0 = time.time()
    for update in range(1, args.updates + 1):
        # linear learning-rate decay
        for g in opt.param_groups:
            g["lr"] = args.lr * (1 - (update - 1) / args.updates)

        # ---------------- collect one full episode ----------------
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "perm", "olp")}
        obs = env.reset()
        ep_cost = np.zeros(E)
        for _ in range(T):
            o = torch.as_tensor(obs)
            a, logp, v, perm, olp = model.act(o)
            obs, r, _, info = env.step(a.numpy())
            ep_cost += info["cost"]
            for k, x in zip(buf, (o, a, logp, v, torch.as_tensor(r, dtype=torch.float32), perm, olp)):
                buf[k].append(x)
        b = {k: torch.stack(v) for k, v in buf.items()}  # axis 0 is time

        # ---------------- GAE (one value per agent, shared team reward) ----------------
        adv = torch.zeros(T, E, N)
        last = torch.zeros(E, N)
        for t in reversed(range(T)):
            next_v = b["val"][t + 1] if t + 1 < T else torch.zeros(E, N)  # finite horizon: no bootstrap at the end
            delta = b["rew"][t][:, None] + args.gamma * next_v - b["val"][t]
            last = delta + args.gamma * args.gae_lambda * last
            adv[t] = last
        ret = adv + b["val"]

        flat = lambda x: x.reshape(T * E, *x.shape[2:])
        data = {k: flat(v) for k, v in b.items()}
        data["adv"], data["ret"] = flat(adv), flat(ret)

        # ---------------- PPO update ----------------
        batch = T * E
        mb_size = batch // args.minibatches
        for _ in range(args.ppo_epochs):
            idx = torch.randperm(batch)
            for start in range(0, batch, mb_size):
                mb = {k: v[idx[start:start + mb_size]] for k, v in data.items()}
                logp, ent, val, olp, oent = model.evaluate(mb["obs"], mb["act"], mb["perm"])

                a_n = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
                ratio = torch.exp(logp - mb["logp"])
                pg_loss = -torch.min(ratio * a_n,
                                     ratio.clamp(1 - args.clip, 1 + args.clip) * a_n).mean()
                v_loss = F.huber_loss(val, mb["ret"], delta=10.0)
                loss = pg_loss + args.vf_coef * v_loss - args.ent_coef * ent.mean()

                if args.order_mode == "learned":
                    # the order is part of the action: add a PPO term driven by the team advantage (mean over agents)
                    a_o = a_n.mean(dim=1)
                    o_ratio = torch.exp(olp - mb["olp"])
                    o_loss = -torch.min(o_ratio * a_o,
                                        o_ratio.clamp(1 - args.clip, 1 + args.clip) * a_o).mean()
                    loss = loss + o_loss - args.order_ent_coef * oent.mean()

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                opt.step()

        # ---------------- logging ----------------
        if update % args.eval_every == 0 or update == args.updates:
            eval_cost, ranks = evaluate(model, cfg, args.eval_episodes)
            log.writerow([update, update * T * E, f"{ep_cost.mean():.2f}", f"{eval_cost:.2f}"]
                         + [f"{x:.3f}" for x in ranks])
            log_f.flush()
            print(f"[{args.order_mode} s{args.seed}] update {update:4d} | "
                  f"train cost {ep_cost.mean():7.1f} | eval cost {eval_cost:7.1f} | "
                  f"positions {np.round(ranks, 2).tolist()} | {time.time() - t0:5.0f}s", flush=True)

    torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))
    log_f.close()


if __name__ == "__main__":
    main()
