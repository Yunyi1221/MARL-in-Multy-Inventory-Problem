"""Minimal Multi-Agent Transformer (discrete actions) with four action-generation orders.

order_mode:
  fixed   agents act in index order 0..N-1 (original MAT)
  random  a fresh random order at every step (the random order used by HAPPO / MAT)
  urgency the order is supplied by the caller (OR urgency ranking, most urgent first); it is
          part of the state, not of the action, so it carries no log-probability
  learned a scoring network + Plackett-Luce sampling learns the order (the core idea of PMAT)
  none    no autoregression; each agent only sees the encoder output (parameter-shared,
          MAPPO-style simultaneous decisions)

Convention: perm[b, k] is the index of the agent that acts k-th. Every tensor in the
public interface is indexed by agent; only the decoder works in perm order.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ORDER_MODES = ("fixed", "random", "urgency", "learned", "none")


def init_(m, gain=0.01, relu=False):
    if relu:
        gain = nn.init.calculate_gain("relu")
    nn.init.orthogonal_(m.weight, gain=gain)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, causal):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.causal = n_head, causal
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        self.proj = init_(nn.Linear(n_embd, n_embd))

    def forward(self, key, value, query):
        B, L, D = query.shape
        h = self.n_head
        k = self.key(key).view(B, L, h, D // h).transpose(1, 2)
        q = self.query(query).view(B, L, h, D // h).transpose(1, 2)
        v = self.value(value).view(B, L, h, D // h).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        if self.causal:
            mask = torch.ones(L, L, dtype=torch.bool, device=att.device).tril()
            att = att.masked_fill(~mask, float("-inf"))
        y = F.softmax(att, dim=-1) @ v
        return self.proj(y.transpose(1, 2).reshape(B, L, D))


class EncodeBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, causal=False)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, n_embd), relu=True), nn.GELU(),
                                 init_(nn.Linear(n_embd, n_embd)))

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        return self.ln2(x + self.mlp(x))


class DecodeBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1, self.ln2, self.ln3 = (nn.LayerNorm(n_embd) for _ in range(3))
        self.attn1 = SelfAttention(n_embd, n_head, causal=True)
        self.attn2 = SelfAttention(n_embd, n_head, causal=True)
        self.mlp = nn.Sequential(init_(nn.Linear(n_embd, n_embd), relu=True), nn.GELU(),
                                 init_(nn.Linear(n_embd, n_embd)))

    def forward(self, x, rep):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep + self.attn2(key=x, value=x, query=rep))
        return self.ln3(x + self.mlp(x))


def gather_agents(x, perm):
    """x: (B, N, ...) reordered into decision order given by perm."""
    idx = perm.view(*perm.shape, *([1] * (x.dim() - 2))).expand(*perm.shape, *x.shape[2:])
    return torch.gather(x, 1, idx)


def scatter_agents(x, perm):
    """Inverse of gather_agents: decision order -> agent index order."""
    out = torch.empty_like(x)
    idx = perm.view(*perm.shape, *([1] * (x.dim() - 2))).expand(*perm.shape, *x.shape[2:])
    return out.scatter_(1, idx, x)


def plackett_luce_log_prob(scores, perm):
    """scores: (B, N), perm: (B, N). Returns the log-probability of the permutation, (B,)."""
    s = torch.gather(scores, 1, perm)
    # the normalizer at position k is the logsumexp over the remaining items k..N-1
    tail_lse = torch.logcumsumexp(s.flip(1), dim=1).flip(1)
    return (s - tail_lse).sum(dim=1)


def plackett_luce_entropy_proxy(scores):
    """Entropy of the first-pick distribution, used as an exploration bonus for the order
    (the exact entropy would need all N! permutations)."""
    return Categorical(logits=scores).entropy()


class MultiAgentTransformer(nn.Module):
    def __init__(self, obs_dim, n_actions, n_agents, n_embd=64, n_head=2, n_block=1,
                 order_mode="fixed"):
        super().__init__()
        if order_mode not in ORDER_MODES:
            raise ValueError(f"order_mode must be one of {ORDER_MODES}")
        self.n_agents, self.n_actions, self.order_mode = n_agents, n_actions, order_mode

        # encoder: observations -> representations + a value per agent
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), relu=True), nn.GELU())
        self.enc_ln = nn.LayerNorm(n_embd)
        self.enc_blocks = nn.ModuleList(EncodeBlock(n_embd, n_head) for _ in range(n_block))
        self.value_head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), relu=True), nn.GELU(),
                                        nn.LayerNorm(n_embd), init_(nn.Linear(n_embd, 1)))

        # decoder: previous agent's action (one-hot, one extra slot for the start token) -> current agent's action distribution
        self.action_encoder = nn.Sequential(
            init_(nn.Linear(n_actions + 1, n_embd, bias=False), relu=True), nn.GELU())
        self.dec_ln = nn.LayerNorm(n_embd)
        self.dec_blocks = nn.ModuleList(DecodeBlock(n_embd, n_head) for _ in range(n_block))
        self.policy_head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), relu=True), nn.GELU(),
                                         nn.LayerNorm(n_embd), init_(nn.Linear(n_embd, n_actions)))

        # order scoring network (only used in learned mode)
        self.order_scorer = nn.Sequential(init_(nn.Linear(n_embd, n_embd), relu=True), nn.GELU(),
                                          init_(nn.Linear(n_embd, 1)))

    # ------------------------------------------------------------------ #
    def encode(self, obs):
        x = self.enc_ln(self.obs_encoder(obs))
        for blk in self.enc_blocks:
            x = blk(x)
        return x, self.value_head(x).squeeze(-1)

    def order_scores(self, rep):
        return self.order_scorer(rep).squeeze(-1)

    def sample_order(self, rep, deterministic=False):
        """Returns perm (B, N) and its log-probability (B,); the log-probability is 0 outside learned mode."""
        B, N = rep.shape[:2]
        dev = rep.device
        if self.order_mode == "learned":
            scores = self.order_scores(rep)
            if deterministic:
                perm = scores.argsort(dim=1, descending=True)
            else:  # the Gumbel-top-k trick is equivalent to Plackett-Luce sampling
                g = -torch.log(-torch.log(torch.rand_like(scores).clamp_min(1e-10)))
                perm = (scores + g).argsort(dim=1, descending=True)
            return perm, plackett_luce_log_prob(scores, perm)
        if self.order_mode == "random":  # random order stays random at evaluation; deterministic only affects actions
            perm = torch.rand(B, N, device=dev).argsort(dim=1)
        else:
            perm = torch.arange(N, device=dev).expand(B, N)
        return perm, torch.zeros(B, device=dev)

    def _decode(self, shifted_onehot, rep_ordered):
        x = self.dec_ln(self.action_encoder(shifted_onehot))
        for blk in self.dec_blocks:
            x = blk(x, rep_ordered)
        return self.policy_head(x)

    def _start_token(self, B, N, dev):
        tok = torch.zeros(B, N, self.n_actions + 1, device=dev)
        tok[..., -1] = 1.0
        return tok

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def act(self, obs, deterministic=False, perm=None):
        """obs: (B, N, obs_dim). Returns actions, per-agent logp, values, perm, order_logp.
        perm is required in urgency mode and ignored otherwise."""
        B, N = obs.shape[:2]
        dev = obs.device
        rep, values = self.encode(obs)
        if self.order_mode == "urgency":
            if perm is None:
                raise ValueError("urgency mode needs perm from the caller")
            order_logp = torch.zeros(B, device=dev)
        else:
            perm, order_logp = self.sample_order(rep, deterministic)
        rep_o = gather_agents(rep, perm)

        dec_in = self._start_token(B, N, dev)
        actions_o = torch.zeros(B, N, dtype=torch.long, device=dev)
        logp_o = torch.zeros(B, N, device=dev)

        if self.order_mode == "none":
            dist = Categorical(logits=self._decode(dec_in, rep_o))
            actions_o = dist.probs.argmax(-1) if deterministic else dist.sample()
            logp_o = dist.log_prob(actions_o)
        else:
            for k in range(N):  # autoregressive: agent k sees the actions of the k-1 agents before it
                logits = self._decode(dec_in, rep_o)[:, k]
                dist = Categorical(logits=logits)
                a = logits.argmax(-1) if deterministic else dist.sample()
                actions_o[:, k], logp_o[:, k] = a, dist.log_prob(a)
                if k + 1 < N:
                    dec_in[:, k + 1] = F.one_hot(a, self.n_actions + 1).float()

        actions = scatter_agents(actions_o, perm)
        logp = scatter_agents(logp_o, perm)
        return actions, logp, values, perm, order_logp

    def evaluate(self, obs, actions, perm):
        """Training-time pass: teacher forcing gives every agent's logp and entropy in one go."""
        B, N = obs.shape[:2]
        rep, values = self.encode(obs)
        rep_o = gather_agents(rep, perm)
        dec_in = self._start_token(B, N, obs.device)
        if self.order_mode != "none":
            actions_o = gather_agents(actions, perm)
            dec_in[:, 1:] = F.one_hot(actions_o[:, :-1], self.n_actions + 1).float()
        dist = Categorical(logits=self._decode(dec_in, rep_o))
        logp = scatter_agents(dist.log_prob(gather_agents(actions, perm)), perm)
        entropy = scatter_agents(dist.entropy(), perm)

        if self.order_mode == "learned":
            scores = self.order_scores(rep)
            order_logp = plackett_luce_log_prob(scores, perm)
            order_entropy = plackett_luce_entropy_proxy(scores)
        else:
            order_logp = order_entropy = torch.zeros(B, device=obs.device)
        return logp, entropy, values, order_logp, order_entropy
