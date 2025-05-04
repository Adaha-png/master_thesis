from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import Policy


def _freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module


def _list_policy_ids(algo: PPO) -> List[str]:
    if hasattr(algo, "get_policy_ids"):
        return list(algo.get_policy_ids())

    raw_conf = algo.config() if callable(algo.config) else algo.config
    ma_conf = raw_conf.get("multiagent") or raw_conf.get("multi_agent") or {}
    pols = ma_conf().get("policies")
    if pols is None:
        return ["default_policy"]
    if isinstance(pols, dict):
        return list(pols.keys())
    if isinstance(pols, list):
        # RLlib sometimes stores a list of (id, obs_space, act_space, config)
        return [p[0] if isinstance(p, tuple) else p for p in pols]
    raise RuntimeError("Unable to determine policy IDs for the given PPO object")


@lru_cache(maxsize=None)
def _build_policy_nets(
    algo_id: int,
    device_str: str,
    memory: str,
) -> Tuple[Dict[str, Any], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    algo: PPO = _ALGO_REGISTRY[algo_id]
    device = torch.device(device_str)

    nets: Dict[str, Any] = {}
    lstm_defaults: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    for pid in _list_policy_ids(algo):
        policy: Policy = algo.get_policy(pid)
        if policy is None:
            raise ValueError(f"Algorithm has no policy with id {pid}")

        model = policy.model

        if memory == "no_memory":
            net = _freeze(nn.Sequential(model._hidden_layers, model._logits)).to(device)
        elif memory == "lstm":
            net = (
                _freeze(model._hidden_layers).to(device),
                _freeze(model.lstm).to(device),
                _freeze(model._logits_branch).to(device),
            )
            hsize = model.lstm.hidden_size
            lstm_defaults[pid] = (
                torch.zeros(1, hsize, device=device),
                torch.zeros(1, hsize, device=device),
            )
        else:
            raise ValueError(f"Unsupported memory mode: {memory}")

        nets[pid] = net

    return nets, lstm_defaults


_ALGO_REGISTRY: Dict[int, PPO] = {}


class SimRunner:
    """Reusable rollout helper; does heavy work only once."""

    def __init__(
        self,
        algo: PPO,
        env_creator,
        *,
        memory: str = "no_memory",
        device: str | torch.device = "cpu",
    ) -> None:
        self.algo = algo
        self.env = env_creator()
        self.memory = memory
        self.device = torch.device(device)

        raw_conf = algo.config() if callable(algo.config) else algo.config
        ma_conf = raw_conf.get("multiagent") or raw_conf.get("multi_agent") or {}

        first_policy = _list_policy_ids(algo)[0]
        self.policy_mapping_fn = ma_conf().get(
            "policy_mapping_fn", lambda aid, *_, **__: first_policy
        )

        _ALGO_REGISTRY[id(algo)] = algo
        self.policy_nets, self.lstm_defaults = _build_policy_nets(
            id(algo), str(self.device), memory
        )

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def sim_steps(
        self, num_steps: int, *, seed: int | None = None
    ) -> List[Dict[str, Any]]:
        observations, _ = self.env.reset(seed=seed)
        history: List[Dict[str, Any]] = []
        done = {a: False for a in self.env.agents}
        actions = {a: 0 for a in self.env.possible_agents}
        mem: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        for _ in range(num_steps):
            if all(done.values()):
                break

            crit: Dict[str, float] = {}
            for agent, obs in observations.items():
                pid = self.policy_mapping_fn(agent, 0)
                net = self.policy_nets[pid]
                obs_t = torch.as_tensor(
                    obs, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                if self.memory == "no_memory":
                    logits = net(obs_t)
                else:
                    enc, lstm, head = net
                    h0, c0 = mem.get(agent, self.lstm_defaults[pid])
                    x = enc(obs_t)
                    x, (h1, c1) = lstm(x, (h0, c0))
                    mem[agent] = (h1, c1)
                    logits = head(x)

                actions[agent] = int(torch.argmax(logits).item())
                crit[agent] = (logits.max() - logits.min()).item()

            next_obs, rewards, done, _, _ = self.env.step(actions)
            history.append(
                {
                    a: {
                        "observation": observations[a],
                        "action": actions[a],
                        "reward": rewards.get(a),
                        "criticality": crit[a],
                        **({"memory": mem[a]} if self.memory != "no_memory" else {}),
                    }
                    for a in actions
                }
            )
            observations = next_obs

        return history

    def close(self) -> None:
        self.env.close()
