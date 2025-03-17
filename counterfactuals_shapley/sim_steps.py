import argparse
import glob
import inspect
import os
import pprint
import random
import sys
import warnings

import numpy as np
import supersuit as ss
import torch
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3

from train_tune_eval.rllib_train import env_creator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from .wrappers import par_env_with_seed


def sim_steps_partial(env, policy, seq, num_steps=20, seed=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = PPO.load(policy)
    if seed:
        env = par_env_with_seed(env, seed)
    else:
        print(
            f"The method {inspect.stack()[0][3]}, called by {inspect.stack()[1][3]} requires a seed"
        )
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    obs = env.reset()
    rollout = []
    for step in seq:
        step_dict = {
            "observation": obs,
        }

        act = step["action"]

        obs, reward, termination, info = env.step(act)
        step_dict.update(
            {
                "reward": reward,
                "action": act,
            }
        )
        rollout.append(step_dict)
        if termination.all():
            break

    for _ in range(len(seq), num_steps):
        step_dict = {
            "observation": obs,
        }
        act = model.predict(obs, deterministic=True)[0]
        obs, reward, termination, info = env.step(act)
        step_dict.update(
            {
                "reward": reward,
                "action": act,
            }
        )

        rollout.append(step_dict)

    env.close()
    return rollout


def sim_steps(net, num_steps, memory, seed):
    env = env_creator()
    observations, _ = env.reset(seed)

    history = []
    if not memory == "no_memory":
        obs = iter(observations.values())
        vals = net[1](net[0](obs))
        if len(vals.shape) == 2:
            vals = vals.unsqueeze(1)  # becomes [B, 1, 64]

        B = vals.size(0)  # current batch size

        mem = {}
        mem["def"] = (
            torch.zeros(1, B, 64, device=vals.device),
            torch.zeros(1, B, 64, device=vals.device),
        )

    done = {agent: False for agent in env.agents}
    i = 0
    while not all(done.values()) and i < num_steps:
        actions = {}
        for agent, obs in observations.items():
            if memory == "no_memory":
                vals = net.forward(torch.Tensor(obs))
                actions[agent] = int(np.argmax(vals))
            else:
                vals = torch.tanh(net[1](torch.tanh(net[0](obs))))
                vals, _mem = net[2](vals, mem.get(agent, mem["def"]))
                vals = net[3](vals)
                mem[agent] = _mem

                actions[agent] = int(np.argmax(vals))

        observations_next, rewards, done, _, _ = env.step(actions)

        step_record = {}
        for agent in actions.keys():
            step_record[agent] = {
                "observation": observations[agent],
                "action": actions[agent],
                "reward": rewards.get(agent),
            }
            if memory != "no_memory":
                step_record[agent]["memory"] = mem[agent]

        history.append(step_record)

        observations = observations_next
        i += 1

    env.close()

    return history
