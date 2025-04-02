import os
import sys

import numpy as np
import torch

from train_tune_eval.rllib_train import env_creator

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def sim_steps(net, num_steps, memory, device, seed):
    env = env_creator()
    observations, _ = env.reset(seed)

    history = []
    if memory == "lstm":
        for agent, obs in observations.items():
            deviced = []
            for i in range(len(net)):
                deviced.append(net[i].to(device))

            net = deviced
            obs = torch.Tensor(obs).to(device)

            mem = {}
            mem["default"] = (
                torch.zeros(1, 64, device=obs.device),
                torch.zeros(1, 64, device=obs.device),
            )

    done = {agent: False for agent in env.agents}
    i = 0
    actions = {agent: 0 for agent in env.possible_agents}
    while not all(done.values()) and i < num_steps:
        criticality = {}
        for agent, obs in observations.items():
            obs = torch.Tensor(obs).unsqueeze(0).to(device)
            if memory == "no_memory":
                out = net(obs.cpu())
                actions[agent] = int(np.argmax(out.cpu()))

            elif memory == "lstm":
                out = net[0](obs)
                out, _mem = net[1](
                    out,
                    mem.get(agent, mem["default"]),
                )
                mem[agent] = _mem

                out = net[3](out)
                actions[agent] = int(np.argmax(out.cpu()))

            else:
                raise NotImplementedError

            criticality[agent] = torch.max(out).item() - torch.min(out).item()

        observations_next, rewards, done, _, _ = env.step(actions)
        step_record = {}
        for agent in actions.keys():
            step_record[agent] = {
                "observation": observations[agent],
                "action": actions[agent],
                "reward": rewards.get(agent),
                "criticality": criticality[agent],
            }
            if memory != "no_memory":
                step_record[agent]["memory"] = mem[agent]
        history.append(step_record)

        observations = observations_next
        i += 1

    env.close()

    return history
