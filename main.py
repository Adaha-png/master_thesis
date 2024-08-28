import copy

import torch
from pettingzoo.butterfly import knights_archers_zombies_v10
from skrl.memories.torch.base import Memory
from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.trainers.torch.parallel import (
    PARALLEL_TRAINER_DEFAULT_CONFIG,
    ParallelTrainer,
)

from critic import c_CNN, c_MLP
from policy import p_CNN, p_MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = knights_archers_zombies_v10.parallel_env(
    spawn_rate=20,
    num_archers=2,
    num_knights=2,
    max_zombies=10,
    max_arrows=10,
    max_cycles=900,
    vector_state=False,
    render_mode="human",
)
observations, infos = env.reset()

obs_spaces = {agent: env.observation_space(agent) for agent in env.possible_agents}
act_spaces = {agent: env.action_space(agent) for agent in env.possible_agents}
memories = {agent: Memory(20, device=device) for agent in env.possible_agents}


# using the same policy for every agent

models = {}
for agent in env.agents:
    models[agent] = {}
    models[agent]["policy"] = p_CNN(obs_spaces[agent], act_spaces[agent], device)
    models[agent]["value"] = c_CNN(
        obs_spaces[agent], act_spaces[agent], device, clip_actions=False
    )

cfg_agent = IPPO_DEFAULT_CONFIG.copy()

agents = IPPO(
    possible_agents=env.possible_agents,
    models=models,
    memories=memories,
    cfg=cfg_agent,
    observation_spaces=obs_spaces,
    action_spaces=act_spaces,
    device=device,
)

train_conf = copy.copy(PARALLEL_TRAINER_DEFAULT_CONFIG)
trainer = ParallelTrainer(
    env,
    agents,
    cfg=train_conf,
)

trainer.train()

env.close()
