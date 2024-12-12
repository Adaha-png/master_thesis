from types import MethodType

import numpy as np
import torch


def par_env_with_seed(env, seed):
    # pettingzoo/gymnasium/supersuit compatibility, use this function if you want seeded env
    def custom_reset(self, seed=None, options=None):
        if self._seed is not None:
            seed = self._seed

        self.aec_env.reset(seed=seed, options=options)
        self.agents = self.aec_env.agents[:]
        observations = {
            agent: self.aec_env.observe(agent)
            for agent in self.aec_env.agents
            if not (self.aec_env.terminations[agent] or self.aec_env.truncations[agent])
        }

        infos = dict(**self.aec_env.infos)
        return observations, infos

    def set_seed(self, seed):
        self._seed = seed

    env.seed = MethodType(set_seed, env)
    env.reset = MethodType(custom_reset, env)
    env.seed(seed)

    return env


def numpyfy(X):
    if isinstance(X, torch.Tensor):
        return np.array(X.cpu())
    else:
        return np.array(X)
