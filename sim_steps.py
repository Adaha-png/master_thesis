import argparse
import glob
import os
import pprint
import random
from types import MethodType

import numpy as np
import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO

from custom_env_utils import par_env_with_seed


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


def sim_steps(
    env,
    policy,
    chosen_actions=None,
    num_steps=20,
):
    if chosen_actions is None:
        model = PPO.load(policy)

    obs = env.reset()
    rollout = []
    for step in range(num_steps):
        step_dict = {
            "observation": obs,
        }

        if chosen_actions is None:
            act = model.predict(obs, deterministic=True)[0]
        else:
            act = chosen_actions[step]

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

    env.close()
    return rollout


if __name__ == "__main__":
    seed = 42
    # Superseeding, might be unnecessary
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(description="Simulation")
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        help="Which environment to use",
        default="spread",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        help="Steps to simulate",
        default=10,
    )
    parser.add_argument(
        "-r",
        "--render",
        type=str,
        help="Render mode, default None",
        default=None,
    )

    args = parser.parse_args()
    if args.env == "spread":
        env_fn = simple_spread_v3

        env_kwargs = dict(
            N=3,
            local_ratio=0.5,
            max_cycles=25,
            continuous_actions=False,
        )
    elif args.env == "kaz":
        env_fn = knights_archers_zombies_v10

        env_kwargs = dict(
            spawn_rate=6,
            num_archers=2,
            num_knights=2,
            max_zombies=10,
            max_arrows=10,
            max_cycles=900,
            vector_state=True,
        )
    else:
        print("Invalid env entered")
        exit(0)

    env = env_fn.parallel_env(render_mode=args.render, **env_kwargs)

    env = par_env_with_seed(env, seed)

    try:
        latest_policy = max(
            glob.glob(f"{str(env.metadata['name'])}/*.zip"),
            key=os.path.getctime,
        )
        print(latest_policy)
    except ValueError:
        print("Policy not found.")
        exit(0)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(sim_steps(env, latest_policy, num_steps=args.steps))
