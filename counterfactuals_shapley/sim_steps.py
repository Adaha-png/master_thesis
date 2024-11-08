import argparse
import glob
import inspect
import os
import pprint
import random
import sys

import numpy as np
import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_env_utils import par_env_with_seed


def sim_steps_partial(env, policy, seq, num_steps=20, seed=None):
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


def sim_steps(env, model, chosen_actions=None, num_steps=20, seed=None):
    env = par_env_with_seed(env, seed)

    if chosen_actions is None and type(model) == str:
        model = PPO.load(model)

    if seed:
        env = par_env_with_seed(env, seed)

    else:
        print(
            f"The method {inspect.stack()[0][3]}, called by {inspect.stack()[1][3]} requires a seed."
        )
        exit(0)

    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

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

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(sim_steps(env, latest_policy, num_steps=args.steps))
