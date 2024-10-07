import argparse
import glob
import os
from functools import partial
from typing import Dict, List

import numpy as np
import supersuit as ss
from nsga2.evolution import Evolution
from nsga2.problem import Problem
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3

from custom_env_utils import par_env_with_seed
from sim_steps import sim_steps


def action_difference(sequence1, *actions_list2):
    # Extract values associated with the key "action" from both lists
    actions_list1 = np.array([val["action"] for val in sequence1]).flatten()
    # Initialize a counter for differing values

    actions_list2 = np.floor(actions_list2)
    differences = np.sum(actions_list1 != actions_list2)
    return differences


def reward_difference(env, sequence1, *chosen_actions):
    # Extract values associated with the key "action" from both lists
    rewards_list1 = [val["reward"] for val in sequence1]
    chosen_actions = [int(a) for a in chosen_actions]

    chosen_actions = np.reshape(
        chosen_actions, shape=(len(sequence1), len(rewards_list1[0]))
    )
    chosen_actions = np.clip(chosen_actions, min=0, max=4)
    sequence2 = sim_steps(
        env,
        None,
        chosen_actions=chosen_actions,
        num_steps=len(chosen_actions),
    )

    rewards_list2 = [val["reward"] for val in sequence2]

    diff = 0

    for i in range(len(rewards_list1)):
        for j in range(len(rewards_list1[i])):
            diff += rewards_list1[i][j] - rewards_list2[i][j]
    return diff


def counterfactuals(env, sequence: List[Dict]):
    action_objective = partial(action_difference, sequence)
    reward_objective = partial(reward_difference, env, sequence)

    problem = Problem(
        [action_objective, reward_objective],
        len(sequence) * len(sequence[0]["action"]),
        [(0, 5)],
        same_range=True,
    )

    evolution = Evolution(problem)

    print(evolution.evolve())


if __name__ == "__main__":
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

    env = env_fn.parallel_env(**env_kwargs)
    env = par_env_with_seed(env, 42)

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

    seq = sim_steps(env, latest_policy, num_steps=10)
    counterfactuals(env, seq)
