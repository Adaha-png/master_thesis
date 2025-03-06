import argparse
import glob
import os
from copy import deepcopy
from functools import partial
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import supersuit as ss
from nsga2.evolution import Evolution
from nsga2.problem import Problem
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from sklearn.model_selection import train_test_split

from .sim_steps import sim_steps, sim_steps_partial
from .wrappers import numpyfy, par_env_with_seed


def action_difference(sequence1, *actions_list2):
    # Extract values associated with the key "action" from both lists
    actions_list1 = numpyfy([val["action"] for val in sequence1]).flatten()
    # Initialize a counter for differing values

    actions_list2 = np.floor(actions_list2)
    actions_list2 = np.clip(actions_list2, a_min=0, a_max=4)
    differences = -np.sum(actions_list1 != actions_list2)
    return differences


def reward_difference(env, sequence1, *chosen_actions):
    # Extract values associated with the key "action" from both lists
    rewards_list1 = [val["reward"] for val in sequence1]
    chosen_actions = [int(a) for a in chosen_actions]

    chosen_actions = np.reshape(
        chosen_actions, newshape=(len(sequence1), len(rewards_list1[0]))
    )
    chosen_actions = np.clip(chosen_actions, a_min=0, a_max=4)
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
    return -diff


def counterfactuals(env, sequence: List[Dict]):
    action_objective = partial(action_difference, sequence)
    reward_objective = partial(reward_difference, env, sequence)

    problem = Problem(
        [action_objective, reward_objective],
        len(sequence) * len(sequence[0]["action"]),
        [(0, env.action_space.n)],
        same_range=True,
    )

    evolution = Evolution(
        problem,
        num_of_generations=2,
        num_of_individuals=400,
        num_of_tour_particips=4,
        tournament_prob=0.85,
    )

    individuals = numpyfy(evolution.evolve())
    ind_plotting = numpyfy(
        [
            [-action_objective(*i.features), reward_objective(*i.features)]
            for i in individuals
        ]
    )
    plt.rcParams.update(
        {
            "font.family": "serif",
            # Use LaTeX default serif font.
            "font.serif": [],
            "pgf.texsystem": "pdflatex",
        }
    )
    # Plotting the points
    plt.scatter(
        ind_plotting[:, 0],
        ind_plotting[:, 1],
        color="red",
        label="Pareto Optimal Points",
    )
    plt.xlabel("Action change")
    plt.ylabel("Reward change")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.title("Pareto Optimal Set")
    plt.savefig("tex/images/best_counterfactuals.pgf", backend="pgf")


def reward_difference_with_model(
    env, policy, sequence1, n_actions, seed, *chosen_actions
):
    if sum(numpyfy(chosen_actions) >= 0) - 1 == 0:
        return 0

    index = min(int(list(chosen_actions).pop(0)), len(sequence1) - 1)
    # Extract values associated with the key "action" from both lists
    rewards_list1 = [val["reward"] for val in sequence1]

    sequence2 = deepcopy(sequence1[0 : index + 1])

    sequence2[index]["action"] = [
        int(chosen_actions[i])
        if chosen_actions[i] >= 0
        else sequence1[index]["action"][i - 1]
        for i in range(1, len(chosen_actions))
    ]
    sequence2[index]["action"] = np.clip(
        sequence2[index]["action"],
        a_min=0,
        a_max=n_actions - 1,
    )

    sequence2 = sim_steps_partial(
        env,
        policy,
        sequence2,
        seed=seed,
        num_steps=len(sequence1),
    )

    rewards_list2 = [val["reward"] for val in sequence2]

    diff = 0
    for i in range(len(rewards_list1)):
        for j in range(len(rewards_list1[i])):
            diff += rewards_list1[i][j] - rewards_list2[i][j]
    return -diff


def action_difference_with_model(*actions):
    return sum(numpyfy(actions) >= 0) - 1


def counterfactuals_with_model(env, sequence, policy, seed):
    reward_objective = partial(reward_difference_with_model, env, policy, sequence)
    action_objective = partial(action_difference_with_model)

    env = par_env_with_seed(env, seed)
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    acts = env.action_space.n

    reward_objective = partial(reward_objective, acts, seed)

    problem = Problem(
        [action_objective, reward_objective],
        1 + len(sequence[0]["action"]),
        [(0, len(sequence))] + [(-1, acts) for _ in range(len(sequence[0]["action"]))],
    )

    evolution = Evolution(
        problem,
        num_of_generations=10,
        num_of_individuals=10,
        num_of_tour_particips=2,
        tournament_prob=0.9,
    )

    individuals = numpyfy(evolution.evolve())
    # ind_plotting = numpyfy(
    #     [
    #         [action_objective(*i.features), -reward_objective(*i.features)]
    #         if reward_objective(*i.features) != 0
    #     ]
    # )
    #
    # # Plotting the points
    # plt.scatter(
    #         for i in individuals
    #     ind_plotting[:, 0],
    #     ind_plotting[:, 1],
    #     color="red",
    #     label="Pareto Optimal Points",
    # )
    # plt.xlabel("Action change")
    # plt.ylabel("Reward change")
    # plt.gca().invert_xaxis()
    # plt.legend()
    # plt.title("Pareto Optimal Set")
    # plt.savefig("tex/images/best_counterfactuals_with_model.pdf")
    #
    # return [i.features for i in individuals]
    return individuals


if __name__ == "__main__":
    seed = 10
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
            local_ratio=0.0,
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

    try:
        latest_policy = max(
            glob.glob(f"{str(env.metadata['name'])}/*.zip"),
            key=os.path.getctime,
        )
        print(latest_policy)
    except ValueError:
        print("Policy not found.")
        exit(0)

    seq = sim_steps(env, latest_policy, num_steps=20, seed=seed)
    ind = counterfactuals_with_model(env, seq, latest_policy, seed)
    relevant_obs = seq[int(ind[0])]["observation"]
    print(relevant_obs)
