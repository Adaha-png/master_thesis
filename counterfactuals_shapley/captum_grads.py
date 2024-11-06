import argparse
import glob
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from counterfactuals import action_difference_with_model, counterfactuals_with_model
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from sim_steps import sim_steps
from stable_baselines3 import PPO
from torch import nn


def ig_extract(policy, obs, action, feature_names, act_dict):
    model = PPO.load(policy)
    net = nn.Sequential(
        *model.policy.mlp_extractor.policy_net,
        model.policy.action_net,
        nn.Softmax(),
    )

    ig = IntegratedGradients(net)

    attributions, approximation_error = ig.attribute(
        obs,
        baselines=torch.zeros(obs.shape),
        target=action,
        method="gausslegendre",
        return_convergence_delta=True,
    )

    print(f"{attributions=}")
    print(f"{approximation_error=}")
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.squeeze().detach().numpy()
    sorted_indices = np.argsort(np.abs(attributions))
    attributions = attributions[sorted_indices]
    feature_names = np.array(feature_names)[sorted_indices]

    print(f"Action: {act_dict[action]}, Confidence:{net.forward(obs)[0,action]}")

    # Create a new figure and axis
    _, ax = plt.subplots(figsize=(8, 12))

    # Stem plot
    ax.scatter(attributions, feature_names, s=6)

    # Add horizontal lines for each feature
    for n in feature_names:
        ax.axhline(n, color="gray", linestyle="--", linewidth=0.5)

    # Add vertical line at x=0
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)
    # Set labels and title
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Integrated gradients method")

    # Show the plot
    plt.savefig(f"tex/images/intgrad_{act_dict[action]}.pdf".replace(" ", "_"))


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

        feature_names = [
            "vel x",
            "vel y",
            "pos x",
            "pos y",
            "landmark 1 x",
            "landmark 1 y",
            "landmark 2 x",
            "landmark 2 y",
            "landmark 3 x",
            "landmark 3 y",
            "agent 2 x",
            "agent 2 y",
            "agent 3 x",
            "agent 3 y",
            "comms",
            "comms",
            "comms",
            "comms",
        ]
        act_dict = {
            0: "no action",
            1: "move left",
            2: "move right",
            3: "move down",
            4: "move up",
        }
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
        feature_names = None
        act_dict = None
    else:
        print("Invalid env entered")
        exit(0)

    env = env_fn.parallel_env(render_mode=args.render, **env_kwargs)
    try:
        latest_policy = max(
            glob.glob(f".{str(env.metadata['name'])}/*.zip"),
            key=os.path.getctime,
        )
        print(latest_policy)
    except ValueError:
        print("Policy not found in " + f".{str(env.metadata['name'])}/*.zip")
        exit(0)

    if not os.path.exists("obs_act_ag.pkl"):
        seq = sim_steps(env, latest_policy, num_steps=20, seed=seed)
        individuals = counterfactuals_with_model(env, seq, latest_policy, seed)

        individual = [
            i.features
            for i in individuals
            if action_difference_with_model(*i.features) == 1
        ][0]

        index = int(individual[0])
        relevant_obs = seq[index]["observation"]

        agent = np.argmax(individual[1:])
        action = seq[index]["action"][agent]
        relevant_obs = torch.from_numpy(relevant_obs[agent]).unsqueeze(0)
        print(f"{action := int(action)=}")

        # Assuming the action and relevant_obs variables are already defined
        data_to_save = {
            "action": action,
            "relevant_obs": relevant_obs,
            "agent": agent,
        }

        # Save to a file
        with open("obs_act_ag.pkl", "wb") as f:
            pickle.dump(data_to_save, f)
    else:
        # Load from the file
        with open("obs_act_ag.pkl", "rb") as f:
            loaded_data = pickle.load(f)

        # Extracting the saved data
        action = loaded_data["action"]
        relevant_obs = loaded_data["relevant_obs"]
        agent = loaded_data["agent"]
    print(f"{action=}")
    ig_extract(latest_policy, relevant_obs, action, feature_names, act_dict)
