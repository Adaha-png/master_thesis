import argparse
import glob
import os
import pickle

import numpy as np
import torch
from counterfactuals import action_difference_with_model, counterfactuals_with_model
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from shapley import kernel_explainer, shap_plot
from sim_steps import sim_steps

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
    else:
        print("Invalid env entered")
        exit(0)

    env = env_fn.parallel_env(**env_kwargs)

    try:
        latest_policy = max(
            glob.glob(f".{str(env.metadata['name'])}/*.zip"),
            key=os.path.getctime,
        )
        print(latest_policy)
    except ValueError:
        print("Policy not found.")
        exit(0)

    if not os.path.exists(".obs_act_ag.pkl"):
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
        with open(".obs_act_ag.pkl", "wb") as f:
            pickle.dump(data_to_save, f)
    else:
        # Load from the file
        with open(".obs_act_ag.pkl", "rb") as f:
            loaded_data = pickle.load(f)

        # Extracting the saved data
        action = loaded_data["action"]
        relevant_obs = loaded_data["relevant_obs"]
        relevant_obs = np.array(relevant_obs)
        agent = loaded_data["agent"]
    print(f"{action=}")
    explainer = kernel_explainer(env, latest_policy, agent, action, seed=seed)
    shap_plot(
        relevant_obs, explainer, f"{args.env}_policy", feature_names, act_dict[action]
    )
