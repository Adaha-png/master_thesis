import argparse
import glob
import os
import pprint
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from sim_steps import sim_steps
from stable_baselines3 import PPO
from tqdm import tqdm


def pred(model, act, device, obs):
    obs = torch.tensor(obs).unsqueeze(0).to(device)
    action_net = model.policy.action_net.to(device)
    policy_net = model.policy.mlp_extractor.policy_net.to(device)
    vals = torch.softmax(action_net(policy_net(obs)), 2)
    vals = vals.cpu().detach().numpy()[0, :, act]
    return vals


def kernel_explainer(env, policy, agent, action, device, seed=None):
    X, _ = get_data(
        env, policy, agent=agent, total_steps=100, steps_per_cycle=25, seed=seed
    )
    model = PPO.load(policy)
    explainer = shap.KernelExplainer(partial(pred, model, action, device), X)
    return explainer


def shap_plot(X, explainer, output_file, feature_names, coordinate_name):
    # Compute SHAP values for the given dataset X

    shap_values = explainer.shap_values(X)

    # Handling the case where SHAP values contains multiple outputs
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)

    assert shap_values.shape[1] == len(
        feature_names
    ), "Mismatch between SHAP values and feature names dimensions."

    # Compute mean absolute SHAP values across all instances
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)
    sorted_indices = np.argsort(mean_shap_values)

    sorted_feature_names = np.array(feature_names)[sorted_indices]

    # Flatten SHAP values and corresponding feature values for coloring
    flattened_shap_values = shap_values[:, sorted_indices].flatten()
    repeated_feature_names = np.tile(sorted_feature_names, X.shape[0])
    feature_values = X.flatten()

    # Create color map
    norm = plt.Normalize(np.min(feature_values), np.max(feature_values))
    colors = plt.cm.viridis(norm(feature_values))

    # Create a new figure and axis
    _, ax = plt.subplots(figsize=(8, 12))

    # Scatter plot with color gradient
    scatter = ax.scatter(
        flattened_shap_values,
        repeated_feature_names,
        c=colors,
        s=10,
        cmap="bwr",
    )

    # Add horizontal lines for each feature
    for i in range(len(sorted_feature_names)):
        ax.axhline(i, color="gray", linestyle="--", linewidth=0.5)

    # Add vertical line at x=0
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_ylabel("Feature")
    ax.set_title(f"SHAP values for {coordinate_name} across all instances")

    # Add a color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Feature value")

    # Save the plot
    plt.savefig(
        f"tex/images/{output_file}_{coordinate_name}_shap.pdf".replace(" ", "_")
    )
    plt.close()


def get_data(env, policy, total_steps=10000, steps_per_cycle=250, agent=1, seed=None):
    observations = []
    actions = []
    num_cycles = total_steps // steps_per_cycle
    for i in range(num_cycles):
        if seed:
            step_results = sim_steps(
                env, policy, num_steps=steps_per_cycle, seed=seed + i + 200
            )
        else:
            step_results = sim_steps(env, policy, num_steps=steps_per_cycle)

        for entry in step_results:
            obs = entry.get("observation")
            act = entry.get("action")
            if obs is not None and act is not None:
                actions.append(act[agent])
                observations.append(obs[agent])
    return np.array(observations), np.array(actions)


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
            glob.glob(str(env.metadata["name"]) + "/*.zip"),
            key=os.path.getctime,
        )
        print(latest_policy)
    except ValueError:
        print("Policy not found.")
        exit(0)

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(
        kernel_explainer(
            env,
            latest_policy,
            0,
            0,
            seed=seed,
        )
    )
