import argparse
import glob
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients
from captum_grads import create_baseline
from counterfactuals import action_difference_with_model, counterfactuals_with_model
from n_step_pred import add_action, future_sight, get_future_data, one_hot_action
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from sim_steps import sim_steps
from stable_baselines3 import PPO
from torch import nn

from wrappers import numpyfy


def ig_extract_future(
    env,
    policy,
    net,
    obs,
    agent,
    coordinate,
    feature_names,
    device,
    seed=10923378429,
):
    coordinates = ["x", "y"]
    ig = IntegratedGradients(net)

    if not os.path.exists(f".baseline_future_{env.metadata["name"]}.pt"):
        baseline = create_baseline(
            env, policy, agent, device, steps_per_cycle=1, seed=seed
        )
        torch.save(baseline, f".baseline_future_{env.metadata["name"]}.pt")
    else:
        baseline = torch.load(
            f".baseline_future_{env.metadata["name"]}.pt", map_location=device
        )
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs)

    baseline = (
        torch.Tensor(
            [*baseline[0], *[0 for _ in range(len(obs) - len(baseline[0]))]],
        )
        .to(device=device, dtype=torch.float32)
        .unsqueeze(0)
    )

    obs = obs.to(
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    print(f"{baseline=}")
    print(f"{obs=}")
    attributions, approximation_error = ig.attribute(
        obs,
        baselines=baseline,
        target=coordinate,
        method="gausslegendre",
        return_convergence_delta=True,
    )

    print(f"{attributions=}")
    print(f"{approximation_error=}")
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.squeeze().detach().numpy()
    sorted_indices = np.argsort(np.abs(attributions))
    attributions = attributions[sorted_indices]
    feature_names = numpyfy(feature_names)[sorted_indices]

    print(
        f"Coordinate: {coordinates[coordinate]}, Value:{net.forward(obs)[0,coordinate]}"
    )

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
    plt.savefig(
        f"tex/images/intgrad_{coordinates[coordinate]}_{env.metadata['name']}.pdf".replace(
            " ", "_"
        )
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            "comms 1",
            "comms 2",
            "comms 3",
            "comms 4",
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
        policy_path = max(
            glob.glob(f".{str(env.metadata['name'])}/*.zip"),
            key=os.path.getctime,
        )
        print(policy_path)
    except ValueError:
        print("Policy not found in " + f".{str(env.metadata['name'])}/*.zip")
        exit(0)

    extras = "one-hot"

    if extras == "one-hot":
        feature_names.extend(act_dict.values())
    elif extras == "action":
        feature_names.append(extras)

    model = PPO.load(policy_path)

    if not os.path.exists(".pred_data/.prediction_data.pkl"):
        X, y = get_future_data(
            args.env, env_kwargs, policy_path, agent=0, steps_per_cycle=10, seed=921
        )
        with open(".pred_data/.prediction_data.pkl", "wb") as f:
            pickle.dump((X, y), f)
    else:
        with open(".pred_data/.prediction_data.pkl", "rb") as f:
            X, y = pickle.load(f)

    if extras != "none":
        if not os.path.exists(".pred_data/.prediction_data_action.pkl"):
            add_action(X, model)
            with open(".pred_data/.prediction_data_action.pkl", "rb") as f:
                X = pickle.load(f)
        else:
            with open(".pred_data/.prediction_data_action.pkl", "rb") as f:
                X = pickle.load(f)
        if extras == "one-hot":
            X = one_hot_action(X)

    net = nn.Sequential(
        nn.Linear(len(X[0]), 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, len(y[0])),
    )

    if not os.path.exists(f".pred_models/pred_model_{args.env}_{extras}.pt"):
        net = future_sight(
            args.env,
            device,
            net,
            X,
            y,
            extras=extras,
        )
        net.eval()
    else:
        net.load_state_dict(
            torch.load(
                f".pred_models/pred_model_{args.env}_{extras}.pt",
                weights_only=True,
                map_location=device,
            )
        )
        net.eval()

    ig_extract_future(
        env,
        policy_path,
        net,
        X[74],
        0,
        0,  # 0 coordinate is x, 1 is y
        feature_names,
        device,
    )

    ig_extract_future(
        env,
        policy_path,
        net,
        X[74],
        0,
        1,  # 0 coordinate is x, 1 is y
        feature_names,
        device,
    )
