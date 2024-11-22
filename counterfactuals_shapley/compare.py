import argparse
import glob
import os
import pickle
import random
from functools import partial

import numpy as np
import torch
from captum.attr import IntegratedGradients
from captum_grads import create_baseline
from n_step_pred import (
    add_action,
    add_ig,
    future_sight,
    get_future_data,
    one_hot_action,
)
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from shapley import shap_plot
from stable_baselines3 import PPO
from torch import nn


def distance(predicted, y):
    if not isinstance(predicted, np.ndarray):
        predicted = np.array(predicted)
    if not isinstance(y, np.ndarray):
        y = np.array(y)
    return np.linalg.norm(predicted - y, axis=1)


def compute(
    policy_path,
    feature_names,
    act_dict,
    extras="none",
    explainer_extras="none",
):
    if extras == "one-hot":
        feature_names.extend(act_dict.values())
    elif extras == "action":
        feature_names.append(extras)

    model = PPO.load(policy_path)

    if not os.path.exists(".prediction_data.pkl"):
        X, y = get_future_data(
            args.env, env_kwargs, policy_path, agent=0, steps_per_cycle=10, seed=921
        )
        with open(".prediction_data.pkl", "wb") as f:
            pickle.dump((X, y), f)
    else:
        with open(".prediction_data.pkl", "rb") as f:
            X, y = pickle.load(f)

    if extras != "none":
        if not os.path.exists(".prediction_data_action.pkl"):
            add_action(X, model)
            with open(".prediction_data_action.pkl", "rb") as f:
                X = pickle.load(f)
        else:
            with open(".prediction_data_action.pkl", "rb") as f:
                X = pickle.load(f)
        if extras == "one-hot":
            X = one_hot_action(X)

    if explainer_extras == "ig":
        if not os.path.exists(
            f".prediction_data_ig_{extras}_{env.metadata['name']}.pkl"
        ):
            policy_net = nn.Sequential(
                *model.policy.mlp_extractor.policy_net,
                model.policy.action_net,
                nn.Softmax(),
            ).to(device)

            ig = IntegratedGradients(policy_net)
            if not os.path.exists(f".baseline_future_{env.metadata['name']}.pt"):
                baseline = create_baseline(
                    env, policy_path, 0, device, steps_per_cycle=1, seed=seed
                )
                torch.save(baseline, f".baseline_future_{env.metadata['name']}.pt")
            else:
                baseline = torch.load(
                    f".baseline_future_{env.metadata['name']}.pt",
                    map_location=device,
                    weights_only=True,
                )

            ig_partial = partial(
                ig.attribute,
                baselines=baseline,
                method="gausslegendre",
                return_convergence_delta=False,
            )
            X = add_ig(
                X, ig_partial, env, device, policy_path=policy_path, extras=extras
            )
        else:
            with open(
                f".prediction_data_ig_{extras}_{env.metadata['name']}.pkl",
                "rb",
            ) as f:
                X = pickle.load(f)

    net = nn.Sequential(
        nn.Linear(len(X[0]), 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, len(y[0])),
    ).to(device)

    if not os.path.exists(
        f".pred_models/pred_model_{args.env}_{extras}_{explainer_extras}.pt"
    ):
        net = future_sight(
            args.env,
            device,
            net,
            X,
            y,
            extras=extras,
            explainer_extras=explainer_extras,
        )
        net.eval()
    else:
        net.load_state_dict(
            torch.load(
                f".pred_models/pred_model_{args.env}_{extras}_{explainer_extras}.pt",
                weights_only=True,
                map_location=device,
            )
        )
        net.eval()

    with torch.no_grad():
        criterion = nn.MSELoss()
        if not os.path.exists(".prediction_test_data.pkl"):
            X_test, y_test = get_future_data(
                args.env,
                env_kwargs,
                policy_path,
                agent=0,
                amount_cycles=10000,
                steps_per_cycle=10,
                seed=483927,
            )
            with open(".prediction_test_data.pkl", "wb") as f:
                pickle.dump((X_test, y_test), f)
        else:
            with open(".prediction_test_data.pkl", "rb") as f:
                X_test, y_test = pickle.load(f)

        if not extras == "none":
            X_test = add_action(X_test, model, save=False)
            if extras == "one-hot":
                X_test = one_hot_action(X_test)

        if explainer_extras == "ig":
            policy_net = nn.Sequential(
                *model.policy.mlp_extractor.policy_net,
                model.policy.action_net,
                nn.Softmax(),
            ).to(device)

            ig = IntegratedGradients(policy_net)
            if not os.path.exists(f".baseline_future_{env.metadata['name']}.pt"):
                baseline = create_baseline(
                    env, policy_path, 0, device, steps_per_cycle=1, seed=seed
                )
                torch.save(baseline, f".baseline_future_{env.metadata['name']}.pt")
            else:
                baseline = torch.load(
                    f".baseline_future_{env.metadata['name']}.pt",
                    map_location=device,
                    weights_only=True,
                )
            ig_partial = partial(
                ig.attribute,
                baselines=baseline,
                method="gausslegendre",
                return_convergence_delta=False,
            )

            X_test = add_ig(
                X_test,
                ig_partial,
                env,
                device,
                policy_path=policy_path,
                extras=extras,
                save=False,
            )

        X_test = torch.Tensor(np.array(X_test)).to(device)
        y_test = torch.Tensor(np.array(y_test)).to(device)
        test_outputs = net(X_test)

        test_loss = criterion(test_outputs, y_test).item()
        print(
            f"Loss on test set for {extras} and {explainer_extras} extras: {test_loss:.4f}"
        )

        distances = distance(test_outputs.cpu(), y_test.cpu())
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)

    return (test_loss, avg_distance, max_distance)


def make_plots(explainer, X, filename):
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    coordinate = 0
    coordinate_names = ["x", "y"]

    shap_plot(
        X,
        explainer,
        filename,
        feature_names,
        coordinate_names[coordinate],
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

    extras = ["none", "action", "one-hot"]
    explainer_extras = ["none", "ig", "shap"]

    table = np.zeros((3, len(extras), len(explainer_extras)))

    for i, extra in enumerate(extras):
        for j, expl in enumerate(explainer_extras):
            outs = compute(
                policy_path,
                feature_names,
                act_dict,
                extras=extra,
                explainer_extras=expl,
            )
            for k, out in enumerate(outs):
                table[i, j, k] = out

    print(table)
    with open("table_data.pkl", "wb") as f:
        pickle.dump(table, f)
