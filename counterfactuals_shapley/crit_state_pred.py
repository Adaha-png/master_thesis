import glob
import lzma
import os
import pickle
from functools import partial

import numpy as np
import pandas
import ray
import torch
from captum.attr import IntegratedGradients
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from tqdm import tqdm

from counterfactuals_shapley.captum_grads import create_baseline
from counterfactuals_shapley.compare import get_torch_from_algo, make_plots
from counterfactuals_shapley.n_step_pred import (
    add_action,
    add_ig,
    add_shap,
    future_sight,
    get_future_data,
    one_hot_action,
)
from counterfactuals_shapley.shapley import kernel_explainer, pred
from counterfactuals_shapley.wrappers import numpyfy
from train_tune_eval.rllib_train import env_creator


def find_crit_state(net, seq, device):
    pred_func = partial(pred, net, None, device)
    max_diff = np.max(
        [np.max(pred_func(step)) - np.min(pred_func(step)) for step in seq]
    )
    return max_diff


def get_crit_data(histories, net, agent, device, m):
    histories = np.array(histories)

    X = []  # List of initial observations
    diff_vals = []  # List of difference values used to classify critical states

    for history in tqdm(histories, desc="getting crit data"):
        if len(history) < m:  # Skip histories that are too short
            continue

        n = np.random.randint(0, len(history) - m)
        history = {
            name: np.array([step[name]["observation"] for step in history])
            for name in history[0]
            if agent in name
        }

        for sequence in history.values():
            X.append(sequence[n])
            diff = find_crit_state(net, sequence[n : n + m], device)
            diff_vals.append(diff)

    # Convert to numpy arrays
    X = np.array(X)
    diff_vals = np.array(diff_vals)

    # Determine threshold (75th percentile of diff values)
    threshold = np.quantile(diff_vals, 0.5)

    # Label each initial observation as critical (1) or non-critical (0)
    Y = (diff_vals > threshold).astype(int)

    return X, Y


def compute(
    net,
    agent,
    feature_names,
    act_dict,
    extras="none",
    explainer_extras="none",
    memory="no_memory",
    device=torch.device("cpu"),
):
    if extras == "one-hot":
        feature_names.extend(act_dict.values())
    elif extras == "action":
        feature_names.append(extras)

    env = env_creator()
    env.reset()
    env_name = env.metadata["name"]

    if not os.path.exists(
        f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_data.xz"
    ):
        n_agents = len([ag for ag in env.possible_agents if agent in ag])
        paths = glob.glob(f".{env_name}/{memory}/prediction_data_part[0-9].xz")
        if len(paths) < np.ceil(10 / n_agents):
            paths = get_future_data(net, memory, agent, seed=921, finished=paths)

        X = []
        y = []
        for path in paths:
            with lzma.open(path, "rb") as f:
                seq = pickle.load(f)
            partX, party = get_crit_data(
                seq,
                net,
                agent,
                device,
                5,
            )
            X.extend(partX)
            y.extend(party)

        os.makedirs(f".{env.metadata['name']}/{memory}/{agent}/crit", exist_ok=True)

        with lzma.open(
            f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_data.xz", "wb"
        ) as f:
            pickle.dump((X, y), f)

    else:
        with lzma.open(
            f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_data.xz", "rb"
        ) as f:
            X, y = pickle.load(f)

    if extras != "none":
        if not os.path.exists(
            f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_data_action.xz"
        ):
            add_action(X, net, agent, memory, device, name_ider="crit")

            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_data_action.xz",
                "rb",
            ) as f:
                X = pickle.load(f)
        else:
            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_data_action.xz",
                "rb",
            ) as f:
                X = pickle.load(f)
        if extras == "one-hot":
            X = one_hot_action(X)

    if not os.path.exists(f".{env_name}/{memory}/{agent}/.baseline_future.pt"):
        baseline = create_baseline(X)
        torch.save(baseline, f".{env_name}/{memory}/{agent}/.baseline_future.pt")
    else:
        baseline = torch.load(
            f".{env_name}/{memory}/{agent}/.baseline_future.pt",
            map_location=device,
            weights_only=True,
        )

    if explainer_extras == "ig":
        if not os.path.exists(
            f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_data_{extras}_ig.xz"
        ):
            ig = IntegratedGradients(net)

            ig_partial = partial(
                ig.attribute,
                baselines=baseline,
                method="gausslegendre",
                return_convergence_delta=False,
            )

            X = add_ig(
                net,
                agent,
                memory,
                X,
                ig_partial,
                device,
                extras=extras,
                name_ider="crit",
            )
        else:
            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_data_{extras}_ig.xz",
                "rb",
            ) as f:
                X = pickle.load(f)

    elif explainer_extras == "shap":
        if not os.path.exists(
            f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_data_{extras}_shap.xz"
        ):
            X = add_shap(
                net, agent, memory, X, baseline, device, extras=extras, path_ider="crit"
            )

        else:
            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_data_{extras}_shap.xz",
                "rb",
            ) as f:
                X = pickle.load(f)

    pred_net = nn.Sequential(
        nn.Linear(len(X[0]), 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    ).to(device)

    if not os.path.exists(
        f".{env_name}/{memory}/{agent}/crit_models/pred_model_{extras}_{explainer_extras}.pt"
    ):
        pred_net = future_sight(
            agent,
            memory,
            device,
            pred_net,
            X,
            y,
            extras=extras,
            epochs=200,
            explainer_extras=explainer_extras,
            criterion=nn.BCELoss(),
            name_ider="crit_models",
        )
        pred_net.eval()
    else:
        pred_net.load_state_dict(
            torch.load(
                f".{env_name}/{memory}/{agent}/crit_models/pred_model_{extras}_{explainer_extras}.pt",
                weights_only=True,
                map_location=device,
            )
        )
        pred_net.eval()

    with torch.no_grad():
        criterion = nn.BCELoss()

        if os.path.exists(
            f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_test_data_{extras}_{explainer_extras}.xz"
        ):
            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_test_data_{extras}_{explainer_extras}.xz",
                "rb",
            ) as f:
                X_test, y_test = pickle.load(f)
        else:
            print("Test data not found, creating...")
            path = f".{env_name}/{memory}/prediction_test_data.xz"
            if not os.path.exists(f".{env_name}/{memory}/prediction_test_data.xz"):
                get_future_data(
                    net,
                    memory,
                    agent,
                    amount_cycles=10000,
                    steps_per_cycle=100,
                    test=True,
                    seed=483927,
                )

            if not os.path.exists(
                f".{env_name}/{memory}/{agent}/crit/prediction_test_data.xz"
            ):
                with lzma.open(path, "rb") as f:
                    seq = pickle.load(f)

                X_test, y_test = get_crit_data(seq, net, agent, device, 5)

                with lzma.open(
                    f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_test_data.xz",
                    "wb",
                ) as f:
                    pickle.dump((X_test, y_test), f)
            else:
                with lzma.open(
                    f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_test_data.xz",
                    "rb",
                ) as f:
                    X_test, y_test = pickle.load(f)

            if not os.path.exists(f".{env_name}/{memory}/{agent}/.baseline_future.pt"):
                baseline = create_baseline(X_test)
                torch.save(
                    baseline, f".{env_name}/{memory}/{agent}/.baseline_future.pt"
                )
            else:
                baseline = torch.load(
                    f".{env_name}/{memory}/{agent}/.baseline_future.pt",
                    map_location=device,
                    weights_only=True,
                )

            if not extras == "none":
                X_test = add_action(
                    X_test, net, agent, memory, device, name_ider="crit", save=False
                )
                if extras == "one-hot":
                    X_test = one_hot_action(X_test)

            if explainer_extras == "ig":
                ig = IntegratedGradients(net)
                ig_partial = partial(
                    ig.attribute,
                    baselines=baseline,
                    method="gausslegendre",
                    return_convergence_delta=False,
                )

                X_test = add_ig(
                    net,
                    agent,
                    memory,
                    X_test,
                    ig_partial,
                    device,
                    extras=extras,
                    save=False,
                    name_ider="crit",
                )

            elif explainer_extras == "shap":
                X_test = add_shap(
                    net,
                    agent,
                    memory,
                    X_test,
                    baseline,
                    device,
                    test=True,
                    extras=extras,
                    path_ider="crit",
                )

            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/crit/prediction_test_data_{extras}_{explainer_extras}.xz",
                "wb",
            ) as f:
                pickle.dump((X_test, y_test), f)

        X_test = torch.Tensor(numpyfy(X_test)).to(device)
        y_test = torch.Tensor(numpyfy(y_test)).to(device)
        test_outputs = pred_net(X_test).squeeze()
        test_loss = criterion(test_outputs, y_test).item()
        print(
            f"Loss on test set for {extras} and {explainer_extras} extras: {test_loss:.4f}"
        )

        predicted_labels = torch.round(test_outputs)

        accuracy = torch.sum(predicted_labels == y_test) / len(y_test)
        _, _, f_score, _ = precision_recall_fscore_support(y_test, predicted_labels)

    if not os.path.exists(
        f"tex/images/{env.metadata['name']}/{memory}/{agent}/None_{extras}_{explainer_extras}_shap.pgf"
    ):
        expl = kernel_explainer(pred_net, X_test, 0, device)
        indices = torch.randperm(len(X_test))[:50]
        make_plots(
            expl,
            X_test[indices],
            agent,
            memory,
            extras,
            explainer_extras,
        )

    return (test_loss, accuracy, f_score[0])


def crit_compare(agent, memory, feature_names, act_dict):
    extras = ["action", "one-hot", "none"]
    explainer_extras = ["none", "ig", "shap"]

    table = np.zeros((len(extras), len(explainer_extras), 3))

    ray.init(ignore_reinit_error=True)

    env_name = env_creator().metadata["name"]
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    policy_path = "file://" + os.path.abspath(f".{env_name}/{memory}/policies")

    algo = PPO.from_checkpoint(policy_path)

    net = get_torch_from_algo(algo, agent, memory, logits=True)

    for i, extra in enumerate(extras):
        for j, expl in enumerate(explainer_extras):
            outs = compute(
                net,
                agent,
                feature_names,
                act_dict,
                extras=extra,
                explainer_extras=expl,
                memory=memory,
            )
            for k, out in enumerate(outs):
                table[i, j, k] = out

    table = np.array(table)
    df = pandas.DataFrame(data=table[:, :, 1], columns=explainer_extras)

    with open(f".{env_name}/{memory}/{agent}/table_data_crit.txt", "w") as f:
        f.write(df.to_latex())

    return table
