import glob
import lzma
import os
import pickle
from functools import partial

import numpy as np
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
from counterfactuals_shapley.compare import (
    OneStepLSTM,
    get_torch_from_algo,
    make_plots,
    ttest,
)
from counterfactuals_shapley.n_step_pred import (
    add_action,
    add_ig,
    add_shap,
    future_sight,
    get_future_data,
    one_hot_action,
)
from counterfactuals_shapley.shapley import kernel_explainer
from counterfactuals_shapley.wrappers import numpyfy
from train_tune_eval.rllib_train import env_creator


def find_crit_state(seq):
    max_diff = np.max([step["criticality"] for step in seq])
    return max_diff


def get_crit_data(histories, agent, m):
    histories = np.array(histories)

    X = []
    diff_vals = []

    for history in tqdm(histories, desc="getting crit data"):
        if len(history) < m:  # Skip histories that are too short
            continue

        n = np.random.randint(0, len(history) - m)
        history = {
            name: np.array([step[name] for step in history])
            for name in history[0]
            if agent in name
        }

        for sequence in history.values():
            X.append(sequence[n]["observation"])
            diff = find_crit_state(sequence[n : n + m])

            diff_vals.append(diff)

    # Convert to numpy arrays
    X = np.array(X)
    diff_vals = np.array(diff_vals)
    print(diff_vals)
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
    run,
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
        f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_data.xz"
    ):
        n_agents = len([ag for ag in env.possible_agents if agent in ag])
        paths = glob.glob(f".{env_name}/{memory}/prediction_data_part[0-9].xz")
        if len(paths) < np.ceil(10 / n_agents):
            paths = get_future_data(
                net, memory, agent, device, seed=921, finished=paths
            )

        X = []
        y = []
        for path in paths:
            with lzma.open(path, "rb") as f:
                seq = pickle.load(f)
            partX, party = get_crit_data(
                seq,
                agent,
                5,
            )
            X.extend(partX)
            y.extend(party)

        os.makedirs(
            f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit", exist_ok=True
        )

        with lzma.open(
            f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_data.xz",
            "wb",
        ) as f:
            pickle.dump((X, y), f)

    else:
        with lzma.open(
            f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_data.xz",
            "rb",
        ) as f:
            X, y = pickle.load(f)

    n_feats = len(env.feature_names)
    if memory != "no_memory":
        out_layers = nn.Sequential(*net[2:])
        net = OneStepLSTM(
            inp_layers=net[0],
            lstm=net[1],
            out_layers=out_layers,
            n_feats=n_feats,
        )

    if extras != "none":
        if not os.path.exists(
            f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_data_action.xz"
        ):
            add_action(X, net, agent, run, memory, device, name_ider="crit")

            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_data_action.xz",
                "rb",
            ) as f:
                X = pickle.load(f)
        else:
            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_data_action.xz",
                "rb",
            ) as f:
                X = pickle.load(f)
        if extras == "one-hot":
            X = one_hot_action(X)

    if not os.path.exists(
        f".{env_name}/{memory}/{agent}/run_{run}/.baseline_future.pt"
    ):
        baseline = create_baseline(X, n_feats)
        torch.save(
            baseline, f".{env_name}/{memory}/{agent}/run_{run}/.baseline_future.pt"
        )
    else:
        baseline = torch.load(
            f".{env_name}/{memory}/{agent}/run_{run}/.baseline_future.pt",
            map_location=device,
            weights_only=True,
        )

    if explainer_extras == "ig":
        if not os.path.exists(
            f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_data_{extras}_ig.xz"
        ):
            paths = glob.glob(
                f".{env_name}/{memory}/{agent}/run_{run}/crit/prediction_data_*_ig.xz"
            )
            if len(paths) > 0:
                path = paths[0]
                with lzma.open(path, "rb") as f:
                    Obs_with_ig = pickle.load(f)
                    Obs_with_ig = numpyfy(Obs_with_ig)
                X = [
                    np.array([*X[i], *Obs_with_ig[i, -n_feats:]]) for i in range(len(X))
                ]

                with lzma.open(
                    f".{env_name}/{memory}/{agent}/run_{run}/crit/prediction_data_{extras}_ig.xz",
                    "wb",
                ) as f:
                    pickle.dump(X, f)
            else:
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
                    run,
                    memory,
                    X,
                    ig_partial,
                    device,
                    extras=extras,
                    name_ider="crit",
                )
        else:
            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_data_{extras}_ig.xz",
                "rb",
            ) as f:
                X = pickle.load(f)

    elif explainer_extras == "shap":
        if not os.path.exists(
            f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_data_{extras}_shap.xz"
        ):
            paths = glob.glob(
                f".{env_name}/{memory}/{agent}/run_{run}/crit/prediction_data_*_shap.xz"
            )
            if len(paths) > 0:
                path = paths[0]
                with lzma.open(path, "rb") as f:
                    Obs_with_shap = pickle.load(f)
                    Obs_with_shap = numpyfy(Obs_with_shap)
                feats = len(env.feature_names)
                X = [
                    np.array([*X[i], *Obs_with_shap[i, -feats:]]) for i in range(len(X))
                ]

                with lzma.open(
                    f".{env_name}/{memory}/{agent}/run_{run}/crit/prediction_data_{extras}_shap.xz",
                    "wb",
                ) as f:
                    pickle.dump(X, f)
            else:
                X = add_shap(
                    net,
                    agent,
                    run,
                    memory,
                    X,
                    baseline,
                    device,
                    extras=extras,
                    path_ider="crit",
                )

        else:
            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_data_{extras}_shap.xz",
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
        f".{env_name}/{memory}/{agent}/run_{run}/crit_models/pred_model_{extras}_{explainer_extras}.pt"
    ):
        pred_net = future_sight(
            agent,
            run,
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
                f".{env_name}/{memory}/{agent}/run_{run}/crit_models/pred_model_{extras}_{explainer_extras}.pt",
                weights_only=True,
                map_location=device,
            )
        )
        pred_net.eval()

    with torch.no_grad():
        criterion = nn.BCELoss()

        if os.path.exists(
            f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_test_data_{extras}_{explainer_extras}.xz"
        ):
            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_test_data_{extras}_{explainer_extras}.xz",
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
                    device,
                    amount_cycles=10000,
                    steps_per_cycle=100,
                    test=True,
                    seed=483927,
                )

            if not os.path.exists(
                f".{env_name}/{memory}/{agent}/run_{run}/crit/prediction_test_data.xz"
            ):
                with lzma.open(path, "rb") as f:
                    seq = pickle.load(f)

                X_test, y_test = get_crit_data(seq, agent, 5)

                with lzma.open(
                    f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_test_data.xz",
                    "wb",
                ) as f:
                    pickle.dump((X_test, y_test), f)
            else:
                with lzma.open(
                    f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_test_data.xz",
                    "rb",
                ) as f:
                    X_test, y_test = pickle.load(f)

            if not os.path.exists(
                f".{env_name}/{memory}/{agent}/run_{run}/.baseline_future.pt"
            ):
                baseline = create_baseline(X_test, n_feats)
                torch.save(
                    baseline,
                    f".{env_name}/{memory}/{agent}/run_{run}/.baseline_future.pt",
                )
            else:
                baseline = torch.load(
                    f".{env_name}/{memory}/{agent}/run_{run}/.baseline_future.pt",
                    map_location=device,
                    weights_only=True,
                )

            if not extras == "none":
                X_test = add_action(
                    X_test,
                    net,
                    agent,
                    run,
                    memory,
                    device,
                    name_ider="crit",
                    save=False,
                )
                if extras == "one-hot":
                    X_test = one_hot_action(X_test)

            if explainer_extras == "ig":
                paths = glob.glob(
                    f".{env_name}/{memory}/{agent}/run_{run}/crit/prediction_test_data_*_ig.xz"
                )
                if len(paths) > 0:
                    path = paths[0]
                    with lzma.open(path, "rb") as f:
                        Obs_with_ig = pickle.load(f)[0]
                        Obs_with_ig = numpyfy(Obs_with_ig)

                    X_test = [
                        np.array([*X_test[i], *Obs_with_ig[i, -n_feats:]])
                        for i in tqdm(range(len(X_test)))
                    ]
                else:
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
                        run,
                        memory,
                        X_test,
                        ig_partial,
                        device,
                        extras=extras,
                        save=False,
                        name_ider="crit",
                    )

            elif explainer_extras == "shap":
                paths = glob.glob(
                    f".{env_name}/{memory}/{agent}/run_{run}/crit/prediction_test_data_*_shap.xz"
                )
                if len(paths) > 0:
                    path = paths[0]
                    with lzma.open(path, "rb") as f:
                        Obs_with_shap = pickle.load(f)[0]
                        Obs_with_shap = numpyfy(Obs_with_shap)

                    X_test = [
                        np.array([*X_test[i], *Obs_with_shap[i, -n_feats:]])
                        for i in tqdm(range(len(X_test)))
                    ]

                else:
                    X_test = add_shap(
                        net,
                        agent,
                        run,
                        memory,
                        X_test,
                        baseline,
                        device,
                        test=True,
                        extras=extras,
                        path_ider="crit",
                    )

            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/run_{run}/crit/prediction_test_data_{extras}_{explainer_extras}.xz",
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
    runs = 10
    extras = ["none", "action", "one-hot"]
    explainer_extras = ["none", "ig", "shap"]

    table = np.zeros((len(extras), len(explainer_extras)))

    ray.init(ignore_reinit_error=True)

    env_name = env_creator().metadata["name"]
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    policy_path = "file://" + os.path.abspath(f".{env_name}/{memory}/policies")

    algo = PPO.from_checkpoint(policy_path)

    net = get_torch_from_algo(algo, agent, memory, logits=True)

    finished = glob.glob(f".{env_name}/{memory}/{agent}/run_*/tables/table_crit.pkl")
    for run in range(runs):
        print(f"{run=}")
        if f".{env_name}/{memory}/{agent}/run_{run}/tables/table_crit.pkl" in finished:
            continue

        for i, extra in enumerate(extras):
            for j, expl in enumerate(explainer_extras):
                outs = compute(
                    net,
                    agent,
                    feature_names,
                    act_dict,
                    run,
                    extras=extra,
                    explainer_extras=expl,
                    memory=memory,
                )
                table[i, j] = outs[1]

        table = np.array(table)

        with open(f".{env_name}/{memory}/{agent}/run_{run}/table_crit.pkl", "wb") as f:
            pickle.dump(table, f)

    ttest("crit", agent, memory, explainer_extras)
