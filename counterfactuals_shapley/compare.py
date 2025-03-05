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
from torch import nn
from tqdm import tqdm

from train_tune_eval.rllib_train import env_creator

from .captum_grads import create_baseline
from .n_step_pred import (
    add_action,
    add_ig,
    add_shap,
    future_sight,
    get_future_data,
    one_hot_action,
)
from .shapley import kernel_explainer, shap_plot
from .wrappers import numpyfy


def distance(predicted, y):
    if not isinstance(predicted, np.ndarray):
        predicted = numpyfy(predicted)
    if not isinstance(y, np.ndarray):
        y = numpyfy(y)
    return np.linalg.norm(predicted - y, axis=1)


def extract_observations(history, agent_name, n, m):
    obs_n = []
    obs_n_plus_m = []

    # Extract observation at step n
    if 0 <= n < len(history):
        step_record = history[n]
        for agent, data in step_record.items():
            if agent.startswith(agent_name):
                obs_n.append(data.get("observation"))
    else:
        return

    step_index = n + m
    if 0 <= step_index < len(history):
        step_record = history[step_index]
        for agent, data in step_record.items():
            if agent.startswith(agent_name):
                obs_n_plus_m.append(data.get("observation")[1:3])
    else:
        return
    return obs_n, obs_n_plus_m


def extract_pairs_from_histories(histories, agent_name, m, n=None):
    obs_pairs = [
        extract_observations(
            history, agent_name, n or np.random.randint(0, max(1, len(history) - m)), m
        )
        for history in tqdm(histories, desc="extracting obs and pos")
    ]

    initial_observations = np.array([pair[0] for pair in obs_pairs])
    initial_observations = initial_observations.reshape(
        initial_observations.shape[0] * initial_observations.shape[1],
        initial_observations.shape[2],
    )

    later_observations = np.array([pair[1] for pair in obs_pairs])
    later_observations = later_observations.reshape(
        later_observations.shape[0] * later_observations.shape[1],
        later_observations.shape[2],
    )

    return np.array(initial_observations), np.array(later_observations)


def get_torch_from_algo(algo, agent, memory):
    env = env_creator()
    path_ider = f".{env.metadata['name']}/{memory}/{agent}/torch/*.pt"
    torch_path = glob.glob(path_ider)
    if not torch_path == []:
        policy_net = torch.load(torch_path[0], weights_only=False)
    else:
        algo.get_policy(policy_id=agent).export_model(
            export_dir="/".join(path_ider.split("/")[:-1])
        )
        torch_path = glob.glob(path_ider)
        policy_net = torch.load(torch_path[0], weights_only=False)

    net_with_softmax = nn.Sequential(
        policy_net._hidden_layers, policy_net._logits, nn.Softmax(dim=0)
    )

    return net_with_softmax


def compute(
    policy_path,
    agent,
    feature_names,
    act_dict,
    extras="none",
    explainer_extras="none",
    memory="no_memory",
    seed=42,
    device=torch.device("cpu"),
):
    assert extras in ["none", "one-hot", "action"]
    assert explainer_extras in ["none", "ig", "shap"]
    assert memory in ["no_memory", "lstm", "attention"]

    if extras == "one-hot":
        feature_names.extend(act_dict.values())
    elif extras == "action":
        feature_names.append(extras)

    env = env_creator()
    env.reset()
    env_name = env.metadata["name"]

    ray.init(ignore_reinit_error=True)

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    policy_path = "file://" + os.path.abspath(f".{env_name}/{memory}/policies")

    algo = PPO.from_checkpoint(policy_path)

    net = get_torch_from_algo(algo, agent, memory)

    ray.shutdown()

    if not os.path.exists(f".{env_name}/{memory}/{agent}/pred_data/prediction_data.xz"):
        print("Prediction data not found, creating...")
        paths = glob.glob(f".{env_name}/{memory}/prediction_data_part[0-9].xz")
        if len(paths) < 10:
            paths = get_future_data(net, memory, seed=921, finished=paths)

        X = []
        y = []
        for path in paths:
            with lzma.open(path, "rb") as f:
                seq = pickle.load(f)

            partX, party = extract_pairs_from_histories(seq, agent, 10)
            X.extend(partX)
            y.extend(party)

        X = torch.Tensor(np.array(X))
        y = torch.Tensor(np.array(y))
        with lzma.open(
            f".{env_name}/{memory}/{agent}/pred_data/prediction_data.xz", "wb"
        ) as f:
            pickle.dump((X, y), f)
    else:
        with lzma.open(
            f".{env_name}/{memory}/{agent}/pred_data/prediction_data.xz", "rb"
        ) as f:
            X, y = pickle.load(f)

    if extras != "none":
        if not os.path.exists(
            f".{env_name}/{memory}/{agent}/pred_data/prediction_data_action.xz"
        ):
            add_action(X, net, agent, memory)
            with lzma.open(
                f".{env_name}/{memory}/{agent}/pred_data/prediction_data_action.xz",
                "rb",
            ) as f:
                X = pickle.load(f)
        else:
            with lzma.open(
                f".{env_name}/{memory}/{agent}/pred_data/prediction_data_action.xz",
                "rb",
            ) as f:
                X = pickle.load(f)
        if extras == "one-hot":
            X = one_hot_action(X)

    if explainer_extras == "ig":
        if not os.path.exists(
            f".{env.metadata['name']}/{memory}/{agent}/pred_data/prediction_data_ig_{extras}.xz"
        ):
            ig = IntegratedGradients(net)
            if not os.path.exists(f".{env_name}/{memory}/{agent}/.baseline_future.pt"):
                baseline = create_baseline(
                    net, agent, device, steps_per_cycle=100, seed=seed
                )
                torch.save(
                    baseline, f".{env_name}/{memory}/{agent}/.baseline_future.pt"
                )
            else:
                baseline = torch.load(
                    f".{env_name}/{memory}/{agent}/.baseline_future.pt",
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
                net,
                agent,
                memory,
                X,
                ig_partial,
                device,
                extras=extras,
            )
        else:
            with lzma.open(
                f".{env_name}/{memory}/{agent}/pred_data/prediction_data_ig_{extras}.xz",
                "rb",
            ) as f:
                X = pickle.load(f)

    elif explainer_extras == "shap":
        if not os.path.exists(
            f"{env_name}/{memory}/{agent}/pred_data/prediction_data_shap_{extras}.xz"
        ):
            paths = glob.glob(
                f".{env_name}/{memory}/{agent}/pred_data/prediction_data_shap_*.xz"
            )
            if len(paths) > 0:
                path = paths[0]
                with lzma.open(path, "rb") as f:
                    Obs_with_shap = pickle.load(f)
                    Obs_with_shap = numpyfy(Obs_with_shap)

                X = [np.array([*X[i], *Obs_with_shap[i, -18:]]) for i in range(len(X))]

                with lzma.open(
                    f".{env_name}/{memory}/{agent}/pred_data/prediction_data_shap_{extras}.xz",
                    "wb",
                ) as f:
                    pickle.dump(X, f)
            else:
                num_acts = env.action_space(agent + "_0").n

                expl = [
                    kernel_explainer(net, agent, i, device, seed=372894 * (i + 1))
                    for i in range(num_acts)
                ]

                X = add_shap(
                    net,
                    agent,
                    memory,
                    X,
                    expl,
                    device,
                    extras=extras,
                )

        else:
            with lzma.open(
                f".{env_name}/{memory}/{agent}/pred_data/prediction_data_shap_{extras}.xz",
                "rb",
            ) as f:
                X = pickle.load(f)

    pred_net = nn.Sequential(
        nn.Linear(len(X[0]), 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, len(y[0])),
    ).to(device)

    if not os.path.exists(
        f".{env_name}/{memory}/{agent}/pred_models/pred_model_{extras}_{explainer_extras}.pt"
    ):
        pred_net = future_sight(
            agent,
            memory,
            device,
            pred_net,
            X,
            y,
            extras=extras,
            explainer_extras=explainer_extras,
        )
        pred_net.eval()
    else:
        print(
            f".{env_name}/{memory}/{agent}/pred_models/pred_model_{extras}_{explainer_extras}.pt",
        )
        pred_net.load_state_dict(
            torch.load(
                f".{env_name}/{memory}/{agent}/pred_models/pred_model_{extras}_{explainer_extras}.pt",
                weights_only=True,
                map_location=device,
            )
        )
        pred_net.eval()

    with torch.no_grad():
        criterion = nn.MSELoss()

        if os.path.exists(
            f".{env_name}/{memory}/{agent}/pred_data/prediction_test_data_{extras}_{explainer_extras}.xz"
        ):
            with lzma.open(
                f".{env_name}/{memory}/{agent}/pred_data/prediction_test_data_{extras}_{explainer_extras}.xz",
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
                    amount_cycles=10000,
                    steps_per_cycle=100,
                    test=True,
                    seed=483927,
                )

            if not os.path.exists(
                f".{env_name}/{memory}/{agent}/pred_data/prediction_test_data.xz"
            ):
                with lzma.open(path, "rb") as f:
                    seq = pickle.load(f)

                X_test, y_test = extract_pairs_from_histories(seq, agent, 10)

                os.makedirs(
                    f".{env_name}/{memory}/{agent}/pred_data",
                    exist_ok=True,
                )
                with lzma.open(
                    f".{env_name}/{memory}/{agent}/pred_data/prediction_test_data.xz",
                    "wb",
                ) as f:
                    pickle.dump((X_test, y_test), f)

            else:
                with lzma.open(
                    f".{env_name}/{memory}/{agent}/pred_data/prediction_test_data.xz",
                    "rb",
                ) as f:
                    X_test, y_test = pickle.load(f)

            if not extras == "none":
                X_test = add_action(X_test, net, agent, memory, save=False)
                if extras == "one-hot":
                    X_test = one_hot_action(X_test)

            if explainer_extras == "ig":
                ig = IntegratedGradients(net)
                if not os.path.exists(
                    f".{env_name}/{memory}/{agent}/.baseline_future.pt"
                ):
                    baseline = create_baseline(
                        net, agent, device, steps_per_cycle=100, seed=seed
                    )
                    torch.save(
                        baseline, f".{env_name}/{memory}/{agent}/.baseline_future.pt"
                    )
                else:
                    baseline = torch.load(
                        f".{env_name}/{memory}/{agent}/.baseline_future.pt",
                        map_location=device,
                        weights_only=True,
                    )
                    print(f"{baseline.dtype=}")
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
                )

            elif explainer_extras == "shap":
                paths = glob.glob(
                    f".{env_name}/{memory}/{agent}/pred_data/prediction_test_data_*_shap.xz"
                )
                if len(paths) > 0:
                    path = paths[0]
                    with lzma.open(path, "rb") as f:
                        Obs_with_shap = pickle.load(f)[0]
                        Obs_with_shap = numpyfy(Obs_with_shap)

                    X_test = [
                        np.array([*X_test[i], *Obs_with_shap[i, -18:]])
                        for i in tqdm(range(len(X_test)))
                    ]
                else:
                    num_acts = env.action_space(agent + "_0").n

                    expl = [
                        kernel_explainer(net, agent, i, device, seed=372894 * (i + 1))
                        for i in range(num_acts)
                    ]

                    X_test = add_shap(
                        net,
                        agent,
                        memory,
                        X_test,
                        expl,
                        device,
                        extras=extras,
                    )

            with lzma.open(
                f".{env_name}/{memory}/{agent}/pred_data/prediction_test_data_{extras}_{explainer_extras}.xz",
                "wb",
            ) as f:
                pickle.dump((X_test, y_test), f)

        X_test = torch.Tensor(numpyfy(X_test)).to(device)
        y_test = torch.Tensor(numpyfy(y_test)).to(device)
        test_outputs = pred_net(X_test)

        test_loss = criterion(test_outputs, y_test).item()
        print(
            f"Loss on test set for {extras} and {explainer_extras} extras: {test_loss:.4f}"
        )

        distances = distance(test_outputs.cpu(), y_test.cpu())
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)
    expl = [kernel_explainer(pred_net, X_test, i, device) for i in range(len(y[0]))]
    print(y.shape)
    indices = torch.randperm(len(X_test))[:10]
    for i in range(len(y[0])):
        make_plots(
            expl,
            X_test[indices],
            agent,
            memory,
            extras,
            explainer_extras,
            i,
        )

    return (test_loss, avg_distance, max_distance)


def make_plots(explainer, X, agent, memory, extras, explainer_extras, target):
    if not isinstance(X, np.ndarray):
        X = numpyfy(X)

    env = env_creator()
    feature_names = env.feature_names

    if extras == "action":
        feature_names.append("action")
    elif extras == "one-hot":
        feature_names.extend(env.act_dict.keys())

    if explainer_extras != "none":
        feature_names.extend(
            [
                feature_name + " " + explainer_extras
                for feature_name in env.feature_names
            ]
        )
    if isinstance(explainer, list):
        for expl in explainer:
            shap_plot(
                agent,
                memory,
                X,
                expl,
                feature_names,
                target,
            )
    else:
        shap_plot(
            agent,
            memory,
            X,
            explainer,
            feature_names,
            target,
        )


def run_compare(policy_path, agent, memory, feature_names, act_dict, device):
    extras = ["none", "action", "one-hot"]
    explainer_extras = ["none", "ig", "shap"]

    table = np.zeros((3, len(extras), len(explainer_extras)))

    for i, extra in enumerate(extras):
        for j, expl in enumerate(explainer_extras):
            outs = compute(
                policy_path,
                agent,
                feature_names,
                act_dict,
                extras=extra,
                explainer_extras=expl,
                memory=memory,
                device=device,
            )
            for k, out in enumerate(outs):
                table[i, j, k] = out

    print(table)
    with open("table_data.pkl", "wb") as f:
        pickle.dump(table, f)

    return table
