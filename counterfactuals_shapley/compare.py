import copy
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
from scipy.stats import ttest_ind
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
    plot_losses,
)
from .shapley import kernel_explainer, shap_plot
from .wrappers import numpyfy


class OneStepLSTM(nn.Module):
    def __init__(self, inp_layers, lstm, out_layers, n_feats):
        super().__init__()
        self.inp_layers = inp_layers
        self.lstm = lstm
        self.out_layers = out_layers
        self.h_c = None
        self.n_feats = n_feats

    def forward(self, inp):
        x = self.inp_layers(inp)
        x, _ = self.lstm(x, self.h_c)
        x = self.out_layers(x)
        return x

    def set_hidden(self, obs_with_mem):
        inp = obs_with_mem[: self.n_feats].unsqueeze(0)
        mem = obs_with_mem[self.n_feats :]

        mem_len = int(len(mem) / 2)

        h = torch.Tensor(mem[:mem_len]).unsqueeze(0)
        c = torch.Tensor(mem[mem_len:]).unsqueeze(0)

        self.h_c = (h, c)

        return self.forward(inp)


class ExtractTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[0]


def distance(predicted, y):
    if not isinstance(predicted, np.ndarray):
        predicted = numpyfy(predicted)
    if not isinstance(y, np.ndarray):
        y = numpyfy(y)
    return np.linalg.norm(predicted - y, axis=1)


def extract_observations(history, agent_name, memory, n, m, coords_ind):
    obs_n = []
    obs_n_plus_m = []

    if 0 <= n < len(history):
        step_record = history[n]
        for agent, data in step_record.items():
            if agent.startswith(agent_name):
                if memory == "lstm":
                    obs_with_mem = np.array(
                        [
                            *data["observation"],
                            *data["memory"][0].squeeze().cpu().detach(),
                            *data["memory"][1].squeeze().cpu().detach(),
                        ]
                    )
                    obs_n.append(obs_with_mem)

                elif memory == "attention":
                    raise NotImplementedError
                else:
                    obs_n.append(data.get("observation"))

    else:
        return

    step_index = n + m
    if 0 <= step_index < len(history):
        step_record = history[step_index]
        for agent, data in step_record.items():
            if agent.startswith(agent_name):
                obs_n_plus_m.append([data.get("observation")[i] for i in coords_ind])
    else:
        return

    return obs_n, obs_n_plus_m


def extract_pairs_from_histories(histories, agent_name, memory, m, n=None):
    env = env_creator()
    coords_ind = env.coords_ind
    env.close()

    obs_pairs = [
        extract_observations(
            history,
            agent_name,
            memory,
            n or np.random.randint(0, max(1, len(history) - m)),
            m,
            coords_ind,
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
    policy_net = algo.get_policy(policy_id=agent).model

    if memory == "no_memory":
        net = nn.Sequential(policy_net._hidden_layers, policy_net._logits)
        return net

    elif memory == "lstm":
        return (
            policy_net._hidden_layers,
            policy_net.lstm,
            policy_net._logits_branch,
        )

    else:
        return NotImplementedError


def compute(
    net,
    algo,
    agent,
    feature_names,
    act_dict,
    run,
    extras="none",
    explainer_extras="none",
    memory="no_memory",
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

    if not os.path.exists(
        f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data.xz"
    ):
        print("Prediction data not found, creating...")
        paths = glob.glob(f".{env_name}/{memory}/prediction_data_part[0-9].xz")
        n_agents = len([ag for ag in env.possible_agents if agent in ag])
        if len(paths) < int(np.ceil(10 / n_agents)):
            paths = get_future_data(
                algo,
                env_creator,
                agent,
                memory=memory,
                device=device,
                seed=921,
                finished=paths,
            )

        X = []
        y = []
        for path in paths:
            with lzma.open(path, "rb") as f:
                seq = pickle.load(f)

            partX, party = extract_pairs_from_histories(seq, agent, memory, 10)
            X.extend(partX)
            y.extend(party)

        X = torch.Tensor(np.array(X))
        y = torch.Tensor(np.array(y))

        np.random.seed(run)
        indices = np.random.randint(0, len(X) - 1, size=len(X))
        X = X[indices]
        y = y[indices]

        os.makedirs(f".{env_name}/{memory}/{agent}/run_{run}/pred_data", exist_ok=True)
        with lzma.open(
            f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data.xz", "wb"
        ) as f:
            pickle.dump((X, y), f)
    else:
        with lzma.open(
            f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data.xz", "rb"
        ) as f:
            X, y = pickle.load(f)

    n_feats = len(env.feature_names)

    net_list = None
    net_log = copy.deepcopy(net)

    if memory == "no_memory":
        net = nn.Sequential(net, nn.Softmax())

    elif memory == "lstm":
        net_list = copy.deepcopy(net)
        out_layers = nn.Sequential(net[2], nn.Softmax())
        net = OneStepLSTM(
            inp_layers=net[0],
            lstm=net[1],
            out_layers=out_layers,
            n_feats=n_feats,
        )

    if extras != "none":
        if not os.path.exists(
            f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data_action.xz"
        ):
            add_action(X, net, agent, run, memory, device)
            with lzma.open(
                f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data_action.xz",
                "rb",
            ) as f:
                X = pickle.load(f)
        else:
            with lzma.open(
                f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data_action.xz",
                "rb",
            ) as f:
                X = pickle.load(f)
        if extras == "one-hot":
            X = one_hot_action(X)

    if explainer_extras == "ig":
        if not os.path.exists(
            f".{env.metadata['name']}/{memory}/{agent}/run_{run}/pred_data/prediction_data_{extras}_ig.xz"
        ):
            paths = glob.glob(
                f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data_*_ig.xz"
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
                    f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data_{extras}_ig.xz",
                    "wb",
                ) as f:
                    pickle.dump(X, f)
            else:
                ig = IntegratedGradients(net.cpu())
                if not os.path.exists(
                    f".{env_name}/{memory}/{agent}/run_{run}/.baseline_future.pt"
                ):
                    baseline = create_baseline(X, n_feats)
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

                baseline = baseline.to(dtype=torch.float32, device="cpu")
                X = torch.Tensor(X).cpu()
                ig_partial = partial(
                    ig.attribute,
                    baselines=baseline,
                    method="gausslegendre",
                    return_convergence_delta=False,
                )
                X = add_ig(
                    net.cpu(),
                    agent,
                    run,
                    memory,
                    X,
                    ig_partial,
                    device,
                    extras=extras,
                )
        else:
            with lzma.open(
                f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data_{extras}_ig.xz",
                "rb",
            ) as f:
                X = pickle.load(f)

    elif explainer_extras == "shap":
        if not os.path.exists(
            f"{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data_{extras}_shap.xz"
        ):
            paths = glob.glob(
                f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data_*_shap.xz"
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
                    f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data_{extras}_shap.xz",
                    "wb",
                ) as f:
                    pickle.dump(X, f)
            else:
                if not os.path.exists(
                    f".{env_name}/{memory}/{agent}/run_{run}/.baseline_future.pt"
                ):
                    baseline = create_baseline(X, n_feats)
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

                baseline = baseline.to(torch.float32)
                X = add_shap(
                    net,
                    agent,
                    run,
                    memory,
                    torch.Tensor(X),
                    baseline,
                    device,
                    extras=extras,
                )

        else:
            with lzma.open(
                f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_data_{extras}_shap.xz",
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
        f".{env_name}/{memory}/{agent}/run_{run}/pred_models/pred_model_{extras}_{explainer_extras}.pt"
    ):
        pred_net = future_sight(
            agent,
            run,
            memory,
            device,
            pred_net,
            X,
            y,
            epochs=1000,
            extras=extras,
            explainer_extras=explainer_extras,
        )
        pred_net.eval()

    else:
        print(
            f".{env_name}/{memory}/{agent}/run_{run}/pred_models/pred_model_{extras}_{explainer_extras}.pt",
        )
        pred_net.load_state_dict(
            torch.load(
                f".{env_name}/{memory}/{agent}/run_{run}/pred_models/pred_model_{extras}_{explainer_extras}.pt",
                weights_only=True,
                map_location=device,
            )
        )
        pred_net.eval()

    with torch.no_grad():
        criterion = nn.MSELoss()

        if os.path.exists(
            f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_test_data_{extras}_{explainer_extras}.xz"
        ):
            with lzma.open(
                f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_test_data_{extras}_{explainer_extras}.xz",
                "rb",
            ) as f:
                X_test, y_test = pickle.load(f)
        else:
            print("Test data not found, creating...")
            path = f".{env_name}/{memory}/prediction_test_data.xz"
            if not os.path.exists(f".{env_name}/{memory}/prediction_test_data.xz"):
                get_future_data(
                    algo,
                    env_creator,
                    agent,
                    memory=memory,
                    device=device,
                    amount_cycles=5000,
                    steps_per_cycle=100,
                    test=True,
                    seed=483927,
                )

            if not os.path.exists(
                f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_test_data.xz"
            ):
                with lzma.open(path, "rb") as f:
                    seq = pickle.load(f)

                X_test, y_test = extract_pairs_from_histories(seq, agent, memory, 10)

                os.makedirs(
                    f".{env_name}/{memory}/{agent}/run_{run}/pred_data",
                    exist_ok=True,
                )
                with lzma.open(
                    f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_test_data.xz",
                    "wb",
                ) as f:
                    pickle.dump((X_test, y_test), f)

            else:
                with lzma.open(
                    f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_test_data.xz",
                    "rb",
                ) as f:
                    X_test, y_test = pickle.load(f)

            if not extras == "none":
                X_test = add_action(X_test, net, agent, run, memory, device, save=False)
                if extras == "one-hot":
                    X_test = one_hot_action(X_test)

            if explainer_extras == "ig":
                paths = glob.glob(
                    f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_test_data_*_ig.xz"
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
                    ig = IntegratedGradients(net.cpu())
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

                    baseline = baseline.cpu()
                    ig_partial = partial(
                        ig.attribute,
                        baselines=baseline,
                        method="gausslegendre",
                        return_convergence_delta=False,
                    )

                    X_test = torch.Tensor(X_test).cpu()
                    X_test = add_ig(
                        net.cpu(),
                        agent,
                        run,
                        memory,
                        X_test,
                        ig_partial,
                        device,
                        extras=extras,
                        save=False,
                    )

            elif explainer_extras == "shap":
                paths = glob.glob(
                    f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_test_data_*_shap.xz"
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
                    if not os.path.exists(
                        f".{env_name}/{memory}/{agent}/run_{run}/.baseline_future.pt"
                    ):
                        baseline = create_baseline(X, n_feats)
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

                    baseline = baseline.to(torch.float32)
                    X_test = add_shap(
                        net,
                        agent,
                        run,
                        memory,
                        torch.Tensor(X_test),
                        baseline,
                        device,
                        extras=extras,
                        test=True,
                    )

            with lzma.open(
                f".{env_name}/{memory}/{agent}/run_{run}/pred_data/prediction_test_data_{extras}_{explainer_extras}.xz",
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

    plot_paths = glob.glob(
        f"tex/images/{env.metadata['name']}/{memory}/{agent}/[0-9]_{extras}_{explainer_extras}_shap.pgf"
    )

    if len(plot_paths) < len(y[0]):
        expl = [kernel_explainer(pred_net, X_test, i, device) for i in range(len(y[0]))]
        indices = torch.randperm(len(X_test))[:50]
        make_plots(
            expl,
            X_test[indices],
            agent,
            memory,
            extras,
            explainer_extras,
        )

    else:
        print("Plots already created, delete existing to make new ones, skipping..")

    return (test_loss, avg_distance, max_distance)


def make_plots(explainer, X, agent, memory, extras, explainer_extras):
    if not isinstance(X, np.ndarray):
        X = numpyfy(X)

    env = env_creator()
    feature_names = copy.deepcopy(env.feature_names)

    slices = [len(feature_names)]

    if memory == "lstm":
        feature_names.extend([f"h {i}" for i in range(64)])
        slices.append(len(feature_names))

        feature_names.extend([f"c {i}" for i in range(64)])
        slices.append(len(feature_names))

    if extras == "action":
        feature_names.append("Integer")
        slices.append(len(feature_names))
    elif extras == "one-hot":
        feature_names.extend(env.act_dict.keys())
        slices.append(len(feature_names))

    if explainer_extras != "none":
        feature_names.extend(
            [
                feature_name + " " + explainer_extras.upper()
                for feature_name in env.feature_names
            ]
        )
        slices.append(len(feature_names))

    if isinstance(explainer, list):
        for i, expl in enumerate(explainer):
            shap_plot(
                agent,
                memory,
                X,
                expl,
                feature_names,
                i,
                extras,
                explainer_extras,
                slices=slices,
            )
    else:
        shap_plot(
            agent,
            memory,
            X,
            explainer,
            feature_names,
            None,
            extras,
            explainer_extras,
            slices=slices,
        )


def ttest(name_ider, agent, memory, explainer_extras):
    env_name = env_creator().metadata["name"]

    table_paths = glob.glob(f".{env_name}/{memory}/{agent}/run_*/table_{name_ider}.pkl")
    full_table = []
    for path in table_paths:
        with open(path, "rb") as f:
            table = pickle.load(f)

        full_table.append(table)

    full_table = np.array(full_table)
    mean_table = np.mean(
        full_table,
        axis=0,
    )

    with open(f".{env_name}/{memory}/{agent}/table_{name_ider}_mean.txt", "w") as f:
        df = pandas.DataFrame(data=mean_table, columns=explainer_extras)
        f.write(df.to_latex())

    p_table = np.ones((2, 2), dtype=np.float32)
    for i in range(len(p_table)):
        for j in range(len(p_table[i])):
            if j == 0:
                p_table[i, j] = ttest_ind(
                    full_table[:, 0, 0],
                    full_table[:, i, j],
                    alternative="greater" if name_ider == "pred" else "less",
                ).pvalue
            else:
                p_table[i, j] = ttest_ind(
                    full_table[:, i, 0],
                    full_table[:, i, j],
                    alternative="greater" if name_ider == "pred" else "less",
                ).pvalue

    with open(f".{env_name}/{memory}/{agent}/table_{name_ider}_p.txt", "w") as f:
        df = pandas.DataFrame(data=p_table, columns=explainer_extras)
        f.write(df.to_latex())


def run_compare(agent, memory, feature_names, act_dict, device):
    runs = 10

    extras = ["none", "one-hot"]
    explainer_extras = ["none", "ig"]

    table = np.zeros((len(extras), len(explainer_extras)))

    ray.init(ignore_reinit_error=True)
    env_name = env_creator().metadata["name"]

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    policy_path = "file://" + os.path.abspath(f".{env_name}/{memory}/policies")

    algo = PPO.from_checkpoint(policy_path)

    net = get_torch_from_algo(algo, agent, memory)

    for run in range(runs):
        print(f"{run=}")

        for i, extra in enumerate(extras):
            for j, expl in enumerate(explainer_extras):
                print(f"Computing for {extra=} and {expl=}")
                outs = compute(
                    net,
                    algo,
                    agent,
                    feature_names,
                    act_dict,
                    run,
                    extras=extra,
                    explainer_extras=expl,
                    memory=memory,
                    device=device,
                )
                table[i, j] = outs[1]

        table = np.array(table)

        with open(f".{env_name}/{memory}/{agent}/run_{run}/table_pred.pkl", "wb") as f:
            pickle.dump(table, f)

    ray.shutdown()

    ttest("pred", agent, memory, explainer_extras)

    pair_list = [(ext, expl) for ext in extras for expl in explainer_extras]
    plot_losses(pair_list, memory, agent, "pred_models")
