import gc
import lzma
import os
import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from captum.attr import KernelShap
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

from train_tune_eval.rllib_train import env_creator

from .sim_steps import sim_steps
from .wrappers import numpyfy

load_dotenv()


def get_new_obs(obs, extras, net, shap, num_acts, n_feats, baseline, memory):
    shap_obs = obs[:n_feats]

    shap_obs = shap_obs.unsqueeze(0)

    if extras == "one-hot":
        action_idx = torch.argmax(obs[-num_acts:]).item()
    elif extras == "action":
        action_idx = obs[-1].item()
    else:
        if memory == "lstm":
            net.set_hidden(obs)

        action_idx = int(torch.argmax(net(shap_obs)))

    action_idx = int(action_idx)

    shap_values = shap.attribute(
        shap_obs, baselines=baseline, target=action_idx
    ).squeeze()

    return torch.cat((obs, shap_values))


def add_shap(
    net,
    agent,
    run,
    memory,
    X,
    baseline,
    device,
    test=False,
    extras="none",
    path_ider="pred_data",
):
    # Ensure compatibility with multiple input types
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    elif isinstance(X, list):
        X = torch.tensor(X, device=device, dtype=torch.float32)

    env = env_creator()
    num_acts = env.action_space(agent + "_0").n
    n_feats = len(env.feature_names)
    env.close()

    X = torch.Tensor(X).to(device)
    net = net.to(device)
    baseline = baseline.to(device)
    with torch.no_grad():
        shap = KernelShap(net)
        new_X = [
            get_new_obs(
                obs, extras, net, shap, num_acts, n_feats, baseline, memory
            ).cpu()
            for obs in tqdm(X)
        ]

    new_X = numpyfy(new_X)

    if not test:
        with lzma.open(
            f".{env.metadata['name']}/{memory}/{agent}/run_{run}/{path_ider}/prediction_data_{extras}_shap.xz",
            "wb",
        ) as f:
            pickle.dump(new_X, f)
    else:
        with lzma.open(
            f".{env.metadata['name']}/{memory}/{agent}/run_{run}/{path_ider}/test_data_{extras}_shap.xz",
            "wb",
        ) as f:
            pickle.dump(new_X, f)

    return new_X


def add_ig(
    net,
    agent,
    run,
    memory,
    X,
    ig,
    device,
    extras="none",
    save=True,
    name_ider="pred_data",
):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    elif isinstance(X, list):
        X = torch.Tensor(X).to(device=device, dtype=torch.float32)

    env = env_creator()
    n_feats = len(env.feature_names)
    env.close()
    if memory == "lstm":
        with torch.no_grad():
            new_X = [
                numpyfy(
                    [
                        *obs,
                        *ig(
                            obs[:n_feats],
                            target=torch.argmax(net.set_hidden(obs)),
                        ).squeeze(),
                    ]
                )
                for obs in tqdm(X, desc="Adding ig")
            ]
    else:
        new_X = [
            numpyfy(
                [
                    *obs,
                    *ig(
                        obs[:n_feats], target=torch.argmax(net(obs[:n_feats]))
                    ).squeeze(),
                ]
            )
            for obs in tqdm(X, desc="Adding ig")
        ]

    if save:
        os.makedirs(
            f".{env.metadata['name']}/{memory}/{agent}/run_{run}/{name_ider}",
            exist_ok=True,
        )
        with lzma.open(
            f".{env.metadata['name']}/{memory}/{agent}/run_{run}/{name_ider}/prediction_data_{extras}_ig.xz",
            "wb",
        ) as f:
            pickle.dump(new_X, f)

    return new_X


def pred(net, n, device, X):
    net.eval()
    if not type(X) == torch.Tensor:
        X = torch.Tensor(X).to(device)
    return net(X).cpu().detach().numpy()[:, n]


def add_action(X, net, agent, run, memory, device, name_ider="pred_data", save=True):
    X = torch.Tensor(X).to(device)
    net = net.to(device)
    n_feats = len(env_creator().feature_names)
    with torch.no_grad():
        try:
            new_X = [
                numpyfy(
                    [
                        *obs.cpu(),
                        torch.argmax(net(obs.unsqueeze(0))).cpu().item(),
                    ]
                )
                for obs in tqdm(X, desc="Adding action")
            ]
        except RuntimeError as e:
            new_X = [
                numpyfy(
                    [
                        *obs.cpu(),
                        torch.argmax(net(obs[:-n_feats].unsqueeze(0))).cpu().item(),
                    ]
                )
                for obs in tqdm(X, desc="Adding action")
            ]

        if save:
            env = env_creator()
            os.makedirs(
                f".{env.metadata['name']}/{memory}/{agent}/run_{run}/{name_ider}",
                exist_ok=True,
            )
            with lzma.open(
                f".{env.metadata['name']}/{memory}/{agent}/run_{run}/{name_ider}/prediction_data_action.xz",
                "wb",
            ) as f:
                pickle.dump(new_X, f)
        return new_X


def one_hot_action(X):
    X = numpyfy(X)
    num = len(env_creator().act_dict.values())
    new_X = numpyfy([[*obs[:-1], *np.eye(num)[round(obs[-1])]] for obs in X])
    return new_X


def future_sight(
    agent,
    run,
    memory,
    device,
    net,
    X,
    y,
    epochs=300,
    extras="none",
    explainer_extras="none",
    criterion=None,
    name_ider="pred_models",
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    net = train_net(
        net,
        memory,
        agent,
        run,
        X_train,
        y_train,
        X_test,
        y_test,
        device,
        epochs=epochs,
        extras=extras,
        explainer_extras=explainer_extras,
        criterion=criterion,
        name_ider=name_ider,
    )

    env = env_creator()
    env_name = env.metadata["name"]
    env.close()

    os.makedirs(f".{env_name}/{memory}/{agent}/run_{run}/{name_ider}", exist_ok=True)
    torch.save(
        net.state_dict(),
        f".{env_name}/{memory}/{agent}/run_{run}/{name_ider}/pred_model_{extras}_{explainer_extras}.pt",
    )
    return net


def get_future_data(
    net,
    memory,
    agent,
    device,
    amount_cycles=10000,
    steps_per_cycle=100,
    test=False,
    seed=0,
    finished=[],
):
    if ray.is_initialized():
        ray.shutdown()

    env = env_creator()
    env_name = env.metadata["name"]

    n_agents = len([ag for ag in env.possible_agents if agent in ag])

    if test:
        training_packs = 1
    else:
        training_packs = int(np.ceil(10 / n_agents))

    paths = []
    with torch.no_grad():
        for i in range(0, training_packs):
            if test:
                path = f".{env_name}/{memory}/prediction_test_data.xz"
            else:
                path = f".{env_name}/{memory}/prediction_data_part{i}.xz"
            if path in finished:
                continue

            sim_part = partial(sim_steps, net, steps_per_cycle, memory, device)

            seed_values = [seed + i * (amount_cycles) + j for j in range(amount_cycles)]

            results = [sim_part(seed_value) for seed_value in tqdm(seed_values)]
            paths.append(path)
            with lzma.open(path, "wb") as f:
                pickle.dump(results, f)

            del results
            gc.collect()

    if test:
        return paths[0]

    return paths


def train_net(
    net,
    memory,
    agent,
    run,
    X_train,
    y_train,
    X_test,
    y_test,
    device,
    epochs=300,
    batch_size=64,
    extras="none",
    explainer_extras="none",
    criterion=None,
    name_ider="pred_models",
):
    # Convert data to PyTorch tensors
    if isinstance(criterion, nn.CrossEntropyLoss):
        X_train = torch.tensor(numpyfy(X_train), dtype=torch.float32).to(device)
        y_train = torch.tensor(numpyfy(y_train), dtype=torch.long).to(device)
        X_test = torch.tensor(numpyfy(X_test), dtype=torch.float32).to(device)
        y_test = torch.tensor(numpyfy(y_test), dtype=torch.long).to(device)
    else:
        X_train = torch.tensor(numpyfy(X_train), dtype=torch.float32).to(device)
        y_train = torch.tensor(numpyfy(y_train), dtype=torch.float32).to(device)
        X_test = torch.tensor(numpyfy(X_test), dtype=torch.float32).to(device)
        y_test = torch.tensor(numpyfy(y_test), dtype=torch.float32).to(device)

    # Define loss function and optimizer
    if not criterion:
        criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=0.3,
    )

    # Training loop
    last_lr = scheduler.get_last_lr()
    net.train()
    eval_loss = []

    for epoch in range(epochs):
        total_loss = 0
        permutation = torch.randperm(X_train.size()[0])

        for i in range(0, X_train.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i : i + batch_size]
            batch_x, batch_y = X_train[indices], y_train[indices]

            outputs = net(batch_x)

            if len(batch_y.shape) == 1:
                batch_y = batch_y.unsqueeze(1)

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}, LR: {last_lr[0]}                     ",
            end="\r",
        )

        net.eval()
        with torch.no_grad():
            test_outputs = net(X_test)
            if len(y_test.shape) == 1:
                y_test = y_test.unsqueeze(1)
            test_loss = criterion(test_outputs, y_test).item()
        eval_loss.append(test_loss)
        net.train()
        scheduler.step(test_loss)
        if scheduler.get_last_lr() != last_lr:
            last_lr = scheduler.get_last_lr()

        if last_lr[0] < 5e-7:
            break

    plt.clf()
    plt.plot(range(1, 1 + len(eval_loss), 1), eval_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    env = env_creator()

    os.makedirs(
        f".{env.metadata['name']}/{memory}/{agent}/run_{run}/{name_ider}", exist_ok=True
    )

    with open(
        f".{env.metadata['name']}/{memory}/{agent}/run_{run}/{name_ider}/{extras}_{explainer_extras}_train_eval_loss.pkl",
        "wb",
    ) as f:
        pickle.dump(eval_loss, f)

    env.close()

    net.eval()
    with torch.no_grad():
        test_outputs = net(X_test)
        if len(y_test.shape) == 1:
            y_test = y_test.unsqueeze(1)
        test_loss = criterion(test_outputs, y_test).item()
        print(f"Test Loss: {test_loss}")

    return net


def plot_losses(action_expl_pair_list, memory, agent, name_ider, n_runs=10):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )

    env = env_creator()
    all_mean_losses = []
    all_cis = []
    legends = []

    for pair in action_expl_pair_list:
        extras = pair[0]
        explainer_extras = pair[1]
        legend_id = ""
        if extras == "none":
            legend_id += "No action"
        elif extras == "action":
            legend_id += "Integer"
        elif extras == "one-hot":
            legend_id += "One-hot"
        legend_id += " and "
        if explainer_extras == "none":
            legend_id += "No explanation"
        elif explainer_extras == "ig":
            legend_id += "IG"
        elif explainer_extras == "shap":
            legend_id += "Shapley"

        legends.append(legend_id)

        losses_runs = []
        for run in range(n_runs):
            file_path = (
                f".{env.metadata['name']}/{memory}/{agent}/run_{run}/"
                f"{name_ider}/{extras}_{explainer_extras}_train_eval_loss.pkl"
            )
            with open(file_path, "rb") as f:
                eval_loss = pickle.load(f)
            losses_runs.append(eval_loss)

        max_len = max(len(lst) for lst in losses_runs)
        n_rows = len(losses_runs)

        result = np.empty((n_rows, max_len), dtype=np.float32)

        for i, lst in enumerate(losses_runs):
            L = len(lst)
            result[i, :] = lst[-1]
            result[i, :L] = lst

        losses_runs = np.array(result)
        mean_loss = np.mean(losses_runs, axis=0)
        std_loss = np.std(losses_runs, axis=0)

        ci = 1.96 * std_loss / np.sqrt(n_runs)
        all_mean_losses.append(mean_loss)
        all_cis.append(ci)

    max_len = max(len(lst) for lst in all_mean_losses)
    n_rows = len(all_mean_losses)

    result = np.empty((n_rows, max_len), dtype=np.float32)

    for i, lst in enumerate(all_mean_losses):
        L = len(lst)
        result[i, :] = lst[-1]
        result[i, :L] = lst

    all_mean_losses = np.array(result)

    result = np.empty((n_rows, max_len), dtype=np.float32)

    for i, lst in enumerate(all_cis):
        L = len(lst)
        result[i, :] = lst[-1]
        result[i, :L] = lst

    all_cis = np.array(result)

    env.close()

    plt.figure(figsize=(5.5, 4))

    epochs = range(1, len(all_mean_losses[0]) + 1)

    for mean_loss, ci, label in zip(all_mean_losses, all_cis, legends):
        plt.plot(epochs, mean_loss, label=label)
        plt.fill_between(epochs, mean_loss - ci, mean_loss + ci, alpha=0.3)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"tex/images/{env.metadata['name']}/{memory}/{agent}/{name_ider}_losses_plot.pgf",
        backend="pgf",
    )
