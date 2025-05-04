import gc
import lzma
import os
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import KernelShap
from dotenv import load_dotenv
from ray.rllib.algorithms.ppo import PPO
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from train_tune_eval.rllib_train import env_creator

from .sim_steps import SimRunner
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
                        torch.argmax(net(obs[:n_feats].unsqueeze(0))).cpu().item(),
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
    algo: PPO,
    env_creator,
    agent_tag: str,
    *,
    memory: str = "no_memory",
    device: str | torch.device = "cpu",
    amount_cycles: int = 5_000,
    steps_per_cycle: int = 100,
    test: bool = False,
    seed: int = 0,
    finished: list[str] | None = None,
) -> list[str] | str:
    """Generate many short rollouts and save them compressed.

    Creates an internal **`SimRunner`** (one per call) so the caller only needs
    the PPO checkpoint.
    """

    finished = finished or []
    runner = SimRunner(algo, env_creator, memory=memory, device=device)
    env_name = runner.env.metadata["name"]

    n_agents = len([ag for ag in runner.env.possible_agents if agent_tag in ag])
    packs = 1 if test else int(np.ceil(20 / n_agents))

    paths: List[str] = []
    os.makedirs(f".{env_name}/{memory}", exist_ok=True)

    try:
        for idx in range(packs):
            fname = (
                "prediction_test_data.xz" if test else f"prediction_data_part{idx}.xz"
            )
            path = f".{env_name}/{memory}/{fname}"
            if path in finished:
                continue

            seeds = [seed + idx * amount_cycles + j for j in range(amount_cycles)]
            res = [
                runner.sim_steps(steps_per_cycle, seed=s)
                for s in tqdm(seeds, desc=f"pack {idx}")
            ]
            with lzma.open(path, "wb") as fh:
                pickle.dump(res, fh)
            paths.append(path)
            del res
            gc.collect()
    finally:
        runner.close()

    return paths[0] if test else paths


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

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=False,
    )

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
    lr_val = last_lr[0]
    net.train()
    eval_loss = []

    torch.set_num_threads(10)

    for epoch in range(epochs):
        total_loss = 0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            outputs = net(batch_x)

            if len(outputs.size()) == 2:
                outputs = outputs.squeeze()

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}, LR: {lr_val}                     ",
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
        lr_val = last_lr[0]
        if scheduler.get_last_lr() != last_lr:
            last_lr = scheduler.get_last_lr()

        if last_lr[0] < 5e-7:
            break
    print(
        f"Training finished with Loss: {total_loss}, LR: {last_lr[0]}                     "
    )

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

    os.makedirs(f"tex/images/{env.metadata['name']}/{memory}/{agent}", exist_ok=True)
    plt.savefig(
        f"tex/images/{env.metadata['name']}/{memory}/{agent}/{name_ider}_losses_plot.pgf",
        backend="pgf",
    )
