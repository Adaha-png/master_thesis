import gc
import lzma
import os
import pickle
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import tqdm

from train_tune_eval.rllib_train import env_creator

from .sim_steps import sim_steps
from .wrappers import numpyfy

load_dotenv()


def add_shap(
    net,
    agent,
    memory,
    X,
    expl,
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
    env.close()

    # Function to get the SHAP values and construct new observation
    def get_new_obs(obs, extras_type, net):
        obs_np = obs.cpu().numpy()
        if extras_type == "one-hot":
            action_idx = torch.argmax(obs[-num_acts:]).item()
            shap_values = expl[action_idx].shap_values(obs_np[:-num_acts])
        elif extras_type == "action":
            action_idx = obs[-1].item()
            shap_values = expl[int(action_idx)].shap_values(obs_np[:-1])
        else:
            action_idx = int(
                np.argmax(
                    net.forward(
                        obs,
                    )
                )
            )
            shap_values = expl[action_idx].shap_values(obs_np)

        return np.concatenate((obs_np, shap_values))

    new_X = [get_new_obs(obs, extras, net) for obs in tqdm(X, desc="Adding shapley")]
    if not test:
        with lzma.open(
            f"{env.metadata['name']}/{agent}/{memory}/{path_ider}/prediction_data_shap_{extras}.xz",
            "wb",
        ) as f:
            pickle.dump(new_X, f)
    else:
        with lzma.open(
            f"{env.metadata['name']}/{agent}/{memory}/{path_ider}/test_data_shap_{extras}.xz",
            "wb",
        ) as f:
            pickle.dump(new_X, f)

    return new_X


def add_ig(net, agent, memory, X, ig, device, extras="none", save=True):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    elif isinstance(X, list):
        X = torch.Tensor(X).to(device=device, dtype=torch.float32)

    env = env_creator()
    num_acts = env.action_space(agent + "_0").n
    env.close()
    if extras == "one-hot":
        new_X = [
            numpyfy(
                [
                    *obs.cpu(),
                    *ig(obs[:-num_acts], target=torch.argmax(obs[-num_acts:]))[0].cpu(),
                ]
            )
            for obs in tqdm(X, desc="Adding ig")
        ]
    elif extras == "action":
        new_X = [
            numpyfy([*obs.cpu(), *ig(obs[:-1], target=int(obs[-1]))[0].cpu()])
            for obs in tqdm(X, desc="Adding ig")
        ]
    else:
        with torch.no_grad():
            new_X = [
                numpyfy(
                    [
                        *obs.cpu(),
                        *ig(
                            obs,
                            target=torch.argmax(net(obs)),
                        ).squeeze(),
                    ]
                )
                for obs in tqdm(X, desc="Adding ig")
            ]
    if save:
        os.makedirs(
            f".{env.metadata['name']}/{memory}/{agent}/pred_data",
            exist_ok=True,
        )
        with lzma.open(
            f".{env.metadata['name']}/{memory}/{agent}/pred_data/prediction_data_ig_{extras}.xz",
            "wb",
        ) as f:
            pickle.dump(new_X, f)

    return new_X


def pred(net, n, device, X):
    net.eval()
    if not type(X) == torch.Tensor:
        X = torch.Tensor(X).to(device)
    return net(X).cpu().detach().numpy()[:, n]


def add_action(X, net, agent, memory, save=True):
    with torch.no_grad():
        new_X = [
            numpyfy(
                [
                    *obs,
                    np.argmax(net.forward(obs)),
                ]
            )
            for obs in X
        ]
        if save:
            env = env_creator()
            with lzma.open(
                f".{env.metadata['name']}/{agent}/{memory}/pred_data/prediction_data_action.xz",
                "wb",
            ) as f:
                pickle.dump(new_X, f)
        return new_X


def one_hot_action(X):
    X = numpyfy(X)
    num = round(max(X[:, -1]) - min(X[:, -1]) + 1)
    new_X = numpyfy([[*obs[:-1], *np.eye(num)[round(obs[-1])]] for obs in X])
    return new_X


def future_sight(
    agent,
    memory,
    device,
    net,
    X,
    y,
    epochs=200,
    extras="none",
    explainer_extras="none",
    criterion=None,
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    net = train_net(
        net,
        memory,
        agent,
        X_train,
        y_train,
        X_test,
        y_test,
        device,
        epochs=epochs,
        extras=extras,
        explainer_extras=explainer_extras,
        criterion=criterion,
    )

    env = env_creator()
    env_name = env.metadata["name"]
    env.close()

    os.makedirs(f".{env_name}/{memory}/{agent}/pred_models", exist_ok=True)
    torch.save(
        net.state_dict(),
        f".{env_name}/{memory}/{agent}/pred_models/pred_model_{extras}_{explainer_extras}.pt",
    )
    return net


def get_future_data(
    net,
    memory,
    amount_cycles=100000,
    steps_per_cycle=100,
    test=False,
    seed=0,
    finished=[],
):
    if ray.is_initialized():
        ray.shutdown()

    if test:
        training_packs = 1
    else:
        training_packs = 10
    env_name = env_creator().metadata["name"]
    paths = []
    with torch.no_grad():
        for i in range(0, training_packs):
            if test:
                path = f".{env_name}/{memory}/prediction_test_data.xz"
            else:
                path = f".{env_name}/{memory}/prediction_data_part{i}.xz"
            if path in finished:
                continue
            sim_part = partial(sim_steps, net, steps_per_cycle)
            # Compute the list of seed values for this training pack.
            seed_values = [
                seed + i * (amount_cycles // training_packs) + j
                for j in range(amount_cycles // training_packs)
            ]
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
    X_train,
    y_train,
    X_test,
    y_test,
    device,
    epochs=100,
    batch_size=64,
    extras="none",
    explainer_extras="none",
    criterion=None,
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
        optimizer, patience=5, threshold=0.00001
    )

    # Training loop
    last_lr = scheduler.get_last_lr()
    print(f"lr = {last_lr}")
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

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

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
            print(f"New lr: {last_lr}")

    plt.plot(range(1, (1 + len(eval_loss)), 1), eval_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss on evaluation set during training")
    env = env_creator()

    os.makedirs(f"tex/images/{env.metadata['name']}/{memory}/{agent}", exist_ok=True)
    plt.savefig(
        f"tex/images/{env.metadata['name']}/{memory}/{agent}/pred_model_{extras}_{explainer_extras}.pgf"
    )

    # Evaluate on test data
    net.eval()
    with torch.no_grad():
        test_outputs = net(X_test)
        if len(y_test.shape) == 1:
            y_test = y_test.unsqueeze(1)
        print(y_test.shape)
        test_loss = criterion(test_outputs, y_test).item()
        print(f"Test Loss: {test_loss}")

    return net
