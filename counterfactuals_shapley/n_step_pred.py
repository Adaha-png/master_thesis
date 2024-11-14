import argparse
import glob
import multiprocessing
import os
import pickle
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from sim_steps import sim_steps
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from torch import nn


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
    for j, feature in enumerate(sorted_feature_names):
        ax.axhline(j, color="gray", linestyle="--", linewidth=0.5)

    # Add vertical line at x=0
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_ylabel("Feature")
    ax.set_title(f"SHAP values for {coordinate_name} coordinate across all instances")

    # Add a color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Feature value")

    plt.show()
    # # Save the plot
    # plt.savefig(f"{output_file}_{coordinate_name}_instances.pdf".replace(" ", "_"))
    # plt.close()


def pred(net, n, X):
    net.eval()
    if not type(X) == torch.Tensor:
        X = torch.Tensor(X)
    return net(X).detach().numpy()[:, n]


def simulate_cycle(env_name, env_kwargs, policy_path, steps_per_cycle, seed, agent):
    env_fn = simple_spread_v3 if env_name == "spread" else knights_archers_zombies_v10
    env = env_fn.parallel_env(**env_kwargs)
    policy = PPO.load(policy_path)

    seq = sim_steps(env, policy, num_steps=steps_per_cycle, seed=seed)
    X = seq[0]["observation"][agent]
    Y = seq[-1]["observation"][agent][2:4]  # Position of agent after 10 steps
    return X, Y


def add_action(X, model):
    new_X = [np.array([*obs, model.predict(obs)[0]]) for obs in X]
    with open(".prediction_data_action.pkl", "wb") as f:
        pickle.dump(new_X, f)


def one_hot_action(X):
    X = np.array(X)
    num = round(max(X[:, -1]) - min(X[:, -1]) + 1)
    new_X = np.array([[*obs[:-1], *np.eye(num)[round(obs[-1])]] for obs in X])
    return new_X


def future_sight(
    env_name,
    device,
    net,
    X,
    y,
    extras="none",
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    net = train_net(
        net,
        X_train,
        y_train,
        X_test,
        y_test,
        device,
        epochs=200,
        extras=extras,
    )

    os.makedirs(".pred_models", exist_ok=True)
    torch.save(net.state_dict(), f".pred_models/pred_model_{env_name}_{extras}.pt")
    return net


def get_future_data(
    env_name,
    env_kwargs,
    policy_path,
    agent=0,
    amount_cycles=10000,
    steps_per_cycle=10,
    seed=0,
):
    X = []
    Y = []

    multiprocessing.set_start_method("spawn", force=True)
    pool = multiprocessing.Pool()

    sim_func = partial(
        simulate_cycle, env_name, env_kwargs, policy_path, steps_per_cycle, agent=agent
    )
    results = pool.starmap(sim_func, [(seed + c,) for c in range(amount_cycles)])

    pool.close()
    pool.join()

    for result in results:
        x, y = result
        X.append(x)
        Y.append(y)

    return X, Y


def train_net(
    net,
    X_train,
    y_train,
    X_test,
    y_test,
    device,
    epochs=100,
    batch_size=64,
    extras="none",
):
    # Convert data to PyTorch tensors
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
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
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        net.eval()
        with torch.no_grad():
            test_outputs = net(X_test)
            test_loss = criterion(test_outputs, y_test).item()
        eval_loss.append(test_loss)
        net.train()
        scheduler.step(test_loss)
        if scheduler.get_last_lr() != last_lr:
            last_lr = scheduler.get_last_lr()
            print(f"New lr: {last_lr}")
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

    plt.plot(range(1, (1 + len(eval_loss)), 1), eval_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss on evaluation set during training")
    plt.savefig(f"tex/images/pred_model_{extras}.pdf")

    # Evaluate on test data
    net.eval()
    with torch.no_grad():
        test_outputs = net(X_test)
        test_loss = criterion(test_outputs, y_test).item()
        print(f"Test Loss: {test_loss}")

    return net


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

    net = nn.Sequential(
        nn.Linear(len(X[0]), 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, len(y[0])),
    ).to(device)

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

    explainer_x = shap.KernelExplainer(partial(pred, net, 0), shap.kmeans(X, 100))
    explainer_y = shap.KernelExplainer(partial(pred, net, 1), shap.kmeans(X, 100))

    shap_plot(
        X[:100],
        explainer_x,
        f"{args.env}_{extras}",
        feature_names,
        "x",
    )
