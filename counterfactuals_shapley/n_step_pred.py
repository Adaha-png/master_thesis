import argparse
import copy
import glob
import os
import pickle
import random

import numpy as np
import torch
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from sim_steps import sim_steps
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from torch import nn
from tqdm import tqdm


def future_sight(env, policy, device, n=10):
    model = PPO.load(policy)

    if not os.path.exists(".prediction_data.pkl"):
        X, y = get_future_data(env, model, steps_per_cycle=n, seed=921)
        with open(".prediction_data.pkl", "wb") as f:
            pickle.dump((X, y), f)
    else:
        with open(".prediction_data.pkl", "rb") as f:
            X, y = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    net = nn.Sequential(
        *copy.deepcopy(model.policy.mlp_extractor.policy_net),
        nn.Linear(64, len(y_train[0])),
    ).to(device)

    train_net(net, X_train, y_train, X_test, y_test, device)


def get_future_data(
    env, policy, agent=0, amount_cycles=10000, steps_per_cycle=10, seed=0
):
    X = []
    Y = []

    for c in range(amount_cycles):
        seq = sim_steps(env, policy, num_steps=steps_per_cycle, seed=seed + c)
        X.append(seq[0]["observation"][agent])
        Y.append(seq[-1]["observation"][agent][2:4])  # Position of agent after 10 steps

    return X, Y


def train_net(net, X_train, y_train, X_test, y_test, device, epochs=50, batch_size=32):
    # Convert data to PyTorch tensors
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32).to(device)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32).to(device)
    X_test = torch.tensor(np.array(X_test), dtype=torch.float32).to(device)
    y_test = torch.tensor(np.array(y_test), dtype=torch.float32).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)

    # Training loop
    net.train()
    for epoch in tqdm(range(epochs)):
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

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss}")

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
        latest_policy = max(
            glob.glob(f".{str(env.metadata['name'])}/*.zip"),
            key=os.path.getctime,
        )
        print(latest_policy)
    except ValueError:
        print("Policy not found in " + f".{str(env.metadata['name'])}/*.zip")
        exit(0)

    future_sight(env, latest_policy, device)
