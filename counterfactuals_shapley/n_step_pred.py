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
import supersuit as ss
import torch
from captum.attr import IntegratedGradients
from captum_grads import create_baseline
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from shapley import shap_plot
from sim_steps import sim_steps
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from torch import nn
from tqdm import tqdm


def add_ig(X, ig, env, target, device, extras="none"):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).to(device=device, dtype=torch.float32)
    elif isinstance(X, list):
        X = torch.Tensor(X).to(device=device, dtype=torch.float32)

    env_name = env.metadata["name"]
    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")
    num_acts = env.action_space.n

    if extras == "one-hot":
        new_X = [
            np.array([*obs, *ig(obs[:-num_acts], target=target)[0]]) for obs in tqdm(X)
        ]
    elif extras == "action":
        new_X = [np.array([*obs, *ig(obs[:-1], target=target)[0]]) for obs in X]
    else:
        new_X = [np.array([*obs, *ig(obs, target=target)[0]]) for obs in X]

    with open(f".prediction_data_ig_{extras}_{env_name}_{target}.pkl", "wb") as f:
        pickle.dump(new_X, f)

    return new_X


def pred(net, n, device, X):
    net.eval()
    if not type(X) == torch.Tensor:
        X = torch.Tensor(X).to(device)
    return net(X).cpu().detach().numpy()[:, n]


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

<<<<<<< HEAD
    extras = "action"
=======
    extras = "one-hot"  # none, action or one-hot
    explainer_extras = "ig"  # none, ig or shap
    target = 0
>>>>>>> 9fb6934295091e6d4d8cb47bd826a5f4ab7357b6

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
        if (
            not os.path.exists(
                f".prediction_data_ig_{extras}_{env.metadata['name']}_{target}.pkl"
            )
            or True
        ):
            policy_net = nn.Sequential(
                *model.policy.mlp_extractor.policy_net,
                model.policy.action_net,
                nn.Softmax(),
            ).to(device)

            ig = IntegratedGradients(policy_net)
            if not os.path.exists(f".baseline_future_{env.metadata["name"]}.pt"):
                baseline = create_baseline(
                    env, policy_path, 0, device, steps_per_cycle=1, seed=seed
                )
                torch.save(baseline, f".baseline_future_{env.metadata["name"]}.pt")
            else:
                baseline = torch.load(
                    f".baseline_future_{env.metadata["name"]}.pt", map_location=device
                )

            ig_partial = partial(
                ig.attribute,
                baselines=baseline,
                method="gausslegendre",
                return_convergence_delta=False,
            )
            X = add_ig(X, ig_partial, env, target, device, extras=extras)
        else:
            with open(
                f".prediction_data_ig_{extras}_{env.metadata['name']}_{target}).pkl",
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

        X_test = torch.Tensor(np.array(X_test)).to(device)
        y_test = torch.Tensor(np.array(y_test)).to(device)
        test_outputs = net(X_test)
        test_loss = criterion(test_outputs, y_test).item()
        print(test_loss)

    coordinate_names = ["x", "y"]
    explainer = shap.KernelExplainer(
        partial(pred, net, target, device), shap.kmeans(X.to(device="cpu"), 100)
    )

    shap_plot(
        X[:50],
        explainer,
        f"{args.env}_{extras}",
        feature_names,
        coordinate_names[target],
    )
