import argparse
import glob
import os
import pprint
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import shap
import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from stable_baselines3 import PPO
from tqdm import tqdm

from custom_env_utils import par_env_with_seed
from sim_steps import sim_steps


def surrogate_shap(env, policy, seed=None):
    X, y = get_data(env, policy)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    model = RandomForestClassifier(n_estimators=100, random_state=seed)
    model.fit(X_train, y_train)

    # Ensure your model achieves a reasonable performance
    print(f"Model Accuracy: {model.score(X_test, y_test):.2f}")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test,
        show=False,
    )  # Use the SHAP values for class 0
    plt.savefig(f"tex/images/shap_plot_surrogate.pdf", bbox_inches="tight")
    plt.close()


def pred(model, obs):
    return model.predict(obs)[0]


def kernel_shap(env, policy):
    X, _ = get_data(env, policy, total_steps=10000)
    model = PPO.load(policy)
    explainer = shap.KernelExplainer(partial(pred, model), shap.kmeans(X, 10))

    shap_values = explainer.shap_values(X)
    plt.figure()
    shap.summary_plot(
        shap_values,
        X,
        feature_names=[
            "self vel x",
            "self vel y",
            "self pos x",
            "self pos y",
            "landmark1 pos x",
            "landmark1 pos y",
            "landmark2 pos x",
            "landmark2 pos y",
            "landmark3 pos x",
            "landmark3 pos y",
            "agent1 pos x",
            "agent1 pos y",
            "agent2 pos x",
            "agent2 pos y",
            "comm",
            "comm",
            "comm",
            "comm",
        ],
        show=False,
    )
    plt.savefig(f"tex/images/shap_plot_kernel.pdf", bbox_inches="tight")
    plt.close()


def get_data(env, policy, total_steps=10000, steps_per_cycle=250, agent=1):
    observations = []
    actions = []
    num_cycles = total_steps // steps_per_cycle
    for _ in tqdm(range(num_cycles)):
        step_results = sim_steps(env, policy, num_steps=steps_per_cycle)
        for entry in step_results:
            obs = entry.get("observation")
            act = entry.get("action")
            if obs is not None and act is not None:
                actions.append(act[agent])
                observations.append(obs[agent])
    return np.array(observations), np.array(actions)


if __name__ == "__main__":
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
        "-s",
        "--steps",
        type=int,
        help="Steps to simulate",
        default=10,
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
    else:
        print("Invalid env entered")
        exit(0)

    env = env_fn.parallel_env(render_mode=args.render, **env_kwargs)

    env = par_env_with_seed(env, seed)

    env = ss.black_death_v3(env)
    try:
        latest_policy = max(
            glob.glob(str(env.metadata["name"]) + "/*.zip"),
            key=os.path.getctime,
        )
        print(latest_policy)
    except ValueError:
        print("Policy not found.")
        exit(0)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(kernel_shap(env, latest_policy))
