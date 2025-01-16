import argparse
import glob
import os
import random
from functools import partial

import numpy as np
import optuna
import torch
from dotenv import load_dotenv
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3

import counterfactuals_shapley.compare as compare
import train_tune_eval.train_eval as train_eval
import train_tune_eval.tune as tune

load_dotenv()


def train_new_policy(should_tune, env_fn, **env_kwargs):
    if should_tune:
        tuner = partial(
            tune.tuner, env_fn, max_timesteps=os.environ["TUNING_STEPS"], **env_kwargs
        )
        study = tune.optim(
            tuner, os.environ["TRIALS"], os.environ["JOBS"], env_fn, **env_kwargs
        )
    else:
        study = optuna.load_study(
            study_name=f"tuning_{args.env}",
            storage=f"sqlite:///tuning_{args.env}.db",
        )

    trial = study.best_trial
    la = trial.suggest_float("la", 0.9, 0.99)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    lr = trial.suggest_float("lr", 1e-6, 1, log=True)

    print(
        f"Using learning rate: {lr:.6f}, discount factor: {gamma:.3f}, TD parameter: {la:.3f}"
    )

    policy_path = train_eval.train(
        env_fn,
        steps=args.timesteps,
        lr=lr,
        gamma=gamma,
        la=la,
        **env_kwargs,
    )

    train_eval.eval(env_fn, **env_kwargs, seed=283943)

    return policy_path


def get_policy(env_fn, should_tune=False, new_policy=True, **env_kwargs):
    policy_path = None
    if not os.path.exists(
        f".{str(env.metadata['name'])}/{os.environ['RL_MODEL_PATH']}"
    ):
        os.makedirs(f".{str(env.metadata['name'])}/{os.environ['RL_MODEL_PATH']}")
    elif not new_policy:
        try:
            policy_path = max(
                glob.glob(f".{str(env.metadata['name'])}/*.zip"),
                key=os.path.getctime,
            )
            print(policy_path)
        except ValueError:
            print("Policy not found in " + f".{str(env.metadata['name'])}/*.zip")

    if not policy_path:
        policy_path = train_new_policy(should_tune, env_fn, **env_kwargs)

    return policy_path


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
        N = int(env_kwargs["N"])

        feature_base_names = ["vel x", "vel y", "pos x", "pos y"]

        landmark_feature_names = [f"landmark {i} x" for i in range(1, N + 1)] + [
            f"landmark {i} y" for i in range(1, N + 1)
        ]
        landmark_feature_names = [
            feature
            for i in range(1, N + 1)
            for feature in (f"landmark {i} x", f"landmark {i} y")
        ]

        agent_feature_names = [
            feature
            for i in range(2, N + 1)
            for feature in (f"agent {i} x", f"agent {i} y")
        ]

        comms_feature_names = [f"comms {i}" for i in range(1, 5)]

        feature_names = (
            feature_base_names
            + landmark_feature_names
            + agent_feature_names
            + comms_feature_names
        )
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

        n = (
            env_kwargs["num_archers"]
            + 2 * env_kwargs["num_knights"]
            + env_kwargs["max_arrows"]
            + env_kwargs["max_zombies"]
        )

        feature_names = [
            "dist self",
            "pos x self",
            "pos y self",
            "vel x self",
            "vel y self",
        ]
        for i in range(n - 1):
            # first entities are archers, then knights, then swords, then arrows then zombies.
            feature_names.extend(
                [
                    f"dist ent {i}",
                    "rel pos x ent {i}",
                    "rel pos y ent {i}",
                    "vel x ent {i}",
                    "vel y ent {i}",
                ]
            )
        act_dict = {
            0: "idle",
            1: "rotate clockwise",
            2: "rotate counter clockwise",
            3: "move forwards",
            4: "move backwards",
            5: "attack",
        }
    else:
        print("Invalid env entered")
        exit(0)

    env = env_fn.parallel_env(**env_kwargs)

    policy_path = get_policy(env_fn, should_tune=True, new_policy=True, **env_kwargs)
    compare.run_compare(policy_path, feature_names, act_dict)
