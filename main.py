import argparse
import glob
import os
import random

import numpy as np
import torch
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3

import counterfactuals_shapley
import train_tune_eval

def new_policy(env_fn, env_kwargs):
    pass

def run(env_fn, env_kwargs, always_execute_everything = False):
    timesteps_for_tuning = 500_000

    policy_path = None
    if not os.path.exists(f".{str(env.metadata['name'])}/rl_models"):
        os.mkdir(f".{str(env.metadata['name'])}/rl_models")

    elif not always_execute_everything:
        policy_path = max(
            glob.glob(f".{str(env.metadata['name'])}/rl_models/*.zip"),
            key=os.path.getctime,
        )


    if not policy_path:
        new_policy(env_fn, env_kwargs)

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
        run(env, policy_path)
