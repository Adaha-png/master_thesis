import glob
import os
import random
from functools import partial

import numpy as np
import optuna
import torch
from dotenv import load_dotenv

import counterfactuals_shapley.compare as compare
import train_tune_eval.rllib_train as rllib_train
import train_tune_eval.rllib_tune as tune
from train_tune_eval.rllib_train import env_creator

load_dotenv()


def train_new_policy(should_tune, timesteps, memory="no_memory"):
    if should_tune:
        tuner = partial(tune.tuner, max_timesteps=int(os.environ["TUNING_STEPS"]))
        study = tune.optim(tuner, os.environ["TRIALS"], os.environ["JOBS"])
    else:
        study = optuna.load_study(
            study_name=f".{env.metadata['name']}/{memory}/tuning.db",
            storage=f"sqlite:///.{env.metadata['name']}/{memory}/{os.environ['RL_TUNING_PATH']}/tuning.db",
        )

    trial = study.best_trial
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    lr = trial.suggest_float("lr", 1e-6, 1, log=True)

    print(f"Using learning rate: {lr:.3e}, discount factor: {gamma:.3f}")

    _, policy_path = rllib_train.run_train(
        max_timesteps=timesteps,
        lr=lr,
        gamma=gamma,
        tuning=False,
    )

    return policy_path


def get_policy(
    should_tune=False, new_policy=True, memory="no_memory", timesteps=2000000
):
    policy_path = None
    if not os.path.exists(
        f".{str(env.metadata['name'])}/{memory}/{os.environ['RL_TRAINING_PATH']}"
    ):
        os.makedirs(
            f".{str(env.metadata['name'])}/{memory}/{os.environ['RL_TRAINING_PATH']}"
        )
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
        policy_path = train_new_policy(should_tune, timesteps)

    return policy_path


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = env_creator()

    memory = "no_memory"
    policy_path = get_policy(
        should_tune=False, new_policy=True, timesteps=2000000, memory=memory
    )

    agent = env.possible_agents[0].split("_")[0]

    compare.run_compare(policy_path, agent, memory, env.feature_names, env.act_dict)
