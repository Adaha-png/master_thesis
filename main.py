import os
import random
from functools import partial

import numpy as np
import optuna
import ray
import torch
from dotenv import load_dotenv
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

import counterfactuals_shapley.compare as compare
import counterfactuals_shapley.crit_state_pred as crit_state_pred
import train_tune_eval.rllib_train as rllib_train
import train_tune_eval.rllib_tune as tune
from counterfactuals_shapley.compare import get_torch_from_algo, make_plots
from train_tune_eval.rllib_train import env_creator

load_dotenv()


def train_new_policy(should_tune, timesteps, memory="no_memory"):
    if should_tune:
        if os.path.exists(f".{env.metadata['name']}/{memory}/tuning.db"):
            remove = "y"
            remove = input(
                "Tuning db for this environment already exists, do you want to delete it and start over? [y/N]:"
            )
            if remove == "y":
                os.remove(f".{env.metadata['name']}/{memory}/tuning.db")

        tuner = partial(tune.tuner, max_timesteps=int(os.environ["TUNING_STEPS"]))
        study = tune.optim(tuner, os.environ["TRIALS"], os.environ["JOBS"])
    else:
        study = optuna.load_study(
            study_name=f".{env.metadata['name']}/{memory}/tuning.db",
            storage=f"sqlite:///.{env.metadata['name']}/{memory}/tuning.db",
        )

    trial = study.best_trial
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    lr = trial.suggest_float("lr", 1e-6, 1e-2, log=True)

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
        if os.path.exists(
            f".{str(env.metadata['name'])}/{memory}/{os.environ['RL_TRAINING_PATH']}"
        ):
            policy_path = f".{str(env.metadata['name'])}/{memory}/{os.environ['RL_TRAINING_PATH']}"

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
    device = torch.device("cpu")
    env = env_creator()
    memory = "no_memory"
    policy_path = get_policy(
        should_tune=False, new_policy=False, timesteps=2000000, memory=memory
    )

    agent = env.possible_agents[0].split("_")[0]

    crit_state_pred.crit_compare(agent, memory, env.feature_names, env.act_dict)

    compare.run_compare(agent, memory, env.feature_names, env.act_dict, device)

    for memory in ["lstm", "attention"]:
        policy_path = get_policy(
            should_tune=True, new_policy=True, timesteps=2000000, memory=memory
        )
        ray.init(ignore_reinit_error=True)
        env_name = env_creator().metadata["name"]

        register_env(
            env_name, lambda config: ParallelPettingZooEnv(env_creator(config))
        )
        policy_path = "file://" + os.path.abspath(f".{env_name}/{memory}/policies")

        algo = PPO.from_checkpoint(policy_path)

        net = get_torch_from_algo(algo, agent, memory)

        ray.shutdown()

        crit_state_pred.compute(
            net,
            agent,
            env.feature_names,
            env.act_dict,
            extras="one-hot",
            explainer_extras="shapley",
            memory=memory,
            device=device,
        )
        compare.compute(
            net,
            agent,
            env.feature_names,
            env.act_dict,
            extras="one-hot",
            explainer_extras="shap",
            memory=memory,
        )
