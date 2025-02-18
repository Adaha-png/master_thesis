import argparse
import os
import sys
from functools import partial

import optuna
from dotenv import load_dotenv
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rllib_train import env_creator, run_inference, run_train

load_dotenv()


def optim(tuner, trials, jobs, memory="none"):
    env = env_creator()
    study = optuna.create_study(
        study_name=f".{env.metadata['name']}/{memory}/{os.environ['RL_TUNING_PATH']}/tuning.db",
        direction="maximize",
        load_if_exists=True,
        storage=f"sqlite:///.{env.metadata['name']}/{memory}/tuning.db",
    )

    study.optimize(
        tuner,
        n_trials=int(trials),
        n_jobs=int(jobs),
    )

    return study


def tuner(trial, max_timesteps=100_000):
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    lr = trial.suggest_float("lr", 1e-5, 1, log=True)

    algo = run_train(
        max_timesteps=max_timesteps,
        seed=0,
        lr=lr,
        gamma=gamma,
        tuning=True,
    )
    return run_inference(algo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning script")
    # parser.add_argument("-e", "--env", type=str, help="Which environment to use")
    parser.add_argument(
        "-t", "--timesteps", type=int, help="Max timesteps", default=100_000
    )
    env_fn = simple_spread_v3

    env_kwargs = dict(
        N=3,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
    )

    parser.add_argument("-n", "--n_trials", type=int, help="Max trials", default=100)

    parser.add_argument("-j", "--n_jobs", type=int, help="Simultaneous jobs", default=1)
    args = parser.parse_args()
    part_tuner = partial(tuner, max_timesteps=args.timesteps)
    optim(tuner, args.n_trials, args.n_jobs)
