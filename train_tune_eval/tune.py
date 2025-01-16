import argparse
from functools import partial

import optuna
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3

from counterfactuals_shapley.wrappers import pathify

from .train_eval import eval, train


def optim(tuner, trials, jobs, env_fn, **env_kwargs):
    env = env_fn.parallel_env(**env_kwargs)
    study = optuna.create_study(
        study_name=f"{pathify(env)}/tuning.db",
        direction="maximize",
        load_if_exists=True,
        storage=f"sqlite:///{pathify(env)}/tuning.db",
    )
    print(jobs)

    study.optimize(
        tuner,
        n_trials=int(trials),
        n_jobs=int(jobs),
    )

    return study


def tuner(env_fn, trial, max_timesteps=500_000, **env_kwargs):
    la = trial.suggest_float("la", 0.9, 0.99)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    lr = trial.suggest_float("lr", 1e-5, 1, log=True)

    train(
        env_fn,
        steps=max_timesteps,
        seed=0,
        lr=lr,
        gamma=gamma,
        la=la,
        tune=True,
        **env_kwargs,
    )

    return eval(env_fn, num_games=100, render_mode=None, tune=True, **env_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning script")
    parser.add_argument(
        "-e", "--env", type=str, help="Which environment to use", required=True
    )
    parser.add_argument(
        "-t", "--timesteps", type=int, help="Max timesteps", default=500_000
    )
    env_fn = simple_spread_v3

    env_kwargs = dict(
        N=3,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
    )
    parser.add_argument("-n", "--n_trials", type=int, help="Max trials", default=200)

    parser.add_argument("-j", "--n_jobs", type=int, help="Simultaneous jobs", default=4)
    args = parser.parse_args()
    part_tuner = partial(tuner, max_timesteps=args.timesteps)
    optim(tuner, env_fn, **env_kwargs)
