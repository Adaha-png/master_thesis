import argparse
from functools import partial

import optuna
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3

from train_eval import eval, train


def tuner(trial, max_timesteps=500_000, env=None):
    if env == "spread":
        env_fn = simple_spread_v3

        env_kwargs = dict(
            N=3,
            local_ratio=0.5,
            max_cycles=25,
            continuous_actions=False,
        )
    elif env == "kaz":
        env_fn = knights_archers_zombies_v10

        env_kwargs = dict(
            spawn_rate=20,
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
    parser.add_argument("-n", "--n_trials", type=int, help="Max trials", default=200)

    parser.add_argument("-j", "--n_jobs", type=int, help="Simultaneous jobs", default=4)
    args = parser.parse_args()
    part_tuner = partial(tuner, max_timesteps=args.timesteps, env=args.env)

    study = optuna.create_study(
        study_name=f"tuning_{args.env}",
        direction="maximize",
        load_if_exists=True,
        storage=f"sqlite:///tuning_{args.env}.db",
    )

    study.optimize(
        part_tuner,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
    )
