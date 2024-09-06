from __future__ import annotations

import optuna
from pettingzoo.butterfly import knights_archers_zombies_v10

from .train_eval import eval, train


def tuner(trial):
    env_fn = knights_archers_zombies_v10

    env_kwargs = dict(
        spawn_rate=20,
        num_archers=2,
        num_knights=2,
        max_zombies=10,
        max_arrows=10,
        max_cycles=100,
        vector_state=True,
    )
    la = trial.suggest_float("la", 0.9, 0.99)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    lr = trial.suggest_float("lr", 0.0001, 1, log=True)

    train(env_fn, steps=500_000, seed=0, lr=lr, gamma=gamma, la=la, **env_kwargs)

    return eval(env_fn, num_games=10, render_mode=None, **env_kwargs)


if __name__ == "__main__":
    study = optuna.create_study(
        study_name="tuning",
        direction="maximize",
        load_if_exists=True,
        storage="sqlite:///tuning.db",
    )
    study.optimize(tuner, n_trials=32, n_jobs=4)
