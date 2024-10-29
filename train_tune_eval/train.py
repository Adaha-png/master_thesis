import argparse

import optuna
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3.common.callbacks import CheckpointCallback

from train_eval import eval, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "-e", "--env", type=str, help="Which environment to use", required=True
    )
    parser.add_argument(
        "-t", "--timesteps", type=int, help="Max timesteps", default=500_000
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

    study = optuna.load_study(
        study_name=f"tuning_{args.env}",
        storage=f"sqlite:///tuning_{args.env}.db",
    )
    trial = study.best_trial

    la = trial.suggest_float("la", 0.9, 0.99)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    lr = trial.suggest_float("lr", 1e-5, 1, log=True)
    print(
        f"Using learning rate: {lr:.6f}, discount factor: {gamma:.3f}, TD parameter: {la:.3f}"
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=int(1e3),
        save_path=f"{str(env_fn.env(**env_kwargs).metadata['name'])}/",
    )

    train(
        env_fn,
        steps=args.timesteps,
        lr=lr,
        gamma=gamma,
        la=la,
        callback=checkpoint_callback,
        **env_kwargs,
    )
