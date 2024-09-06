import optuna
from pettingzoo.butterfly import knights_archers_zombies_v10
from stable_baselines3.common.callbacks import CheckpointCallback

from train_eval import eval, train

if __name__ == "__main__":
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
    study = optuna.load_study(
        study_name="tuning",
        storage="sqlite:///tuning.db",
    )
    trial = study.best_trial

    la = trial.suggest_float("la", 0.9, 0.99)
    gamma = trial.suggest_float("gamma", 0.8, 0.999)
    lr = trial.suggest_float("lr", 0.0001, 1, log=True)
    print(
        f"Using learning rate: {lr:.6f}, discount factor: {gamma:.3f}, TD parameter: {la:.3f}"
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=1e4, save_path="./model_checkpoints/"
    )
    train(
        env_fn,
        lr=lr,
        gamma=gamma,
        la=la,
        callback=checkpoint_callback,
        **env_kwargs,
    )
