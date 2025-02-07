import glob
import os
import pickle
import warnings

import imageio
import numpy as np
import optuna
import ray
import supersuit as ss
from dotenv import load_dotenv
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()


def simple_spread_env(config=None):
    env_kwargs = dict(
        N=3,
        local_ratio=0.2,
        max_cycles=25,
        continuous_actions=False,
        render_mode="rgb_array",
    )
    env = simple_spread_v3.parallel_env(**env_kwargs)
    # Add black death wrapper so the number of agents stays constant
    env = ss.flatten_v0(env)
    env = ss.black_death_v3(env)
    env.reset()
    return env


def kaz_env(config=None):
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

    env = env_fn.parallel_env(**env_kwargs)

    env = ss.flatten_v0(env)
    env = ss.black_death_v3(env)
    env.reset()

    return env


def env_creator(config=None):
    return simple_spread_env(config)


def run_train(
    lr=0.0003,
    gamma=0.99,
    max_timesteps=2_000_000,
    seed=0,
    tuning=False,
):
    ray.init()

    temp_env = env_creator()
    temp_env.reset()
    env_name = temp_env.metadata["name"]

    print(
        f"Starting training on {env_name} with lr: {lr:.3e} and discount factor: {gamma:.3f}"
    )

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    steps_per_iter = 20000

    config = (
        PPOConfig()
        .environment(
            env=env_name,
            disable_env_checking=True,
        )
        .framework("torch")
        .env_runners(num_env_runners=15)
        .training(
            model={
                "fcnet_hiddens": [32, 32],
                "fcnet_activation": "tanh",
                # "use_lstm": True,
                # "lstm_cell_size": 64,
                # "max_seq_len": 20,
                # "lstm_use_prev_action": True,
                # "lstm_use_prev_reward": True,
                # "use_attention": True,
                # "attention_dim": 64,
                # "attention_num_transformer_units": 2,
                # "attention_num_heads": 2,
                # "attention_memory_training": 30,  # Short-term memory
                # "attention_memory_inference": 30,
            },
            train_batch_size=steps_per_iter,
            minibatch_size=512,
            lr=lr,
            gamma=gamma,
            shuffle_batch_per_epoch=True,
        )
        .multi_agent(
            policies={
                agent_type.split("_")[0]: (
                    None,
                    temp_env.observation_space(agent_type),
                    temp_env.action_space(agent_type),
                    {},
                )
                for agent_type in temp_env.possible_agents
            },
            policy_mapping_fn=lambda agent_id, episode, **kwargs: agent_id.split("_")[
                0
            ],
        )
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    )

    temp_env.close()
    # 5. Build the PPO algorithm
    algo = config.build()

    # 6. Training Loop
    training_iters = max(max_timesteps // steps_per_iter, 1)

    max_reward_mean = -np.inf
    for i in range(training_iters):
        result = algo.train()
        print(
            f"Iteration {i+1}/{training_iters}:\treward: {result['env_runners']['episode_reward_mean']:.2f}"
        )
        if not tuning:
            if result["env_runners"]["episode_reward_mean"] > max_reward_mean:
                max_reward_mean = result["env_runners"]["episode_reward_mean"]
                algo.save(
                    checkpoint_dir=f".{env_name}/{os.environ['RL_TRAINING_PATH']}"
                )

    if tuning:
        algo.save(checkpoint_dir=f".{env_name}/{os.environ['RL_TUNING_PATH']}")

    algo.stop()
    ray.shutdown()

    return algo


def run_inference(algo, num_episodes: int = 100):
    """
    Run inference in the environment for a certain number of episodes
    using a loaded PPO model. Prints out total reward per episode.
    """
    env = env_creator()

    total_reward = 0.0
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = {agent_id: False for agent_id in env.agents}
        while not all(done.values()):
            actions = {}

            # For each agent, get the current observation and compute an action
            for agent_id, agent_obs in obs.items():
                actions[agent_id] = algo.compute_single_action(
                    agent_obs, policy_id=agent_id.split("_")[0]
                )

            # Step the environment with the multi-agent action dict
            obs, rewards, done, _, infos = env.step(actions)

            # Accumulate rewards across all agents
            total_reward += sum(rewards.values())
    return total_reward / num_episodes


def watch_single_episode():
    """
    Loads a trained policy from ./<env_name>/policy and plays a single episode
    in the corresponding PettingZoo environment, rendering each step.

    :param env_name: Name of the environment (also used as checkpoint path).
    """
    frames = []
    env = env_creator()
    env.reset()
    env_name = env.metadata["name"]

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    # 1) Initialize Ray
    ray.init(ignore_reinit_error=True)

    # 2) Load the trained policy from
    checkpoint_path = "file://" + os.path.abspath(f".{env_name}/policies")

    algo = PPO.from_checkpoint(checkpoint_path)

    observations, _ = env.reset()

    done = {agent: False for agent in env.agents}  # Track if each agent is done
    while not all(done.values()):
        actions = {}
        # Gather actions for each agent from the loaded policy
        for agent, obs in observations.items():
            action = algo.compute_single_action(obs, policy_id="agent")
            actions[agent] = action

        # Step the environment
        observations, _, done, _, _ = env.step(actions)

        # Render the environment's current state
        frame = env.render()
        frames.append(frame)

    # 5) Close the environment
    env.close()

    video_filename = f".{env_name}/episode.mp4"
    print(video_filename)
    imageio.mimsave(video_filename, frames, fps=10)
    print(f"Video saved to {video_filename}")

    # 6) Shutdown Ray if you no longer need it
    ray.shutdown()


if __name__ == "__main__":
    env = env_creator()
    #
    # study = optuna.load_study(
    #     study_name=f".{env.metadata['name']}/tuning.db",
    #     storage=f"sqlite:///.{env.metadata['name']}/{os.environ['RL_TUNING_PATH']}/tuning.db",
    # )
    #
    # trial = study.best_trial
    # env.close()
    # gamma = trial.suggest_float("gamma", 0.8, 0.999)
    # lr = trial.suggest_float("lr", 1e-5, 1, log=True)

    # lr = 2e-3
    # gamma = 0.99
    # run_train(gamma=gamma, lr=lr, max_timesteps=2_000_000)
    watch_single_episode()
