import glob
import os
import pickle
import warnings

import ray
import supersuit as ss
from dotenv import load_dotenv
from pettingzoo.mpe import simple_spread_v3
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()


def env_creator(config=None):
    env_kwargs = dict(
        N=3,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
    )
    env = simple_spread_v3.parallel_env(**env_kwargs)
    # Add black death wrapper so the number of agents stays constant
    env = ss.black_death_v3(env)
    env.reset()
    return env


def run_train(lr=0.00001, gamma=0.99):
    ray.init()

    temp_env = env_creator()
    temp_env.reset()
    env_name = temp_env.metadata["name"]

    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

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
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
            },
            train_batch_size=4000,
            lr=lr,
            gamma=gamma,
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
    for i in range(50):  # Increase this number for longer training
        result = algo.train()
        print(
            f"Iteration {i}:\treward: {result['env_runners']['episode_reward_mean']:.2f}"
        )

    with open(f".{env_name}/{os.environ['RL_MODEL_PATH']}", "wb") as f:
        pickle.dump(algo, f)

    algo.stop()
    ray.shutdown()

    return algo


def run_inference(algo, num_episodes: int = 5):
    """
    Run inference in the environment for a certain number of episodes
    using a loaded PPO model. Prints out total reward per episode.
    """

    env = ParallelPettingZooEnv(env_creator(None))

    total_reward = 0.0
    for episode in range(num_episodes):
        obs = env.reset()
        done = {agent_id: False for agent_id in env.agents}

        while not all(done.values()):
            actions = {}
            # For each agent, get the current observation and compute an action
            for agent_id, agent_obs in obs.items():
                # By default, if you're using a single shared policy,
                # "policy_id" is usually "default_policy".
                # If you have a custom multi-agent setup, adjust accordingly.
                actions[agent_id] = algo.compute_single_action(
                    agent_obs, policy_id=agent_id.split("_")[]
                )

            # Step the environment with the multi-agent action dict
            obs, rewards, done, infos = env.step(actions)

            # Accumulate rewards across all agents
            total_reward += sum(rewards.values())

    return total_reward / num_episodes


if __name__ == "__main__":
    run_train()
