import argparse
import glob
import os
import time

import numpy as np
import supersuit as ss
from counterfactuals_shapley.wrappers import par_env_with_seed, pathify
from dotenv import load_dotenv
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

load_dotenv()


def train(
    env_fn,
    tune=False,
    steps: int = 2_000_000,
    seed=0,
    device="auto",
    lr=0.0003,
    gamma=0.99,
    la=0.95,
    callback=None,
    model=None,
    **env_kwargs,
):
    steps = int(steps)
    # Train a single model to play as each agent in an AEC environment
    env = env_fn.parallel_env(**env_kwargs)
    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    env = ss.black_death_v3(env)

    env.reset()

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")

    if not model:
        model = PPO(
            MlpPolicy,
            env,
            verbose=3,
            batch_size=64,
            learning_rate=lr,
            gamma=gamma,
            gae_lambda=la,
            n_epochs=30,
            device=device,
        )

    if not callback:
        model.learn(total_timesteps=steps)
    else:
        model.learn(total_timesteps=steps, callback=callback)

    if tune:
        save_path = f".{pathify(env)}/{os.environ['TUNING_PATH']}/{time.strftime('%Y%m%d-%H%M%S')}"
    else:
        save_path = f".{pathify(env)}/{os.environ['MODEL_PATH']}/{time.strftime('%Y%m%d-%H%M%S')}"

    model.save(save_path)
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()

    return save_path


def eval(
    env_fn,
    num_games: int = 100,
    render_mode=None,
    tune=False,
    seed=None,
    **env_kwargs,
):
    env = env_fn.parallel_env(render_mode=render_mode, **env_kwargs)
    num_agents = len(env_fn.env(**env_kwargs).possible_agents)
    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    if tune:
        try:
            latest_policy = max(
                glob.glob(f"{pathify(env)}/{os.environ['TUNING_PATH']}/*.zip"),
                key=os.path.getctime,
            )
            print(latest_policy)
        except ValueError:
            print("Policy not found.")
            exit(0)

    else:
        try:
            latest_policy = max(
                glob.glob(f"{pathify(env)}/{os.environ['RL_MODEL_PATH']}/*.zip"),
                key=os.path.getctime,
            )
            print(latest_policy)
        except ValueError:
            print("Policy not found.")
            exit(0)

    model = PPO.load(latest_policy)

    env = ss.black_death_v3(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    episode_rewards = []
    agent_reward = np.zeros(num_agents)
    for i in range(num_games):
        if not seed == None:
            env = par_env_with_seed(env, seed * i)
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _states = model.predict(obs)
            obs, rewards, dones, infos = env.step(action)
            agent_reward += rewards
            episode_reward += sum(rewards)
            done = all(dones)
        episode_rewards.append(episode_reward)

    env.close()
    print(agent_reward / num_games)
    return np.mean(episode_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument(
        "-e", "--env", type=str, help="Which environment to use", required=True
    )
    parser.add_argument(
        "-n",
        "--num_games",
        type=int,
        help="How many games to simulate",
        default=100,
    )
    parser.add_argument(
        "-r",
        "--render",
        type=str,
        help="How to render if any, default None",
        default=None,
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

    eval(
        env_fn,
        num_games=args.num_games,
        render_mode=args.render,
        **env_kwargs,
    )
