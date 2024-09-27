import argparse
import glob
import os
import time

import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy


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
    # Train a single model to play as each agent in an AEC environment
    env = env_fn.parallel_env(**env_kwargs)

    # Add black death wrapper so the number of agents stays constant
    # MarkovVectorEnv does not support environments with varying numbers of active agents unless black_death is set to True
    env = ss.black_death_v3(env)

    env.reset(seed=seed)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=1, base_class="stable_baselines3")

    if not model:
        # Use a CNN policy if the observation space is visual
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
        model.save(
            f"tune_{env.unwrapped.metadata.get('name')}/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
        )
    else:
        model.save(
            f"train_{env.unwrapped.metadata.get('name')}/{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
        )
    print("Model has been saved.")
    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

    env.close()


def eval(
    env_fn,
    num_games: int = 100,
    render_mode=None,
    tune=False,
    **env_kwargs,
):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)

    print(
        f"\nStarting evaluation on {str(env.metadata['name'])} (num_games={num_games}, render_mode={render_mode})"
    )

    if tune:
        try:
            latest_policy = max(
                glob.glob(f"tune_{str(env.metadata['name'])}/*.zip"),
                key=os.path.getctime,
            )
            print(latest_policy)
        except ValueError:
            print("Policy not found.")
            exit(0)

    else:
        try:
            latest_policy = max(
                glob.glob(f"{str(env.metadata['name'])}/*.zip"),
                key=os.path.getctime,
            )
            print(latest_policy)
        except ValueError:
            print("Policy not found.")
            exit(0)

    model = PPO.load(latest_policy)

    rewards = {agent: 0 for agent in env.possible_agents}

    for i in range(num_games):
        env.reset(seed=i)
        env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]

            if termination or truncation:
                break
            else:
                if agent == env.possible_agents[0]:
                    act = env.action_space(agent).sample()
                else:
                    act = model.predict(obs, deterministic=True)[0]
            env.step(act)
    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent] / num_games for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
    return avg_reward


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
        num_games=100,
        **env_kwargs,
    )
