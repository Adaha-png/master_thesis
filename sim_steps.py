import argparse
import glob
import os
import pprint

import supersuit as ss
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy, MlpPolicy


def sim_steps(
    env,
    policy,
    num_steps=10,
    seed=54,
):
    model = PPO.load(policy)

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 1, num_cpus=1, base_class="stable_baselines3")

    obs = env.reset()
    rollout = []
    for step in range(num_steps):
        step_dict = {
            "observation": obs,
        }
        act = model.predict(obs, deterministic=True)[0]
        obs, reward, termination, info = env.step(act)

        step_dict.update(
            {
                "reward": reward,
                "action": act,
            }
        )

        rollout.append(step_dict)

        if termination.all():
            break

    env.close()
    return rollout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation")
    parser.add_argument(
        "-e", "--env", type=str, help="Which environment to use", default="spread"
    )
    parser.add_argument("-s", "--steps", type=int, help="Steps to simulate", default=10)

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

    env = env_fn.parallel_env(**env_kwargs)

    try:
        latest_policy = max(
            glob.glob(f"{str(env.metadata['name'])}/*.zip"),
            key=os.path.getctime,
        )
        print(latest_policy)
    except ValueError:
        print("Policy not found.")
        exit(0)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(sim_steps(env, latest_policy))
