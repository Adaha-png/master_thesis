import os

import ray
import supersuit as ss
from pettingzoo.mpe import simple_spread_v3
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.models.configs import ActorCriticEncoderConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn


def env_creator(config):
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


def train(
    steps: int = 2_000_000,
    seed=0,
    device="auto",
    lr=0.0003,
    gamma=0.99,
    la=0.95,
):
    ray.init()
    env = env_creator(None)
    env_name = env.metadata["name"]
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))
    # ModelCatalog.register_custom_model("MLPModel", MLPModel)

    encoder_config = ActorCriticEncoderConfig()
    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .env_runners(num_env_runners=4, rollout_fragment_length=128)
        .training(
            model={
                "encoder_config": encoder_config,
            },
            train_batch_size=512,
            lr=lr,
            gamma=gamma,
            lambda_=la,
            use_gae=True,
            clip_param=0.4,
            grad_clip=None,
            entropy_coeff=0.1,
            vf_loss_coeff=0.25,
            num_epochs=10,
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
    )

    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": steps if not os.environ.get("CI") else 50000},
        checkpoint_freq=10,
        storage_path="~/ray_results/" + env_name,
        config=config.to_dict(),
    )


if __name__ == "__main__":
    train()
