import torch.nn as nn
from skrl.models.torch import CategoricalMixin, Model


# define the model
class p_MLP(CategoricalMixin, Model):
    def __init__(
        self, observation_space, action_space, device, unnormalized_log_prob=True
    ):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class p_CNN(CategoricalMixin, Model):
    def __init__(
        self, observation_space, action_space, device, unnormalized_log_prob=True
    ):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 16),
            nn.Tanh(),
            nn.Linear(16, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, self.num_actions),
        )

    def compute(self, inputs, role):
        # permute (samples, width * height * channels) -> (samples, channels, width, height)
        return self.net(
            inputs["states"].view(-1, *self.observation_space.shape).permute(0, 3, 1, 2)
        ), {}
