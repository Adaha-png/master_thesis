import torch
import torch.nn as nn
from ray.rllib.models.torch.recurrent_net import (
    RecurrentNetwork as TorchRecurrentNetwork,
)


class CustomLSTMWrapper(TorchRecurrentNetwork, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchRecurrentNetwork.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Two fully-connected layers, each 64 wide, using tanh activation.
        input_size = obs_space.shape[0]
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)

        # LSTM layer with input dim 64 and hidden size 64.
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True)

        # Output layer mapping the LSTM's hidden state to action logits.
        self.out = nn.Linear(64, num_outputs)

        # Save hidden size for state initialization.
        self.hidden_size = 64

    def forward(self, input_dict, state, seq_lens):
        # Process observations through two FC layers with tanh activations.
        x = input_dict["obs"]
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        # Add time dimension: assuming a single timestep per forward call.
        x = x.unsqueeze(1)

        # Unpack previous LSTM states (hidden and cell).
        h, c = state[0].unsqueeze(0), state[1].unsqueeze(0)

        # Process through the LSTM.
        lstm_out, (h_new, c_new) = self.lstm(x, (h, c))
        lstm_out = lstm_out.squeeze(1)

        # Compute logits from the LSTM output.
        logits = self.out(lstm_out)

        # Return the logits and the new state (squeezing out the batch dimension).
        return logits, [h_new.squeeze(0), c_new.squeeze(0)]

    def get_initial_state(self):
        # Return zero-filled hidden and cell states.
        return [torch.zeros(self.hidden_size), torch.zeros(self.hidden_size)]
