import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class CustomLSTMModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Flatten the observation.
        input_size = int(np.prod(obs_space.shape))

        # Two hidden fully connected layers of size 64.
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)

        # LSTM layer with input and hidden size = 64.
        self.lstm = nn.LSTM(
            input_size=64, hidden_size=64, num_layers=1, batch_first=True
        )

        # Output layers for policy logits and value function.
        self.fc_out = nn.Linear(64, num_outputs)
        self.value_branch = nn.Linear(64, 1)

        self.layers = [self.fc1, self.fc2, self.lstm, self.fc_out]
        # Placeholders for storing the last value output and LSTM hidden state.
        self._value_out = None
        self._last_hidden = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Get and flatten observations.
        x = input_dict["obs"]
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)

        # Two hidden layers with ReLU activation.
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))

        # Add a time dimension if missing (expected shape: [B, T, features]).
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # becomes [B, 1, 64]

        B = x.size(0)  # current batch size

        # Prepare LSTM state.
        if len(state) == 0:
            # If no state provided, initialize with zeros.
            h = torch.zeros(1, B, 64, device=x.device)
            c = torch.zeros(1, B, 64, device=x.device)
        else:
            h, c = state[0], state[1]
            # If state is 2D, unsqueeze to make it 3D.
            if h.ndim == 2:
                if h.shape[0] != B:
                    # Tile the state if its batch size is smaller than input batch.
                    tile_factor = B // h.shape[0]
                    h = h.repeat(tile_factor, 1)
                    c = c.repeat(tile_factor, 1)
                h = h.unsqueeze(0)  # becomes [1, B, 64]
                c = c.unsqueeze(0)
            # Otherwise, assume state is already 3D.
            else:
                if h.shape[1] != B:
                    tile_factor = B // h.shape[1]
                    h = h.repeat(1, tile_factor, 1)
                    c = c.repeat(1, tile_factor, 1)

        # Pass through the LSTM.
        lstm_out, (h_n, c_n) = self.lstm(x, (h, c))
        last_out = lstm_out[:, -1, :]  # take output from the last timestep
        self.h_n = h_n
        self.c_n = c_n
        # Prepare new state for next step (back to 2D: [B, hidden_size]).
        new_state = [h_n.squeeze(0), c_n.squeeze(0)]
        self._last_hidden = new_state

        # Compute policy logits and value function.
        logits = self.fc_out(last_out)
        self._value_out = self.value_branch(last_out)

        return logits, new_state

    @override(TorchModelV2)
    def get_initial_state(self):
        # Return a "single-sample" state; RLlib will tile it.
        return [torch.zeros(64), torch.zeros(64)]

    @override(TorchModelV2)
    def value_function(self):
        if self._value_out is None:
            raise ValueError("value_function() called before forward()")
        return self._value_out.squeeze(1)

    def get_extra_action_out(self):
        # Expose the latest LSTM hidden state.
        return {"lstm_hidden": self._last_hidden}
