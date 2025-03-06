import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class CustomLSTMModel(TorchModelV2, nn.Module):
    """
    Custom LSTM model for RLlib that returns the hidden (and cell) states
    as part of the extra action outputs.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})
        self.hidden_dim = custom_config.get("hidden_dim", 256)
        self.num_layers = custom_config.get("num_layers", 1)

        # Assume the observation is a flat vector.
        input_size = int(np.prod(obs_space.shape))
        self.fc_in = nn.Linear(input_size, self.hidden_dim)

        # Create an LSTM layer.
        # Note: RLlib expects recurrent state tensors to be of shape [B, hidden_dim].
        # LSTM, however, uses shape [num_layers, B, hidden_dim], so we perform
        # the appropriate reshaping.
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )

        # Final output layer.
        self.fc_out = nn.Linear(self.hidden_dim, num_outputs)

        # Store the most recent LSTM hidden states for external access.
        self._last_hidden = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Process observations.
        x = input_dict["obs"]
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        x = self.fc_in(x)

        # Add a time dimension if missing (assuming a single time step).
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Now shape: [B, 1, hidden_dim]

        B = x.shape[0]
        # Prepare initial LSTM states.
        # RLlib passes in state as a list of two tensors, each of shape [B, hidden_dim].
        # LSTM requires states of shape [num_layers, B, hidden_dim].
        if len(state) == 0:
            h = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
            c = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
        else:
            h = state[0].unsqueeze(0)  # Convert [B, hidden_dim] -> [1, B, hidden_dim]
            c = state[1].unsqueeze(0)

        # Run the LSTM.
        lstm_out, (h_n, c_n) = self.lstm(x, (h, c))

        # Save hidden states for external access.
        self._last_hidden = (h_n, c_n)

        # Update state for the next forward pass (squeeze out the num_layers dimension).
        next_state = [h_n.squeeze(0), c_n.squeeze(0)]

        # Use the output from the last time step.
        last_output = lstm_out[:, -1, :]
        output = self.fc_out(last_output)
        return output, next_state

    @override(TorchModelV2)
    def get_initial_state(self):
        # RLlib expects initial state to be a list of tensors.
        # Here, we initialize both the hidden and cell states as zeros for a batch size of 1.
        return [torch.zeros(self.hidden_dim), torch.zeros(self.hidden_dim)]

    def get_extra_action_out(self):
        extra_out = {}
        if self._last_hidden is not None:
            # Return both hidden (h) and cell (c) states.
            extra_out["lstm_hidden"] = {
                "h": self._last_hidden[0].detach(),
                "c": self._last_hidden[1].detach(),
            }
        return extra_out
