import numpy as np
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override


class GTrXLTransformerLayer(nn.Module):
    """A simple transformer encoder layer that returns attention weights."""

    def __init__(self, d_model, num_heads):
        super(GTrXLTransformerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.ReLU(), nn.Linear(d_model * 4, d_model)
        )
        self.layer_norm_ff = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: [B, T, d_model]
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True)
        x = self.layer_norm(x + attn_out)
        ff_out = self.ff(x)
        out = self.layer_norm_ff(x + ff_out)
        return out, attn_weights


class CustomGTrXLModel(TorchModelV2, nn.Module):
    """
    Custom GTrXL model for RLlib that mimics the behavior of the built-in
    use_attention option in PPO, while also exposing the attention weights.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Whether to expose attention weights.
        self.use_attention = model_config.get("use_attention", False)
        custom_config = model_config.get("custom_model_config", {})
        self.hidden_dim = custom_config.get("hidden_dim", 128)
        self.num_heads = custom_config.get("num_heads", 4)

        # Assuming the observation is a flat vector.
        input_size = int(np.prod(obs_space.shape))
        self.fc_in = nn.Linear(input_size, self.hidden_dim)

        # Create one transformer layer.
        self.transformer = GTrXLTransformerLayer(self.hidden_dim, self.num_heads)

        # Gating mechanism (as in GTrXL, a sigmoid gate on the transformer output).
        self.gate = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Final output layer.
        self.fc_out = nn.Linear(self.hidden_dim, num_outputs)

        # To store the attention weights for external access.
        self._last_attention = None

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Get the observations.
        x = input_dict["obs"]
        # If the observations are not already flat, flatten them.
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)

        # Project observations into the hidden space.
        x = self.fc_in(x)

        # Add a time dimension if missing (assume sequence length = 1).
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Now shape [B, 1, hidden_dim]

        # Pass through the transformer layer.
        transformer_out, attn_weights = self.transformer(x)

        # Apply gating.
        gate_val = torch.sigmoid(self.gate(transformer_out))
        gated_out = gate_val * transformer_out

        # Save attention weights if use_attention is enabled.
        if self.use_attention:
            self._last_attention = attn_weights

        # For prediction, use the output of the last time step.
        last_output = gated_out[:, -1, :]
        output = self.fc_out(last_output)
        return output, state

    @override(TorchModelV2)
    def get_initial_state(self):
        # This model does not use a recurrent state.
        return []

    def get_extra_action_out(self):
        extra_out = {}
        if self.use_attention:
            extra_out["attention_weights"] = self._last_attention
        return extra_out
