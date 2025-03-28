import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients

from train_tune_eval.rllib_train import env_creator

from .wrappers import numpyfy


def ig_extract(net, obs, action, agent, feature_names, act_dict, device):
    ig = IntegratedGradients(net)
    env = env_creator()
    if not os.path.exists(".baseline.pt"):
        baseline = create_baseline(net, agent, device, steps_per_cycle=100)
        torch.save(baseline, ".baseline.pt")
    else:
        baseline = torch.load(
            f".baseline_future_{env.metadata['name']}.pt", map_location=device
        )

    obs.to(device)

    attributions, approximation_error = ig.attribute(
        obs,
        baselines=baseline,
        target=action,
        method="gausslegendre",
        return_convergence_delta=True,
    )

    print(f"{attributions=}")
    print(f"{approximation_error=}")
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.squeeze().detach().numpy()
    sorted_indices = np.argsort(np.abs(attributions))
    attributions = attributions[sorted_indices]
    feature_names = numpyfy(feature_names)[sorted_indices]

    print(f"Action: {act_dict[action]}, Confidence:{net.forward(obs)[0, action]}")

    # Create a new figure and axis
    _, ax = plt.subplots(figsize=(8, 12))

    plt.rcParams.update(
        {
            "font.family": "serif",
            # Use LaTeX default serif font.
            "font.serif": [],
            "pgf.texsystem": "pdflatex",
        }
    )
    # Stem plot
    ax.scatter(attributions, feature_names, s=6)

    # Add horizontal lines for each feature
    for n in feature_names:
        ax.axhline(n, color="gray", linestyle="--", linewidth=0.5)

    # Add vertical line at x=0
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)
    # Set labels and title
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Integrated gradients method")

    # Show the plot
    plt.savefig(
        f"tex/images/intgrad_{act_dict[action]}.pgf".replace(" ", "_"), backend="pgf"
    )


def create_baseline(X, n_feats):
    X = numpyfy(X)

    baseline = torch.mean(torch.tensor(X[:, :n_feats]), dim=0).unsqueeze(0)

    return baseline.to(dtype=torch.float32)
