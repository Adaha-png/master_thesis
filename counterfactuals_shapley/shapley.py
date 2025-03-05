import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from tqdm import tqdm

from counterfactuals_shapley.sim_steps import sim_steps
from counterfactuals_shapley.wrappers import numpyfy
from train_tune_eval.rllib_train import env_creator


def pred(net, act, device, obs):
    obs = torch.tensor(obs).unsqueeze(0).to(device, dtype=torch.float32)
    vals = net(obs)

    if act == None:
        vals = vals.unsqueeze(0).cpu().detach().numpy()
    else:
        vals = vals.cpu().detach().numpy()[0, :, act]
    return vals


def kernel_explainer(net, X, target, device):
    explainer = shap.KernelExplainer(
        partial(pred, net, target, device), shap.kmeans(X, 100)
    )
    return explainer


def shap_plot(agent, memory, X, explainer, feature_names, target):
    env = env_creator()
    if os.path.exists(
        f"tex/images/{env.metadata['name']}/{memory}/{agent}/{target}_shap.pdf"
    ):
        print("Plot already created, delete to make a new one, skipping..")
        return
    # Compute SHAP values for the given dataset X
    shap_values = explainer.shap_values(X)

    # Handling the case where SHAP values contains multiple outputs
    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)

    assert shap_values.shape[1] == len(
        feature_names
    ), "Mismatch between SHAP values and feature names dimensions."

    # Compute mean absolute SHAP values across all instances
    mean_shap_values = np.mean(np.abs(shap_values), axis=0)

    # Get the top 20 features
    top_indices = np.argsort(mean_shap_values)[-20:]
    top_feature_names = np.array(feature_names)[top_indices]

    # Extract SHAP values for the top features
    top_shap_values = shap_values[:, top_indices]

    # Flatten SHAP values and corresponding feature values for coloring
    flattened_shap_values = top_shap_values.flatten()
    repeated_feature_names = np.tile(top_feature_names, X.shape[0])
    feature_values = X[:, top_indices].flatten()

    # Create color map
    norm = plt.Normalize(np.min(feature_values), np.max(feature_values))
    colors = plt.cm.viridis(norm(feature_values))

    # Create a new figure and axis
    _, ax = plt.subplots(figsize=(8, 12))

    # Scatter plot with color gradient
    scatter = ax.scatter(
        flattened_shap_values,
        repeated_feature_names,
        c=colors,
        s=10,
        cmap="bwr",
    )

    # Add horizontal lines for each feature
    for i in range(len(top_feature_names)):
        ax.axhline(i, color="gray", linestyle="--", linewidth=0.5)

    # Add vertical line at x=0
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)

    ax.set_ylabel("Feature")

    # Add a color bar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Feature value")

    # Save the plot
    os.makedirs(f"tex/images/{env.metadata['name']}/{memory}/{agent}", exist_ok=True)

    plt.savefig(
        f"tex/images/{env.metadata['name']}/{memory}/{agent}/{target}_shap.pdf".replace(
            " ", "_"
        )
    )
    plt.close()


def get_data(net, agent, total_steps=10000, steps_per_cycle=100, seed=None):
    observations = []
    actions = []
    num_cycles = total_steps // steps_per_cycle
    with torch.no_grad():
        for i in tqdm(range(num_cycles), desc="Simulating cycles"):
            if seed:
                step_results = sim_steps(net, steps_per_cycle, seed=seed + i + 200)
            else:
                step_results = sim_steps(net, steps_per_cycle, 378429)

            for entry in step_results:
                obs = entry.get("observation")
                act = entry.get("action")
                if obs is not None and act is not None:
                    actions.append(act[agent])
                    observations.append(obs[agent])
        return numpyfy(observations), numpyfy(actions)
