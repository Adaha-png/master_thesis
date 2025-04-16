import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch

from counterfactuals_shapley.wrappers import numpyfy
from train_tune_eval.rllib_train import env_creator


def pred(net, target, device, obs):
    obs = torch.tensor(obs).unsqueeze(0).to(device, dtype=torch.float32)
    obs.squeeze()
    vals = net(obs)

    if target == None:
        vals = vals.unsqueeze(0).cpu().detach().numpy()
    else:
        vals = vals.cpu().detach().numpy()[0, :, target]
    return vals


def kernel_explainer(net, X, target, device):
    X = numpyfy(X)

    try:
        net(torch.Tensor(X[0]).to(device))
    except RuntimeError as e:
        feats = len(env_creator().feature_names)
        X = X[:, :feats]

    explainer = shap.KernelExplainer(
        partial(pred, net, target, device), shap.kmeans(X, 200)
    )
    return explainer


def shap_plot(
    agent,
    memory,
    X,
    explainer,
    feature_names,
    target,
    extras,
    explainer_extras,
    slices,
):
    env = env_creator()
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = np.stack(shap_values, axis=-1)

    assert shap_values.shape[1] == len(
        feature_names
    ), "Mismatch between SHAP values and feature names dimensions."

    if slices and len(slices) > 0:
        aggregated_shap_list = []
        aggregated_feature_list = []
        aggregated_feature_names = []

        for i in range(slices[0]):
            aggregated_shap_list.append(shap_values[:, i])
            aggregated_feature_list.append(X[:, i])
            aggregated_feature_names.append(feature_names[i])

        for j in range(1, len(slices)):
            start = slices[j - 1]
            end = slices[j]
            aggregated_shap = np.sum(shap_values[:, start:end], axis=1)
            aggregated_feature = np.sum(X[:, start:end], axis=1)
            if isinstance(feature_names[start], int):
                combined_name = "One-hot"
            else:
                combined_name = feature_names[start].split(" ")[-1]

            aggregated_shap_list.append(aggregated_shap)
            aggregated_feature_list.append(aggregated_feature)
            aggregated_feature_names.append(combined_name)

        # Convert lists into arrays with shape (n_instances, n_groups)
        aggregated_shap_values = np.column_stack(aggregated_shap_list)
        aggregated_feature_values = np.column_stack(aggregated_feature_list)

        mean_shap_values = np.mean(np.abs(aggregated_shap_values), axis=0)
        top_indices = np.argsort(mean_shap_values)[-15:]
        top_feature_names = np.array(aggregated_feature_names)[top_indices]
        top_shap_values = aggregated_shap_values[:, top_indices]

        flattened_shap_values = top_shap_values.flatten()
        repeated_feature_names = np.tile(
            top_feature_names, aggregated_feature_values.shape[0]
        )
        flattened_feature_values = aggregated_feature_values[:, top_indices].flatten()

    else:
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        top_indices = np.argsort(mean_shap_values)[-15:]
        top_feature_names = np.array(feature_names)[top_indices]
        top_shap_values = shap_values[:, top_indices]
        flattened_shap_values = top_shap_values.flatten()
        repeated_feature_names = np.tile(top_feature_names, X.shape[0])
        flattened_feature_values = X[:, top_indices].flatten()

    norm = plt.Normalize(np.min(flattened_feature_values), 1)

    colors = plt.cm.viridis(norm(flattened_feature_values))

    plt.rcParams.update(
        {
            "font.family": "serif",
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )

    _, ax = plt.subplots(figsize=(6, 7.4))

    scatter = ax.scatter(
        flattened_shap_values,
        repeated_feature_names,
        c=colors,
        s=10,
        cmap="bwr",
    )

    unique_features = list(dict.fromkeys(repeated_feature_names))
    for i, feat in enumerate(unique_features):
        ax.axhline(i, color="gray", linestyle="--", linewidth=0.5)

    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_ylabel("Feature")
    ax.set_xlabel("Shapley Value")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Feature value")

    out_dir = f"tex/images/{env.metadata['name']}/{memory}/{agent}"
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{target}_{extras}_{explainer_extras}_shap.pgf".replace(" ", "_")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), backend="pgf")
    plt.close()

    if explainer_extras != "none":
        plt.clf()

        start = slices[-2]
        end = slices[-1]
        shap_subset = shap_values[:, start:end]
        X_subset = X[:, start:end]
        feature_names_subset = feature_names[start:end]

        mean_shap_values = np.mean(np.abs(shap_subset), axis=0)
        top_indices = np.argsort(mean_shap_values)[-15:]
        top_feature_names = np.array(feature_names_subset)[top_indices]
        top_shap_values = shap_subset[:, top_indices]

        flattened_shap_subset = top_shap_values.flatten()
        repeated_feature_names_subset = np.tile(top_feature_names, X_subset.shape[0])
        flattened_feature_values_subset = X_subset[:, top_indices].flatten()

        norm_subset = plt.Normalize(
            np.min(flattened_feature_values_subset),
            np.max(flattened_feature_values_subset),
        )
        colors_subset = plt.cm.viridis(norm_subset(flattened_feature_values_subset))

        _, ax_subset = plt.subplots(figsize=(6, 7.4))
        scatter_subset = ax_subset.scatter(
            flattened_shap_subset,
            repeated_feature_names_subset,
            c=colors_subset,
            s=10,
            cmap="bwr",
        )

        unique_features_subset = list(dict.fromkeys(repeated_feature_names_subset))
        for i, feat in enumerate(unique_features_subset):
            ax_subset.axhline(i, color="gray", linestyle="--", linewidth=0.5)

        ax_subset.axvline(0, color="gray", linestyle="-", linewidth=0.5)
        ax_subset.set_ylabel("Feature")
        ax_subset.set_xlabel("Shapley Value")

        cbar_subset = plt.colorbar(scatter_subset, ax=ax_subset)
        cbar_subset.set_label("Feature value")

        filename_subset = (
            f"{target}_{extras}_{explainer_extras}_slice_shap.pgf".replace(" ", "_")
        )

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename_subset), backend="pgf")
        plt.close()
