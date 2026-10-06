import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list


def dendrogram_figure(linkage_matrix):
    figure, axis = plt.subplots(figsize=(10, 4))
    dendrogram(linkage_matrix, no_labels=True, ax=axis)
    axis.set(title="Dendrogramme des trajectoires", xlabel="Individus", ylabel="Distance")
    figure.tight_layout()
    return figure


def cluster_heatmap_figures(data, index_col, clusters, linkage_matrix=None, max_clusters=8):
    state_columns = [column for column in data.columns if column != index_col]
    labels = np.asarray(clusters)
    ordered = (
        leaves_list(linkage_matrix)
        if linkage_matrix is not None
        else np.argsort(labels, kind="stable")
    )
    unique_clusters = np.unique(labels[ordered])[:max_clusters]
    figures = []
    values = data[state_columns].astype(str)
    states = list(pd.unique(values.to_numpy().ravel()))
    palette = plt.get_cmap("viridis", max(len(states), 2))
    state_codes = {state: index for index, state in enumerate(states)}
    encoded = values.replace(state_codes).to_numpy(dtype=int)

    for cluster in unique_clusters:
        rows = [row for row in ordered if labels[row] == cluster]
        figure, axis = plt.subplots(figsize=(10, max(2, min(6, len(rows) * 0.12))))
        image = axis.imshow(encoded[rows], aspect="auto", interpolation="nearest", cmap=palette)
        axis.set(
            title=f"Cluster {cluster} — {len(rows)} individu(s)",
            xlabel="Période",
            ylabel="Individus",
        )
        axis.set_xticks(np.arange(len(state_columns)))
        axis.set_xticklabels(state_columns, rotation=45, ha="right")
        colorbar = figure.colorbar(image, ax=axis, ticks=np.arange(len(states)))
        colorbar.ax.set_yticklabels(states)
        figure.tight_layout()
        figures.append(figure)
    return figures


def status_percentage_figure(data, index_col, clusters):
    state_columns = [column for column in data.columns if column != index_col]
    labels = np.asarray(clusters)
    cluster_values = np.unique(labels)
    figure, axes = plt.subplots(
        math.ceil(len(cluster_values) / 2),
        min(2, len(cluster_values)),
        figsize=(12, 4 * math.ceil(len(cluster_values) / 2)),
        squeeze=False,
    )
    palette = plt.get_cmap("tab10")
    for position, cluster in enumerate(cluster_values):
        axis = axes[position // 2, position % 2]
        subset = data.iloc[np.flatnonzero(labels == cluster)][state_columns]
        states = list(pd.unique(subset.to_numpy().ravel()))
        for state_index, state in enumerate(states):
            percentages = subset.eq(state).sum(axis=0) * 100 / max(len(subset), 1)
            axis.plot(
                range(len(state_columns)),
                percentages,
                marker="o",
                label=str(state),
                color=palette(state_index % 10),
            )
        axis.set(
            title=f"Cluster {cluster}",
            xlabel="Période",
            ylabel="Individus (%)",
            xticks=range(len(state_columns)),
            xticklabels=state_columns,
        )
        axis.tick_params(axis="x", labelrotation=45)
        axis.grid(alpha=0.25)
        axis.legend(title="Statut", bbox_to_anchor=(1.02, 1), loc="upper left")
    for position in range(len(cluster_values), axes.size):
        figure.delaxes(axes.flat[position])
    figure.tight_layout()
    return figure


def phenotype_intensity_figure(phenotypes, index_col):
    columns = [column for column in phenotypes.columns if column != index_col]
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.boxplot([phenotypes[column].dropna().to_numpy() for column in columns])
    axis.set_xticks(range(1, len(columns) + 1))
    axis.set_xticklabels(columns)
    axis.set(title="Intensité des phénotypes SWoTTeD", ylabel="Intensité normalisée")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure
