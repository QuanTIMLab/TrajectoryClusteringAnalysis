import numpy as np
import pandas as pd
import streamlit as st

from trajectoryclusteringanalysis.tca import TCA


MAX_UNIDIMENSIONAL_ROWS = 300


def to_wide_sequences(data, index_col, sequence_cols, data_format, time_col=None, state_col=None):
    if data.empty:
        raise ValueError("Le fichier ne contient aucune ligne.")
    if index_col not in data.columns:
        raise ValueError(f"La colonne identifiant « {index_col} » est absente.")

    if data_format == "Format long":
        if time_col not in data.columns or state_col not in data.columns:
            raise ValueError("Choisissez une colonne de temps et une colonne de statut.")
        if len({index_col, time_col, state_col}) != 3:
            raise ValueError("Les colonnes individu, temps et statut doivent être différentes.")
        long_data = data[[index_col, time_col, state_col]].copy()
        long_data = long_data.dropna(subset=[index_col, time_col, state_col])
        if long_data.empty:
            raise ValueError("Aucune trajectoire complète après suppression des valeurs manquantes.")
        duplicate_count = long_data.duplicated([index_col, time_col]).sum()
        wide = long_data.pivot_table(
            index=index_col, columns=time_col, values=state_col, aggfunc="first", sort=True
        )
        if duplicate_count:
            st.warning(
                f"{duplicate_count} doublon(s) identifiant/temps : le premier statut est conservé."
            )
        wide.columns = [str(column) for column in wide.columns]
        wide.index.name = index_col
        wide = wide.reset_index()
    else:
        if not sequence_cols:
            raise ValueError("Sélectionnez au moins une colonne de séquence.")
        if index_col in sequence_cols:
            raise ValueError("La colonne identifiant ne peut pas faire partie des colonnes de séquence.")
        missing_cols = [column for column in sequence_cols if column not in data.columns]
        if missing_cols:
            raise ValueError(f"Colonnes de séquence absentes : {', '.join(missing_cols)}")
        wide = data[[index_col, *sequence_cols]].copy()

    if wide[index_col].isna().any():
        raise ValueError("La colonne identifiant contient des valeurs manquantes.")
    if wide[index_col].duplicated().any():
        raise ValueError("Chaque individu doit apparaître une seule fois dans le format large.")
    if len(wide) < 2:
        raise ValueError("Il faut au moins deux individus pour effectuer un clustering.")
    time_columns = [column for column in wide.columns if column != index_col]
    if not time_columns:
        raise ValueError("Aucune colonne de statut/temps exploitable.")
    wide = wide[[index_col, *time_columns]].copy()
    wide[time_columns] = wide[time_columns].fillna("(manquant)").astype(str)
    states = list(pd.unique(wide[time_columns].to_numpy().ravel()))
    if len(states) < 2:
        raise ValueError("Au moins deux statuts différents sont nécessaires.")
    if any("-" in state for state in states):
        raise ValueError("Les valeurs de statut ne doivent pas contenir le caractère « - ».")
    return wide, time_columns, states


@st.cache_data(show_spinner="Calcul du clustering unidimensionnel…")
def run_unidim_pipeline(data, index_col, metric, method, num_clusters):
    if len(data) > MAX_UNIDIMENSIONAL_ROWS:
        raise ValueError(f"La limite de calcul est de {MAX_UNIDIMENSIONAL_ROWS} individus.")
    states = list(pd.unique(data.drop(columns=[index_col]).to_numpy().ravel()))
    model = TCA(
        data=data,
        index_col=index_col,
        alphabet=states,
        states=states,
        mode="unidimensional",
    )
    distance_matrix = None
    linkage_matrix = None
    if method != "K-means (fréquences)":
        substitution_costs = (
            model.compute_substitution_cost_matrix() if metric == "optimal_matching" else None
        )
        distance_matrix = model.compute_distance_matrix(
            data,
            metric=metric,
            substitution_cost_matrix=substitution_costs,
        )
        linkage_matrix = model.hierarchical_clustering(
            distance_matrix, method="average", optimal_ordering=True
        )

    if method == "CAH":
        clusters = model.assign_clusters(linkage_matrix, num_clusters)
    elif method == "K-medoids":
        clusters, _, _ = model.kmedoids_clustering(
            distance_matrix, num_clusters=num_clusters, random_state=0
        )
    else:
        clusters, _, _ = model.kmeans_on_frequency(
            num_clusters=num_clusters, random_state=0, n_init=10
        )

    return {
        "clusters": np.asarray(clusters),
        "linkage_matrix": linkage_matrix,
        "distance_matrix": distance_matrix,
    }
