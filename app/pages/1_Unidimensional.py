import zipfile

import pandas as pd
import streamlit as st

from app.components.io import selected_dataset
from app.components.pipeline_unidim import (
    MAX_UNIDIMENSIONAL_ROWS,
    run_unidim_pipeline,
    to_wide_sequences,
)
from app.components.plots import (
    cluster_heatmap_figures,
    dendrogram_figure,
    status_percentage_figure,
)


st.header("Analyse unidimensionnelle")
st.write("Importez des séquences larges ou des observations au format long (individu, temps, statut).")
dataset_name = st.session_state.get("demo_dataset", "unidimensional_data.csv")
uploaded_file = st.file_uploader("Charger un CSV ou Excel", type=["csv", "xlsx"])

try:
    source_data, source_name = selected_dataset(uploaded_file, dataset_name)
except (OSError, ValueError, ImportError, zipfile.BadZipFile) as error:
    st.error(f"Impossible de charger le jeu de données : {error}")
    st.stop()

if source_data.empty:
    st.error("Le jeu de données est vide.")
    st.stop()
st.caption(f"Source : {source_name} · {len(source_data):,} lignes")
columns = list(source_data.columns)
if not columns:
    st.error("Aucune colonne détectée dans le fichier.")
    st.stop()

suggested_index = next((column for column in columns if column.lower() in {"id", "patient_id", "patientid"}), columns[0])
index_col = st.selectbox("Colonne identifiant", columns, index=columns.index(suggested_index))
format_guess = "Format long" if {"month", "care_status"}.issubset({column.lower() for column in columns}) else "Format large"
data_format = st.radio(
    "Format des séquences",
    ["Format large", "Format long"],
    index=int(format_guess == "Format long"),
    horizontal=True,
)

if data_format == "Format long":
    suggested_time = next((column for column in columns if column.lower() in {"month", "time", "date"}), columns[1 if len(columns) > 1 else 0])
    suggested_state = next((column for column in columns if column.lower() in {"care_status", "status", "state", "event"}), columns[-1])
    time_col = st.selectbox("Colonne de temps", columns, index=columns.index(suggested_time))
    state_col = st.selectbox("Colonne de statut", columns, index=columns.index(suggested_state))
    sequence_cols = []
else:
    time_col = None
    state_col = None
    sequence_cols = st.multiselect(
        "Colonnes de séquence (ordre temporel)",
        [column for column in columns if column != index_col],
        default=[column for column in columns if column != index_col],
    )

source_signature = int(pd.util.hash_pandas_object(source_data, index=True).sum())
current_prepare_signature = (
    source_name,
    source_signature,
    index_col,
    data_format,
    tuple(sequence_cols),
    time_col,
    state_col,
)
if st.button("Préparer les trajectoires", type="secondary"):
    try:
        wide, time_columns, states = to_wide_sequences(
            source_data,
            index_col,
            sequence_cols,
            data_format,
            time_col=time_col,
            state_col=state_col,
        )
        st.session_state["unidim_prepared"] = wide
        st.session_state["unidim_index_col"] = index_col
        st.session_state["unidim_time_columns"] = time_columns
        st.session_state["unidim_prepare_signature"] = current_prepare_signature
        st.success(f"{len(wide)} individus · {len(time_columns)} périodes · {len(states)} statuts")
    except (KeyError, ValueError) as error:
        st.error(str(error))

prepared = st.session_state.get("unidim_prepared")
if prepared is not None:
    if st.session_state.get("unidim_prepare_signature") != current_prepare_signature:
        st.info("Les données ou colonnes ont changé. Préparez de nouveau les trajectoires.")
        st.stop()
    prepared_index = st.session_state.get("unidim_index_col")
    if prepared_index != index_col:
        st.warning("La colonne identifiant a changé : préparez de nouveau les trajectoires.")
    else:
        st.dataframe(prepared.head(8), use_container_width=True)
        row_limit = min(
            MAX_UNIDIMENSIONAL_ROWS,
            st.session_state.get("global_row_limit", MAX_UNIDIMENSIONAL_ROWS),
        )
        if len(prepared) > row_limit:
            st.warning(f"Seuls les {row_limit} premiers individus seront analysés.")
            prepared = prepared.head(row_limit).copy()
        if len(prepared) < 2:
            st.error("Il faut au moins deux individus.")
            st.stop()

        method = st.selectbox("Méthode de clustering", ["CAH", "K-medoids", "K-means (fréquences)"])
        metric = st.selectbox("Métrique", ["hamming", "optimal_matching", "levenshtein"])
        if method == "K-means (fréquences)":
            st.caption("La métrique n’est pas utilisée par k-means sur les fréquences d’états.")
        cluster_count = st.number_input(
            "Nombre de clusters",
            min_value=2,
            max_value=min(10, len(prepared)),
            value=min(3, len(prepared)),
            step=1,
        )
        run_signature = (
            int(pd.util.hash_pandas_object(prepared, index=True).sum()),
            prepared_index,
            metric,
            method,
            int(cluster_count),
        )
        if st.button("Lancer le clustering", type="primary"):
            try:
                with st.spinner("Préparation et calcul des clusters…"):
                    result = run_unidim_pipeline(
                        prepared,
                        prepared_index,
                        metric,
                        method,
                        int(cluster_count),
                    )
                st.session_state["unidim_result"] = result
                st.session_state["unidim_analyzed_data"] = prepared
                st.session_state["unidim_result_signature"] = run_signature
            except (ImportError, RuntimeError, TypeError, ValueError) as error:
                st.error(f"Le calcul du clustering a échoué : {error}")

        result = st.session_state.get("unidim_result")
        analyzed = st.session_state.get("unidim_analyzed_data")
        if (
            result is not None
            and analyzed is not None
            and analyzed.equals(prepared)
            and st.session_state.get("unidim_result_signature") == run_signature
        ):
            clusters = result["clusters"]
            output = analyzed.copy()
            output["cluster"] = clusters
            st.subheader("Affectations")
            st.dataframe(output, use_container_width=True)
            if method == "CAH" and result["linkage_matrix"] is not None:
                st.subheader("Dendrogramme")
                st.pyplot(dendrogram_figure(result["linkage_matrix"]), clear_figure=True)
            st.subheader("Heatmaps par cluster")
            figures = cluster_heatmap_figures(
                analyzed,
                prepared_index,
                clusters,
                result["linkage_matrix"],
            )
            for figure in figures:
                st.pyplot(figure, clear_figure=True)
            st.subheader("Pourcentage des statuts au cours du temps")
            st.pyplot(status_percentage_figure(analyzed, prepared_index, clusters), clear_figure=True)
