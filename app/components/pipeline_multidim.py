import pandas as pd
import streamlit as st

from trajectoryclusteringanalysis.multidimensional.analysis import MultidimensionalAnalyzer


MAX_MULTIDIMENSIONAL_INDIVIDUALS = 100
MAX_MULTIDIMENSIONAL_EPOCHS = 20
MAX_MULTIDIMENSIONAL_TIME_POINTS = 60
MAX_MULTIDIMENSIONAL_EVENTS = 50


def validate_multidimensional_data(data, index_col, time_col, event_col):
    if data.empty:
        raise ValueError("Le fichier ne contient aucune ligne.")
    missing = [column for column in (index_col, time_col, event_col) if column not in data.columns]
    if missing:
        raise ValueError(f"Colonnes requises absentes : {', '.join(missing)}")
    if len({index_col, time_col, event_col}) != 3:
        raise ValueError("Les colonnes individu, temps et événement doivent être différentes.")
    valid = data.dropna(subset=[index_col, time_col, event_col]).copy()
    if valid.empty:
        raise ValueError("Aucune ligne complète sur les colonnes individu, temps et événement.")
    return valid


@st.cache_data(show_spinner="Entraînement SWoTTeD sur CPU…")
def run_multidim_pipeline(
    data,
    index_col,
    time_col,
    event_col,
    rank,
    time_window_length,
    epochs,
    reg_term_ns,
    reg_term_s,
):
    if epochs > MAX_MULTIDIMENSIONAL_EPOCHS:
        raise ValueError(f"La limite de cette démo est de {MAX_MULTIDIMENSIONAL_EPOCHS} epochs.")
    individual_ids = data[index_col].drop_duplicates().head(MAX_MULTIDIMENSIONAL_INDIVIDUALS)
    sample = data[data[index_col].isin(individual_ids)].copy()
    if sample[index_col].nunique() < 2:
        raise ValueError("Il faut au moins deux individus pour lancer SWoTTeD.")
    if sample[time_col].nunique() > MAX_MULTIDIMENSIONAL_TIME_POINTS:
        raise ValueError(
            f"La limite de cette démo est de {MAX_MULTIDIMENSIONAL_TIME_POINTS} périodes."
        )
    if sample[event_col].nunique() > MAX_MULTIDIMENSIONAL_EVENTS:
        raise ValueError(
            f"La limite de cette démo est de {MAX_MULTIDIMENSIONAL_EVENTS} événements."
        )

    analyzer = MultidimensionalAnalyzer(sample, index_col, time_col, event_col)
    if not analyzer.has_time_event_structure():
        raise ValueError("Le jeu de données doit contenir les colonnes individu, temps et événement.")
    analyzer.transform_time_event_structure_to_tensor()
    if time_window_length > analyzer.T:
        raise ValueError(f"La fenêtre temporelle ne peut pas dépasser {analyzer.T} périodes.")
    if rank > analyzer.N:
        raise ValueError(f"Le rang ne peut pas dépasser {analyzer.N} événements.")
    analyzer.fit_swotted_decomposition(
        analyzer.get_tensor(),
        rank=rank,
        time_window_length=time_window_length,
        reg_term_ns=reg_term_ns,
        reg_term_s=reg_term_s,
        n_epochs=epochs,
    )
    return analyzer.to_phenotype_intensity(), len(individual_ids), analyzer.T, analyzer.N
