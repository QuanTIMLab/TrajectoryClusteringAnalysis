import zipfile

import pandas as pd
import streamlit as st

from app.components.io import selected_dataset
from app.components.pipeline_multidim import (
    MAX_MULTIDIMENSIONAL_EPOCHS,
    MAX_MULTIDIMENSIONAL_EVENTS,
    MAX_MULTIDIMENSIONAL_INDIVIDUALS,
    MAX_MULTIDIMENSIONAL_TIME_POINTS,
    run_multidim_pipeline,
    validate_multidimensional_data,
)
from app.components.plots import phenotype_intensity_figure


st.header("Analyse multidimensionnelle")
st.write("Décomposition exploratoire d’événements longitudinaux avec SWoTTeD.")
st.warning(
    "Calcul CPU potentiellement long. Cette démo limite l’analyse à "
    f"{MAX_MULTIDIMENSIONAL_INDIVIDUALS} individus et {MAX_MULTIDIMENSIONAL_EPOCHS} epochs."
)
dataset_name = st.session_state.get("demo_dataset", "multidimensional_data.csv")
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
suggested_id = next((column for column in columns if column.lower() in {"id", "patient_id", "patientid"}), columns[0])
suggested_time = next((column for column in columns if column.lower() in {"time", "month", "date"}), columns[min(1, len(columns) - 1)])
suggested_event = next((column for column in columns if column.lower() in {"event", "care_event", "status"}), columns[-1])
index_col = st.selectbox("Colonne individu", columns, index=columns.index(suggested_id))
time_col = st.selectbox("Colonne temps", columns, index=columns.index(suggested_time))
event_col = st.selectbox("Colonne événement", columns, index=columns.index(suggested_event))

try:
    valid_data = validate_multidimensional_data(source_data, index_col, time_col, event_col)
except ValueError as error:
    st.error(str(error))
    st.stop()

individual_count = valid_data[index_col].nunique()
time_count = valid_data[time_col].nunique()
event_count = valid_data[event_col].nunique()
st.write(f"{individual_count} individus · {time_count} périodes · {event_count} événements")
if individual_count > MAX_MULTIDIMENSIONAL_INDIVIDUALS:
    st.info(f"Un sous-échantillon des {MAX_MULTIDIMENSIONAL_INDIVIDUALS} premiers individus sera utilisé.")
if time_count > MAX_MULTIDIMENSIONAL_TIME_POINTS or event_count > MAX_MULTIDIMENSIONAL_EVENTS:
    st.warning(
        f"Le sous-échantillon doit contenir au plus {MAX_MULTIDIMENSIONAL_TIME_POINTS} périodes "
        f"et {MAX_MULTIDIMENSIONAL_EVENTS} événements."
    )
rank = st.number_input("Rang SWoTTeD", min_value=1, max_value=min(4, max(event_count, 1)), value=min(2, max(event_count, 1)))
time_window = st.number_input("Fenêtre temporelle", min_value=1, max_value=max(time_count, 1), value=min(3, max(time_count, 1)))
epochs = st.slider("Epochs", 1, MAX_MULTIDIMENSIONAL_EPOCHS, 5)
reg_term_ns = st.number_input("Régularisation non-succession", 0.0, 1.0, 0.5, 0.1)
reg_term_s = st.number_input("Régularisation parcimonie", 0.0, 1.0, 0.5, 0.1)

run_signature = (
    int(pd.util.hash_pandas_object(valid_data, index=True).sum()),
    index_col,
    time_col,
    event_col,
    int(rank),
    int(time_window),
    int(epochs),
    float(reg_term_ns),
    float(reg_term_s),
)
if st.button("Lancer la décomposition SWoTTeD", type="primary"):
    try:
        with st.spinner("SWoTTeD s’entraîne sur CPU ; l’interface reste disponible."):
            phenotypes, sampled_individuals, periods, events = run_multidim_pipeline(
                valid_data,
                index_col,
                time_col,
                event_col,
                int(rank),
                int(time_window),
                int(epochs),
                float(reg_term_ns),
                float(reg_term_s),
            )
        st.session_state["multidim_result"] = (
            phenotypes,
            index_col,
            sampled_individuals,
            periods,
            events,
        )
        st.session_state["multidim_result_signature"] = run_signature
    except (ImportError, RuntimeError, TypeError, ValueError) as error:
        st.error(f"La décomposition SWoTTeD a échoué : {error}")

result = st.session_state.get("multidim_result")
if (
    result is not None
    and result[1] == index_col
    and st.session_state.get("multidim_result_signature") == run_signature
):
    phenotypes, _, sampled_individuals, periods, events = result
    st.success(
        f"Décomposition terminée : {sampled_individuals} individus, "
        f"{events} événements, {periods} périodes."
    )
    st.pyplot(phenotype_intensity_figure(phenotypes, index_col), clear_figure=True)
    st.dataframe(phenotypes, use_container_width=True)
