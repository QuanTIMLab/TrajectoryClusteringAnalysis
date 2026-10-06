import sys
from pathlib import Path

import streamlit as st

# Streamlit only puts the script directory on sys.path; the repository root is needed for `app.*` imports.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.components.io import external_dataset_options


APP_DIRECTORY = Path(__file__).resolve().parent
st.set_page_config(
    page_title="TCA — démonstrateur",
    page_icon="🧭",
    layout="wide",
)

pages = [
    st.Page(APP_DIRECTORY / "pages" / "1_Unidimensional.py", title="Unidimensionnel", icon="📈"),
    st.Page(APP_DIRECTORY / "pages" / "2_Multidimensional.py", title="Multidimensionnel", icon="🧩"),
]

with st.sidebar:
    st.title("TCA")
    st.caption("Trajectory Clustering Analysis")
    mode = st.radio("Mode", ["Unidimensionnel", "Multidimensionnel"])
    dataset_options = (
        ["unidimensional_data.csv"]
        if mode == "Unidimensionnel"
        else ["multidimensional_data.csv"]
    )
    dataset_options.extend(external_dataset_options())
    if st.session_state.get("demo_dataset") not in dataset_options:
        st.session_state["demo_dataset"] = dataset_options[0]
    st.selectbox("Jeu d’exemple", dataset_options, key="demo_dataset")
    st.slider("Limite globale d’individus", 2, 300, 150, 10, key="global_row_limit")
    st.caption("Les fichiers importés sur une page remplacent l’exemple.")

st.title("Démonstrateur interactif TCA")
st.write(
    "Explorez le clustering de trajectoires unidimensionnelles ou la décomposition "
    "SWoTTeD sur des événements longitudinaux. Les calculs sont exécutés localement "
    "et mis en cache pendant la session."
)
st.info(
    "Choisissez une page dans la navigation ci-dessus. Pour une analyse unidimensionnelle, "
    "les distances sont quadratiques en nombre d’individus ; le démonstrateur limite donc "
    "la taille des jeux."
)

st.navigation(pages).run()
