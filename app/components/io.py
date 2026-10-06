from io import BytesIO
from pathlib import Path

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "unidimensional_data.csv": REPOSITORY_ROOT / "data" / "unidimensional_data.csv",
    "multidimensional_data.csv": REPOSITORY_ROOT / "data" / "multidimensional_data.csv",
}
EXTERNAL_DATA_DIRECTORIES = (
    Path("/app/external-data"),
    REPOSITORY_ROOT / "external-data",
)


def external_dataset_options():
    return {
        f"Externe : {path.name}": path
        for directory in EXTERNAL_DATA_DIRECTORIES
        if directory.is_dir()
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in {".csv", ".xlsx"}
    }


def read_uploaded_file(uploaded_file):
    """Read a CSV or Excel upload and return a dataframe with its source name."""
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        data = pd.read_csv(BytesIO(uploaded_file.getvalue()))
    elif suffix == ".xlsx":
        data = pd.read_excel(BytesIO(uploaded_file.getvalue()))
    else:
        raise ValueError("Format non pris en charge : utilisez un fichier CSV ou XLSX.")
    return data, uploaded_file.name


def read_sample_dataset(dataset_name):
    path = DATASETS.get(dataset_name) or external_dataset_options().get(dataset_name)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"Jeu de démonstration introuvable : {dataset_name}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path), dataset_name
    return pd.read_excel(path), dataset_name


def selected_dataset(uploaded_file, dataset_name):
    if uploaded_file is not None:
        return read_uploaded_file(uploaded_file)
    return read_sample_dataset(dataset_name)
