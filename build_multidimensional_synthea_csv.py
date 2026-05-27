from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def month_index(dates: pd.Series) -> pd.Series:
    """Encode a datetime series to a monotonic month index (year*12 + month)."""
    return dates.dt.year * 12 + dates.dt.month


def load_medication_events(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["PATIENT", "START", "DESCRIPTION"])
    df = df.rename(columns={"PATIENT": "patient_id", "START": "event_date", "DESCRIPTION": "medication_events"})
    # Parse all dates in UTC then drop timezone info to keep a consistent dtype.
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce", utc=True).dt.tz_convert(None)
    df = df.dropna(subset=["patient_id", "event_date"])
    df["medication_events"] = df["medication_events"].astype(str).str.strip()
    df = df[df["medication_events"].notna() & (df["medication_events"] != "") & (df["medication_events"].str.lower() != "nan")]
    return df[["patient_id", "event_date", "medication_events"]]


def build_multidimensional_dataset(input_dir: Path) -> pd.DataFrame:
    csv_path = input_dir / "medications.csv"
    if not csv_path.exists():
        raise FileNotFoundError("medications.csv was not found in the input directory.")

    all_events = load_medication_events(csv_path)

    # Use month granularity to ensure simultaneous events in the same month share identical time.
    all_events["event_month_index"] = month_index(all_events["event_date"])

    first_month_index = (
        all_events.groupby("patient_id", as_index=False)["event_month_index"]
        .min()
        .rename(columns={"event_month_index": "first_month_index"})
    )

    all_events = all_events.merge(first_month_index, on="patient_id", how="left")
    all_events["time"] = all_events["event_month_index"] - all_events["first_month_index"]

    # Keep only the first 180 months of follow-up.
    all_events = all_events[all_events["time"] < 180]

    # Keep only non-negative times (safety against malformed dates).
    all_events = all_events[all_events["time"] >= 0]

    # Keep only the 10 most frequent medication descriptions in the current dataset.
    top_medications = all_events["medication_events"].value_counts().head(10).index
    all_events = all_events[all_events["medication_events"].isin(top_medications)]
    
    # Keep a single row per patient, month, and medication description.
    all_events = all_events.drop_duplicates(subset=["patient_id", "time", "medication_events"])

    # Keep only patients who have at least one month with simultaneous medication events.
    simultaneous_patients = (
        all_events.groupby(["patient_id", "time"]).size().loc[lambda sizes: sizes >= 2].index.get_level_values("patient_id").unique()
    )
    all_events = all_events[all_events["patient_id"].isin(simultaneous_patients)]

    result = all_events[["patient_id", "time", "medication_events"]].copy()
    result = result.sort_values(["patient_id", "time", "medication_events"]).reset_index(drop=True)

    # Replace original UUID patient IDs with sequential integer IDs (1..N).
    ordered_patients = pd.Index(result["patient_id"].drop_duplicates())
    patient_id_map = pd.Series(range(1, len(ordered_patients) + 1), index=ordered_patients)
    result["patient_id"] = result["patient_id"].map(patient_id_map).astype(int)

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a medication-only long-format CSV from Synthea exports. "
            "Output includes patient_id, time (months since first medication event), "
            "and medication_events containing the DESCRIPTION text."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data") / "10k_synthea_covid19_csv",
        help="Directory containing Synthea CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "multidimensional_synthea_10k.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_multidimensional_dataset(args.input_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print(f"Saved {len(dataset)} rows to {args.output}")
    print(f"Columns: {', '.join(dataset.columns)}")


if __name__ == "__main__":
    main()
