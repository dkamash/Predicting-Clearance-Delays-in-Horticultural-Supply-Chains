import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "tlip_like_consignments_5000.jsonl"
MODEL_PATH = BASE_DIR / "models" / "log_reg_pipeline.pkl"


@st.cache_resource
def load_model():
    # Cache the pickled pipeline so it is loaded only once per session.
    with MODEL_PATH.open("rb") as file:
        return pickle.load(file)


@st.cache_data
def load_data():
    # Cache the dataset for dropdown population; not used for predictions.
    if not DATA_PATH.exists():
        return None
    return pd.read_json(DATA_PATH, lines=True)


def options_for(df: pd.DataFrame, column: str):
    # Provide sorted unique values for select boxes.
    if df is None or column not in df.columns:
        return []
    values = df[column].dropna().unique().tolist()
    return sorted(values)


def main() -> None:
    # Build the Streamlit UI and run predictions on demand.
    st.set_page_config(page_title="Clearance Delay Predictor", layout="centered")
    st.title("Clearance Delay Risk Prediction")
    st.caption("Baseline logistic regression model (early-warning features).")

    model_payload = load_model()
    model = model_payload["model"]
    features = model_payload["features"]

    df = load_data()

    st.subheader("Shipment Details")
    shipment_mode = st.selectbox(
        "Shipment mode",
        options=options_for(df, "shipment_mode") or ["Unknown"],
    )
    commodity = st.selectbox(
        "Commodity",
        options=options_for(df, "commodity") or ["Unknown"],
    )
    hs_code = st.selectbox(
        "HS code",
        options=options_for(df, "hs_code") or ["Unknown"],
    )
    origin_country = st.selectbox(
        "Origin country",
        options=options_for(df, "origin_country") or ["Unknown"],
    )
    destination_country = st.selectbox(
        "Destination country",
        options=options_for(df, "destination_country") or ["Unknown"],
    )
    exporter_profile = st.selectbox(
        "Exporter profile",
        options=options_for(df, "exporter_profile") or ["Unknown"],
    )

    st.subheader("Documentation and Operations")
    doc_completeness_score = st.number_input(
        "Document completeness score", min_value=0.0, max_value=1.0, value=0.8, step=0.01
    )
    missing_docs_proxy = st.number_input(
        "Missing docs proxy", min_value=0.0, value=0.0, step=1.0
    )
    doc_amendments = st.number_input(
        "Document amendments", min_value=0.0, value=0.0, step=1.0
    )
    congestion_index = st.number_input(
        "Congestion index", min_value=0.0, value=0.5, step=0.05
    )
    gross_weight_kg = st.number_input(
        "Gross weight (kg)", min_value=0.0, value=1000.0, step=10.0
    )
    declared_value_usd = st.number_input(
        "Declared value (USD)", min_value=0.0, value=10000.0, step=100.0
    )
    is_weekend_created = st.selectbox(
        "Created on weekend?",
        options=[0, 1],
        format_func=lambda v: "Yes" if v == 1 else "No",
    )

    input_df = pd.DataFrame(
        [
            {
                "shipment_mode": shipment_mode,
                "commodity": commodity,
                "hs_code": hs_code,
                "origin_country": origin_country,
                "destination_country": destination_country,
                "exporter_profile": exporter_profile,
                "doc_completeness_score": doc_completeness_score,
                "missing_docs_proxy": missing_docs_proxy,
                "doc_amendments": doc_amendments,
                "congestion_index": congestion_index,
                "is_weekend_created": is_weekend_created,
                "gross_weight_kg": gross_weight_kg,
                "declared_value_usd": declared_value_usd,
            }
        ]
    )

    input_df = input_df[features]

    if st.button("Predict delay risk"):
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0, 1])

        label = "Delayed" if prediction == 1 else "On time"
        st.metric("Prediction", label)
        st.metric("Delay probability", f"{probability:.2%}")


if __name__ == "__main__":
    main()
