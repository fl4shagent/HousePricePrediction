import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@st.cache_resource
def load_prediction_model():
    from api.predict import load_models
    load_models()


@st.cache_data
def load_raw_data():
    return pd.read_csv("data/raw/resale_transactions.csv", low_memory=False)


def render():
    st.header("Price Predictor")
    st.markdown("Enter flat details to get a predicted resale price.")

    load_prediction_model()
    df = load_raw_data()

    col1, col2 = st.columns(2)

    with col1:
        town = st.selectbox("Town", sorted(df["town"].unique()))
        flat_type = st.selectbox("Flat Type", ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM",
                                                "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"])
        floor_area = st.number_input("Floor Area (sqm)", min_value=20.0, max_value=300.0, value=93.0, step=1.0)
        flat_model = st.selectbox("Flat Model", sorted(df["flat_model"].unique()))

    with col2:
        storey_options = sorted(df["storey_range"].dropna().unique())
        storey_range = st.selectbox("Storey Range", storey_options, index=2)
        remaining_years = st.slider("Remaining Lease (years)", 40, 99, 70)
        remaining_lease = f"{remaining_years} years"
        block = st.text_input("Block", value="406")
        street_name = st.text_input("Street Name", value="ANG MO KIO AVE 10")

    transaction_month = st.text_input("Transaction Month (YYYY-MM)", value="2025-10")

    if st.button("Predict Price", type="primary", use_container_width=True):
        from api.predict import predict_price
        from api.schemas import PredictionRequest

        req = PredictionRequest(
            town=town, flat_type=flat_type, floor_area_sqm=floor_area,
            flat_model=flat_model, storey_range=storey_range,
            remaining_lease=remaining_lease, block=block,
            street_name=street_name, transaction_month=transaction_month,
        )

        result = predict_price(req)

        st.success(f"### Predicted Price: {result['predicted_price_formatted']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Model Version", result["model_version"])
        col2.metric("Features Used", result["features_used"])
        col3.metric("Model Accuracy", "95.1% PER10")

        # Show comparable transactions
        st.subheader("Recent Comparable Transactions")
        comparable = df[
            (df["town"] == town) &
            (df["flat_type"] == flat_type) &
            (df["month"] >= "2024-01")
        ].sort_values("month", ascending=False).head(10)

        if len(comparable) > 0:
            display_cols = ["month", "block", "street_name", "storey_range",
                           "floor_area_sqm", "remaining_lease", "resale_price"]
            available_cols = [c for c in display_cols if c in comparable.columns]
            st.dataframe(
                comparable[available_cols].reset_index(drop=True),
                use_container_width=True,
                column_config={"resale_price": st.column_config.NumberColumn(format="$%d")},
            )
        else:
            st.info("No recent comparable transactions found.")
