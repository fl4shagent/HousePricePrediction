import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def render():
    st.header("Model Performance")
    st.markdown("LGBM + XGBoost ensemble trained on 667k transactions (2000–2025)")

    # --- Overall Metrics ---
    st.subheader("Final Model Metrics (Tuned Ensemble)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", "$24,476", help="Mean Absolute Error — average prediction error")
    col2.metric("MAPE", "3.7%", help="Mean Absolute Percentage Error")
    col3.metric("PER10", "95.1%", help="% of predictions within ±10% of actual price")
    col4.metric("R²", "0.9724", help="Proportion of variance explained")

    # --- Model Comparison ---
    st.subheader("All Models Comparison")

    models_data = {
        "Model": ["Ensemble (tuned)", "XGBoost (tuned)", "LightGBM (tuned)", "Random Forest",
                   "Decision Tree", "Linear Regression", "Ridge", "Lasso", "ElasticNet"],
        "MAE": [24476, 24609, 25565, 31014, 36524, 69964, 69965, 74475, 277548],
        "MAPE (%)": [3.7, 3.7, 3.9, 4.5, 5.4, 10.7, 10.7, 11.3, 36.9],
        "PER10 (%)": [95.1, 94.8, 94.6, 91.6, 85.9, 59.2, 59.2, 56.8, 10.3],
        "R²": [0.9724, 0.9721, 0.9702, 0.9561, 0.9374, 0.7766, 0.7766, 0.7452, -1.6391],
    }
    models_df = pd.DataFrame(models_data)

    fig = px.bar(models_df[models_df["MAE"] < 100000], x="Model", y="MAE",
                 color="MAE", color_continuous_scale="RdYlGn_r",
                 labels={"MAE": "MAE (SGD)"})
    fig.update_layout(height=400, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(models_df, use_container_width=True, hide_index=True,
                 column_config={"MAE": st.column_config.NumberColumn(format="$%d"),
                                "R²": st.column_config.NumberColumn(format="%.4f")})

    # --- Iteration History ---
    st.subheader("Model Iteration History")
    st.markdown("Each iteration built on the previous to improve performance.")

    iterations = {
        "Iteration": ["v2: No Macro (default)", "v3: With Macro (default)", "v3: With Macro (tuned)"],
        "MAE": [26000, 26044, 24476],
        "MAPE (%)": [4.0, 3.9, 3.7],
        "PER10 (%)": [94.0, 94.2, 95.1],
        "What Changed": [
            "Baseline — 30 spatial + structural features",
            "Added SORA 3M + CPI (macro barely helped tree models)",
            "RandomizedSearchCV tuning (6% MAE improvement)",
        ],
    }
    iter_df = pd.DataFrame(iterations)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=iter_df["Iteration"], y=iter_df["MAE"],
                          marker_color=["#ff7f0e", "#2ca02c", "#1f77b4"],
                          text=[f"${v:,}" for v in iter_df["MAE"]], textposition="outside"))
    fig2.update_layout(yaxis_title="MAE (SGD)", height=350, margin=dict(t=30))
    st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(iter_df, use_container_width=True, hide_index=True,
                 column_config={"MAE": st.column_config.NumberColumn(format="$%d")})

    # --- Error by Price Quartile ---
    st.subheader("Error by Price Quartile")
    st.markdown("How accurate is the model across different price segments?")

    quartile_data = {
        "Quartile": ["Q1 (cheapest)", "Q2", "Q3", "Q4 (expensive)"],
        "Count": [4724, 4830, 4582, 4691],
        "MAE": [18788, 20006, 22874, 42664],
        "MAPE (%)": [4.6, 3.5, 3.3, 4.4],
        "PER10 (%)": [90.2, 96.3, 97.3, 93.0],
    }
    q_df = pd.DataFrame(quartile_data)

    fig3 = px.bar(q_df, x="Quartile", y="MAE", color="PER10 (%)",
                  color_continuous_scale="RdYlGn", text=[f"${v:,}" for v in q_df["MAE"]],
                  labels={"MAE": "MAE (SGD)"})
    fig3.update_layout(height=350)
    st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(q_df, use_container_width=True, hide_index=True,
                 column_config={"MAE": st.column_config.NumberColumn(format="$%d")})

    # --- vs Reference ---
    st.subheader("Comparison with Reference Project")

    ref_data = {
        "Metric": ["MAE", "RMSE", "MAPE", "Training Data", "Features", "MRT Distances"],
        "Reference Project": ["~$27,000", "~$39,000", "~5.7%", "232k rows (2017+)", "197", "Static"],
        "Our Model": ["$24,476", "$35,234", "3.7%", "667k rows (2000+)", "32", "Time-varying"],
    }
    ref_df = pd.DataFrame(ref_data)
    st.dataframe(ref_df, use_container_width=True, hide_index=True)

    st.success("Our model outperforms the reference project with fewer features and a more rigorous validation strategy (time-varying MRT distances, walk-forward backtesting).")
