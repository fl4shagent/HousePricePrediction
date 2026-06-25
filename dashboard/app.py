import streamlit as st

st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏠",
    layout="wide",
)

st.title("Singapore HDB Resale Price Dashboard")
st.markdown("Explore 25 years of HDB resale transactions, predict prices, and understand what drives property values.")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Market Insights", "Price Predictor", "Model Performance"])

if page == "Market Insights":
    from pages import market_insights
    market_insights.render()
elif page == "Price Predictor":
    from pages import price_predictor
    price_predictor.render()
elif page == "Model Performance":
    from pages import model_performance
    model_performance.render()
