import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/resale_transactions.csv", low_memory=False)
    df["month_dt"] = pd.to_datetime(df["month"])
    df["year"] = df["month_dt"].dt.year
    return df


def render():
    st.header("Market Insights")
    st.markdown("25 years of Singapore HDB resale transactions (2000–2026)")

    df = load_data()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        year_range = st.slider("Year Range", int(df["year"].min()), int(df["year"].max()),
                               (2015, int(df["year"].max())))
    with col2:
        selected_towns = st.multiselect("Towns (leave empty for all)", sorted(df["town"].unique()))

    mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
    if selected_towns:
        mask = mask & df["town"].isin(selected_towns)
    filtered = df[mask]

    st.metric("Total Transactions", f"{len(filtered):,}")

    # --- Price Trends ---
    st.subheader("Price Trends Over Time")

    monthly = filtered.groupby("month_dt").agg(
        median_price=("resale_price", "median"),
        count=("resale_price", "count")
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly["month_dt"], y=monthly["median_price"],
                             mode="lines", name="Median Price", line=dict(color="#1f77b4", width=2)))
    fig.add_trace(go.Bar(x=monthly["month_dt"], y=monthly["count"],
                         name="Volume", yaxis="y2", opacity=0.3, marker_color="gray"))

    # Cooling measure annotations
    annotations = [
        ("2013-01", "TDSR + ABSD hike"),
        ("2018-07", "ABSD raised"),
        ("2021-12", "ABSD raised again"),
    ]
    for date, label in annotations:
        dt = pd.Timestamp(date)
        if year_range[0] <= dt.year <= year_range[1]:
            fig.add_vline(x=dt, line_dash="dot", line_color="red", opacity=0.5)
            fig.add_annotation(x=dt, y=monthly["median_price"].max(), text=label,
                               showarrow=False, yshift=10, font=dict(size=10, color="red"))

    fig.update_layout(
        yaxis=dict(title="Median Price (SGD)"),
        yaxis2=dict(title="Monthly Volume", overlaying="y", side="right"),
        height=450, margin=dict(t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Price by Town ---
    st.subheader("Median Price by Town")

    town_median = filtered.groupby("town")["resale_price"].median().sort_values(ascending=True).reset_index()

    fig2 = px.bar(town_median, x="resale_price", y="town", orientation="h",
                  color="resale_price", color_continuous_scale="RdYlGn_r",
                  labels={"resale_price": "Median Price (SGD)", "town": ""})
    fig2.update_layout(height=600, margin=dict(l=150), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

    # --- Price by Flat Type ---
    st.subheader("Price Distribution by Flat Type")

    fig3 = px.box(filtered, x="flat_type", y="resale_price",
                  category_orders={"flat_type": ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM",
                                                  "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]},
                  color="flat_type", labels={"resale_price": "Resale Price (SGD)", "flat_type": "Flat Type"})
    fig3.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig3, use_container_width=True)

    # --- YoY Change ---
    st.subheader("Year-over-Year Price Change")

    yearly = filtered.groupby("year")["resale_price"].median().reset_index()
    yearly["yoy_pct"] = yearly["resale_price"].pct_change() * 100

    colors = ["green" if v > 0 else "red" for v in yearly["yoy_pct"].fillna(0)]
    fig4 = go.Figure(go.Bar(x=yearly["year"], y=yearly["yoy_pct"], marker_color=colors))
    fig4.update_layout(yaxis_title="Change (%)", height=350, margin=dict(t=20))
    fig4.add_hline(y=0, line_color="black", line_width=0.5)
    st.plotly_chart(fig4, use_container_width=True)

    # --- Key Stats ---
    st.subheader("Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Median Price", f"${filtered['resale_price'].median():,.0f}")
    col2.metric("Mean Price", f"${filtered['resale_price'].mean():,.0f}")
    col3.metric("Highest Sale", f"${filtered['resale_price'].max():,.0f}")
    col4.metric("Avg Floor Area", f"{filtered['floor_area_sqm'].mean():.0f} sqm")
