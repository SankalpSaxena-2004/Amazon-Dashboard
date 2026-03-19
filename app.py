
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime
from functools import lru_cache

st.set_page_config(page_title="E‑commerce Analytics (Amazon‑like)", layout="wide")

@st.cache_data
def load_data():
    customers = pd.read_csv("customers.csv", parse_dates=["signup_date"])
    products = pd.read_csv("products.csv")
    orders = pd.read_csv("orders.csv", parse_dates=["order_date"])
    items = pd.read_csv("order_items.csv")
    reviews = pd.read_csv("reviews.csv", parse_dates=["review_date"])
    return customers, products, orders, items, reviews

def compute_rfm(orders_df):
    snapshot_date = orders_df["order_date"].max() + pd.Timedelta(days=1)
    rfm = orders_df.groupby("customer_id").agg({
        "order_date": lambda x: (snapshot_date - x.max()).days,
        "order_id": "nunique",
        "order_total": "sum"
    }).rename(columns={"order_date":"recency", "order_id":"frequency", "order_total":"monetary"}).reset_index()
    rfm["r_score"] = pd.qcut(rfm["recency"], 5, labels=list("54321")).astype(int)
    rfm["f_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=list("12345")).astype(int)
    rfm["m_score"] = pd.qcut(rfm["monetary"], 5, labels=list("12345")).astype(int)
    rfm["rfm_segment"] = rfm["r_score"].astype(str)+rfm["f_score"].astype(str)+rfm["m_score"].astype(str)
    rfm["rfm_score"] = rfm[["r_score","f_score","m_score"]].sum(axis=1)
    return rfm

def cohort_retention(orders_df):
    df = orders_df.copy()
    df["order_month"] = pd.to_datetime(df["order_date"]).dt.to_period("M").dt.to_timestamp()
    first = df.groupby("customer_id")["order_month"].min().rename("cohort")
    df = df.join(first, on="customer_id")
    cohort_pivot = (
        df.groupby(["cohort", "order_month"])["customer_id"]
        .nunique()
        .rename("active_customers")
        .reset_index()
    )
    cohort_sizes = cohort_pivot.groupby("cohort")["active_customers"].first()
    cohort_pivot["period"] = (
        (cohort_pivot["order_month"].dt.year - cohort_pivot["cohort"].dt.year) * 12
        + (cohort_pivot["order_month"].dt.month - cohort_pivot["cohort"].dt.month)
    )
    retention = cohort_pivot.pivot(
        index="cohort", columns="period", values="active_customers"
    ).fillna(0)
    retention = retention.div(cohort_sizes, axis=0).round(3)
    return retention
    
def kpi_card(label, value, helptext=None):
    st.metric(label, value, help=helptext if helptext else None)

def main():
    customers, products, orders, items, reviews = load_data()

    st.title("📊 Amazon‑like E‑commerce Analytics Dashboard")

    st.sidebar.header("Filters")
    min_date, max_date = orders["order_date"].min(), orders["order_date"].max()
    date_range = st.sidebar.date_input("Order Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple):
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date, end_date = min_date, max_date

    selected_countries = st.sidebar.multiselect("Countries", sorted(orders["country"].unique().tolist()))
    cats = st.sidebar.multiselect("Categories", sorted(products["category"].unique().tolist()))
    brands = st.sidebar.multiselect("Brands", sorted(products["brand"].unique().tolist()))
    min_rating = st.sidebar.slider("Min. Review Rating (for review charts)", 1, 5, 1, 1)

    df = orders.merge(items, on="order_id").merge(products, on="product_id")
    df = df[(df["order_date"] >= start_date) & (df["order_date"] <= end_date)]
    if selected_countries:
        df = df[df["country"].isin(selected_countries)]
    if cats:
        df = df[df["category"].isin(cats)]
    if brands:
        df = df[df["brand"].isin(brands)]

    total_revenue = (df["quantity"] * df["unit_price"]).sum()
    total_orders = df["order_id"].nunique()
    total_units = df["quantity"].sum()
    unique_customers = df["customer_id"].nunique()
    aov = total_revenue / total_orders if total_orders else 0.0

    order_counts = orders.groupby("customer_id")["order_id"].nunique()
    repeat_rate = (order_counts.gt(1).sum() / len(order_counts)) if len(order_counts) else 0.0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Revenue", f"${total_revenue:,.0f}")
    with c2: kpi_card("Orders", f"{total_orders:,}")
    with c3: kpi_card("Units", f"{total_units:,}")
    with c4: kpi_card("Customers", f"{unique_customers:,}")
    with c5: kpi_card("AOV", f"${aov:,.2f}")
    with c6: kpi_card("Repeat Rate", f"{repeat_rate*100:,.1f}%")

    st.divider()

    time_series = (df.assign(line_total=df["quantity"] * df["unit_price"])
                     .groupby(pd.Grouper(key="order_date", freq="W"))["line_total"]
                     .sum().reset_index())
    fig_ts = px.line(time_series, x="order_date", y="line_total", title="Revenue Over Time (Weekly)")
    st.plotly_chart(fig_ts, use_container_width=True)

    colA, colB = st.columns((1,1))
    with colA:
        cat_mix = (df.assign(line_total=df["quantity"] * df["unit_price"])
                     .groupby("category")["line_total"].sum().reset_index().sort_values("line_total", ascending=False))
        fig_cat = px.bar(cat_mix, x="category", y="line_total", title="Category Revenue Mix", text_auto=".2s")
        st.plotly_chart(fig_cat, use_container_width=True)
    with colB:
        top_products = (df.assign(line_total=df["quantity"] * df["unit_price"])
                          .groupby(["product_id","product_name"])["line_total"]
                          .sum().reset_index().sort_values("line_total", ascending=False).head(15))
        fig_top = px.bar(top_products, x="line_total", y="product_name", orientation="h", title="Top 15 Products by Revenue", text_auto=".2s")
        st.plotly_chart(fig_top, use_container_width=True)

    colC, colD = st.columns((1,1))
    with colC:
        geo = (df.assign(line_total=df["quantity"] * df["unit_price"])
                 .groupby("country")["line_total"].sum().reset_index())
        fig_geo = px.choropleth(geo, locations="country", locationmode="country names", color="line_total",
                                title="Revenue by Country", projection="natural earth")
        st.plotly_chart(fig_geo, use_container_width=True)
    with colD:
        df_reviews = (df.merge(reviews, on=["order_id","product_id"], how="left")
                        .dropna(subset=["rating"]))
        df_reviews = df_reviews[df_reviews["rating"] >= min_rating]
        if len(df_reviews):
            by_rating = df_reviews.groupby("rating").size().reset_index(name="count")
            fig_rev = px.bar(by_rating, x="rating", y="count", title="Review Count by Rating", text_auto=True)
            st.plotly_chart(fig_rev, use_container_width=True)
        else:
            st.info("No reviews after applying filters and min rating.")

    st.divider()

    st.subheader("Customer Analytics")
    rfm = compute_rfm(orders)
    c1, c2 = st.columns((1,1))
    with c1:
        st.caption("RFM Segments (Top 10 Customers by RFM Score)")
        st.dataframe(rfm.sort_values("rfm_score", ascending=False).head(10), use_container_width=True)
    with c2:
        st.caption("Cohort Retention (Month 0..N)")
        retention = cohort_retention(orders)
        st.dataframe((retention*100).round(1), use_container_width=True)

    st.subheader("🔎 Product Drill‑Down")
    prod_sel = st.selectbox("Select a product", options=products["product_name"].tolist())
    pid = products.loc[products["product_name"] == prod_sel, "product_id"].iloc[0]
    p_df = df[df["product_id"] == pid].assign(line_total=lambda x: x["quantity"]*x["unit_price"])

    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Revenue", f"${p_df['line_total'].sum():,.0f}")
    with c2: kpi_card("Units Sold", f"{p_df['quantity'].sum():,}")
    with c3: kpi_card("Orders", f"{p_df['order_id'].nunique():,}")

    ts_p = p_df.groupby(pd.Grouper(key="order_date", freq="W"))["line_total"].sum().reset_index()
    fig_ts_p = px.line(ts_p, x="order_date", y="line_total", title=f"Revenue Over Time — {prod_sel}")
    st.plotly_chart(fig_ts_p, use_container_width=True)

    st.caption("Recent Orders")
    st.dataframe(p_df.sort_values("order_date", ascending=False)[["order_id","order_date","country","quantity","unit_price"]].head(20), use_container_width=True)

    st.divider()
    st.caption("Tip: Use the filters in the sidebar to slice the entire dashboard by country, category, brand, date range, and minimum review rating.")
    st.caption("Data is synthetic for demo purposes.")

if __name__ == "__main__":
    main()
