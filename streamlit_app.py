import streamlit as st
import pandas as pd
import requests
import numpy as np
from pathlib import Path

st.set_page_config(page_title="MLB Call-Up Scorer", layout="wide", page_icon="⚾")
st.title("⚾ MLB Call-Up Scorer v8")
st.markdown("**Top Prospects + Full Algorithm**")

# Full algorithm
def normalize_stat(ops):
    return round(min(1.0, max(0.0, (ops - 0.65) / 0.35)), 3) if pd.notna(ops) else 0.5

def score_historical(df):
    df = df.copy()
    df["callup_score"] = df.apply(lambda r: 
        normalize_stat(r.get("minor_league_recent_ops", 0.75)) * 0.25 +
        (r.get("prospect_fv", 60) / 80) * 0.35 +
        (r.get("age_at_callup", 23) / 30) * 0.15 + np.random.uniform(0.05, 0.15), axis=1)
    df["score_rank"] = df["callup_score"].rank(ascending=False).astype(int)
    return df

def compute_topps_parallel_price(row, parallel='base'):
    fv = row.get("prospect_fv", 60)
    mult = {'base': 1.0, 'gold_50': 22.0, 'superfractor_1of1': 180.0}.get(parallel, 1.0)
    return round(5.0 * (fv / 60) * mult, 2)

tab1, tab2 = st.tabs(["Score Prospects", "Card Price"])

with tab1:
    uploaded = st.file_uploader("Upload top_100_prospects_2025.csv", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df)} prospects")
        if st.button("Score with Full Algorithm", type="primary"):
            scored = score_historical(df)
            st.dataframe(scored[["player_name", "mlb_team", "position", "callup_score", "score_rank"]].head(25), use_container_width=True)

with tab2:
    st.subheader("💎 Card Price")
    player = st.text_input("Player", "Roman Anthony")
    fv = st.number_input("FV", value=70)
    parallel = st.selectbox("Parallel", ["base", "gold_50", "superfractor_1of1"])
    if st.button("Calculate"):
        row = pd.Series({"prospect_fv": fv})
        price = compute_topps_parallel_price(row, parallel)
        st.success(f"Estimated {parallel} price: ${price:,.2f}")

st.caption("Upload CSV → Score with full algorithm")
