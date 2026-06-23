import streamlit as st
import pandas as pd
from algorithm_v8_final import load_historical_callups, score_historical, time_based_validation, compute_topps_parallel_price, ensure_directories

st.set_page_config(page_title="MLB Call-Up Scorer", layout="wide")
st.title("⚾ MLB Call-Up Scorer v8")

ensure_directories()

if st.button("Run Scoring"):
    df = load_historical_callups()
    scored = score_historical(df)
    st.dataframe(scored.head(15)[["player_name", "mlb_team", "callup_score"]])

st.subheader("Card Price Estimator")
parallel = st.selectbox("Parallel", ["base", "gold_50", "superfractor_1of1"])
if st.button("Estimate"):
    row = pd.Series({'team_market_score': 9.0, 'prospect_fv': 70, 'age_at_callup': 22, 'position': 'OF'})
    price = compute_topps_parallel_price(row, parallel=parallel)
    st.success(f"Estimated price: ${price}")
