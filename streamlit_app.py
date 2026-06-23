import streamlit as st
import pandas as pd
import requests
from algorithm_v8_final import (
    ensure_directories,
    load_historical_callups, 
    score_historical, 
    time_based_validation, 
    compute_topps_parallel_price
)

st.set_page_config(page_title="MLB Call-Up Scorer", layout="wide", page_icon="⚾")
st.title("⚾ MLB Call-Up Scorer v8 - Full Algorithm")
st.markdown("**Live MLB data + full original algorithm**")

ensure_directories()

tab1, tab2 = st.tabs(["Prospect Rankings", "Card Price Estimator"])

with tab1:
    if st.button("Pull Real MLB Players + Score", type="primary"):
        with st.spinner("Fetching and scoring..."):
            try:
                df = load_historical_callups()
                scored = score_historical(df)
                st.success(f"Scored {len(scored)} players with full algorithm!")
                st.dataframe(scored[["player_name", "mlb_team", "position", "callup_score", "score_rank"]].head(20), use_container_width=True)
            except Exception as e:
                st.error(f"Error: {str(e)[:150]}")
                st.info("Make sure algorithm_v8_final.py has all functions.")

with tab2:
    st.subheader("💎 Card Price Estimator")
    player = st.text_input("Player Name", "Roman Anthony")
    fv = st.number_input("FV", value=70)
    parallel = st.selectbox("Parallel", ["base", "gold_50", "superfractor_1of1"])
    if st.button("Calculate"):
        row = pd.Series({'prospect_fv': fv})
        price = compute_topps_parallel_price(row, parallel)
        st.success(f"Estimated price: ${price:,.2f}")

st.caption("Full original algorithm active")
