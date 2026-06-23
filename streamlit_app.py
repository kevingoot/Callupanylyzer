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
st.title("⚾ MLB Call-Up Scorer v8")
st.markdown("**Real-time prospect scoring with live MLB API**")

ensure_directories()

tab1, tab2 = st.tabs(["Prospect Rankings", "Card Price Estimator"])

with tab1:
    if st.button("Pull Live MLB Data + Score", type="primary"):
        with st.spinner("Fetching from MLB API..."):
            try:
                df = load_historical_callups()
                scored = score_historical(df)
                st.success(f"Loaded {len(df)} prospects")
                st.dataframe(scored[["player_name", "mlb_team", "position", "callup_score", "score_rank"]].head(20), use_container_width=True)
                
                st.subheader("Time-Based Validation")
                metrics = time_based_validation(df)
                st.json(metrics)
            except Exception as e:
                st.error(f"Error: {str(e)[:200]}... Using fallback data.")
                df = load_historical_callups()
                scored = score_historical(df)
                st.dataframe(scored.head(10))

with tab2:
    st.subheader("💎 Card Price Estimator")
    col1, col2 = st.columns(2)
    with col1:
        player = st.text_input("Player Name", "Roman Anthony")
        fv = st.number_input("Future Value (FV)", value=70, min_value=40, max_value=80)
    with col2:
        parallel = st.selectbox("Parallel Type", ["base", "gold_50", "superfractor_1of1"])
    
    if st.button("Calculate Price", type="primary"):
        row = pd.Series({
            'team_market_score': 9.0, 
            'prospect_fv': fv, 
            'age_at_callup': 22,
            'minor_league_recent_ops': 0.82, 
            'position': 'OF',
            'player_name': player
        })
        price = compute_topps_parallel_price(row, parallel=parallel)
        st.success(f"**Estimated {parallel} price for {player}: ${price:,.2f}**")

st.caption("Full original algorithm • Live MLB data pull")
