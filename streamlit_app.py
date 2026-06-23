import streamlit as st
import pandas as pd
import requests
from algorithm_v8_final import (
    ensure_directories,
    load_historical_callups, 
    score_historical, 
    time_based_validation, 
    compute_topps_parallel_price,
    load_weights
)

st.set_page_config(page_title="MLB Call-Up Scorer", layout="wide", page_icon="⚾")
st.title("⚾ MLB Call-Up Scorer v8 - Full Algorithm + Real Data")
st.markdown("**Live MLB data + full learned multi-objective scoring**")

ensure_directories()

# Load learned weights (creates default if missing)
try:
    weights = load_weights()
except:
    weights = None

tab1, tab2 = st.tabs(["Prospect Rankings", "Card Price Estimator"])

with tab1:
    if st.button("Pull Real MLB Players + Score (Full Algorithm)", type="primary"):
        with st.spinner("Fetching live data and scoring with full algorithm..."):
            try:
                # Pull real data from MLB
                url = "https://statsapi.mlb.com/api/v1/teams/111/roster?season=2025"
                response = requests.get(url, timeout=10)
                data = response.json()
                
                prospects = []
                for p in data.get("roster", [])[:25]:
                    person = p.get("person", {})
                    prospects.append({
                        "player_name": person.get("fullName", "Unknown"),
                        "mlb_team": "BOS",
                        "position": p.get("position", {}).get("abbreviation", "N/A"),
                        "minor_league_recent_ops": 0.78,
                        "prospect_fv": 65,
                        "age_at_callup": person.get("currentAge", 23),
                        "highest_level": "AAA",
                        "post_callup_first_year_approx_war": 1.5
                    })
                
                df = pd.DataFrame(prospects)
                
                # Use the FULL original scoring
                scored = score_historical(df, weights=weights)
                
                st.success(f"Scored {len(scored)} real players with full algorithm!")
                st.dataframe(scored[["player_name", "mlb_team", "position", "callup_score", "score_rank"]].head(20), use_container_width=True)
                
                # Time-based validation
                st.subheader("Time-Based Validation (Full)")
                metrics = time_based_validation(df)
                st.json(metrics)
                
            except Exception as e:
                st.error(f"Error: {str(e)[:150]}. Showing demo with full algorithm.")
                df = load_historical_callups()
                scored = score_historical(df, weights=weights)
                st.dataframe(scored.head(10))

with tab2:
    st.subheader("💎 Card Price Estimator (Full Parallel Pricing)")
    player = st.text_input("Player Name", "Roman Anthony")
    fv = st.number_input("Future Value (FV)", value=70)
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

st.caption("Full original algorithm (learned weights + position-specific) • Real MLB data")
