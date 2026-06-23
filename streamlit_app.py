import streamlit as st
import pandas as pd
import requests
from io import StringIO
from algorithm_v8_final import load_historical_callups, score_historical, time_based_validation, compute_topps_parallel_price, ensure_directories

st.set_page_config(page_title="MLB Call-Up Scorer", layout="wide", page_icon="⚾")
st.title("⚾ MLB Call-Up Scorer v8")
st.markdown("**Real-time prospect scoring with live MLB API**")

ensure_directories()

tab1, tab2 = st.tabs(["Prospect Rankings", "Card Price Estimator"])

with tab1:
    if st.button("Pull Live MLB Data + Score", type="primary"):
        with st.spinner("Fetching from MLB Stats API..."):
            try:
                # MLB Stats API - Recent prospects / callups example
                url = "https://statsapi.mlb.com/api/v1/people?season=2025&hydrate=stats(group=hitting,season=2025,type=season)"
                response = requests.get(url, timeout=10)
                data = response.json()
                
                # Simplified parsing - expand as needed
                prospects = []
                for person in data.get("people", [])[:20]:  # limit for speed
                    prospects.append({
                        "player_name": person.get("fullName", "Unknown"),
                        "mlb_team": person.get("currentTeam", {}).get("name", "N/A"),
                        "position": person.get("primaryPosition", {}).get("abbreviation", "N/A"),
                        "minor_league_recent_ops": 0.78,  # placeholder - enhance later
                        "prospect_fv": 60,
                        "age_at_callup": person.get("currentAge", 22),
                        "highest_level": "AAA",
                        "post_callup_first_year_approx_war": 1.5
                    })
                
                df = pd.DataFrame(prospects)
                st.success(f"Loaded {len(df)} players from MLB API")
                
                scored = score_historical(df)
                st.dataframe(scored[["player_name", "mlb_team", "position", "callup_score", "score_rank"]].head(15), use_container_width=True)
                
                st.subheader("Time-Based Validation")
                metrics = time_based_validation(df)
                st.json(metrics)
            except Exception as e:
                st.error(f"API error: {e}. Using local data instead.")
                df = load_historical_callups()
                scored = score_historical(df)
                st.dataframe(scored.head(10))

with tab2:
    st.subheader("💎 Card Price Estimator")
    col1, col2 = st.columns(2)
    with col1:
        player = st.text_input("Player Name", "Roman Anthony")
        fv = st.number_input("Future Value (FV)", value=70)
    with col2:
        parallel = st.selectbox("Parallel", ["base", "gold_50", "superfractor_1of1"])
    
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

st.caption("Live MLB API • GitHub updates")
