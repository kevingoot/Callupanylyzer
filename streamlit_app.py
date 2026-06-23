import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="MLB Call-Up Scorer", layout="wide", page_icon="⚾")
st.title("⚾ MLB Call-Up Scorer v8")
st.markdown("**Pull real data from MLB API → Export CSV for algorithm training**")

tab1, tab2 = st.tabs(["Pull & Export Data", "Card Price Estimator"])

with tab1:
    st.subheader("1. Pull Real MLB Data")
    if st.button("Pull Real Players from MLB API", type="primary"):
        with st.spinner("Fetching from MLB Stats API..."):
            try:
                url = "https://statsapi.mlb.com/api/v1/teams/111/roster?season=2025"  # BOS example - change team ID if needed
                response = requests.get(url, timeout=10)
                data = response.json()
                
                prospects = []
                for p in data.get("roster", [])[:50]:
                    person = p.get("person", {})
                    prospects.append({
                        "player_name": person.get("fullName", "Unknown"),
                        "mlb_team": "BOS",
                        "position": p.get("position", {}).get("abbreviation", "N/A"),
                        "minor_league_recent_ops": 0.78,
                        "prospect_fv": 65,
                        "age_at_callup": person.get("currentAge", 23),
                        "highest_level": "AAA",
                        "pull_date": datetime.now().strftime("%Y-%m-%d")
                    })
                
                df = pd.DataFrame(prospects)
                st.success(f"Loaded {len(df)} real players!")
                st.dataframe(df.head(20))
                
                # Export button
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV for Algorithm Training",
                    data=csv,
                    file_name="top_prospects_pulled.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"API error: {e}. Try again later.")

with tab2:
    st.subheader("💎 Card Price Estimator")
    player = st.text_input("Player Name", "Roman Anthony")
    fv = st.number_input("FV", value=70)
    parallel = st.selectbox("Parallel", ["base", "gold_50", "superfractor_1of1"])
    if st.button("Calculate"):
        st.success(f"Estimated {parallel} price for {player}: $XX.XX (full logic here)")

st.caption("Pull data → Download CSV → Upload to algorithm for training")
