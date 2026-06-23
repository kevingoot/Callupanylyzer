import streamlit as st
import pandas as pd
import numpy as np
import random

st.set_page_config(page_title="Top 100 Prospects CSV Generator", page_icon="⚾")
st.title("⚾ Top 100 Prospects CSV Generator")
st.markdown("**Download real-ish top prospects data for the algorithm**")

if st.button("Generate & Download Top 100 Prospects CSV", type="primary"):
    with st.spinner("Generating realistic Top 100..."):
        # Real top prospects base
        base = [
            {"player_name": "Jesús Made", "mlb_team": "MIL", "position": "SS", "prospect_fv": 60, "age": 19},
            {"player_name": "Leo De Vries", "mlb_team": "OAK", "position": "SS", "prospect_fv": 60, "age": 19},
            {"player_name": "Colson Montgomery", "mlb_team": "CHW", "position": "SS", "prospect_fv": 65, "age": 23},
            {"player_name": "Roman Anthony", "mlb_team": "BOS", "position": "OF", "prospect_fv": 70, "age": 22},
            {"player_name": "Dylan Crews", "mlb_team": "WSN", "position": "OF", "prospect_fv": 65, "age": 23},
            {"player_name": "Jackson Jobe", "mlb_team": "DET", "position": "RHP", "prospect_fv": 65, "age": 22},
            {"player_name": "Roki Sasaki", "mlb_team": "LAD", "position": "RHP", "prospect_fv": 65, "age": 23},
        ]
        
        prospects = []
        for i in range(100):
            if i < len(base):
                p = base[i]
            else:
                p = {
                    "player_name": f"Prospect {i+1}",
                    "mlb_team": random.choice(["NYY", "LAD", "BOS", "HOU", "ATL", "PHI"]),
                    "position": random.choice(["SS", "OF", "RHP", "3B", "C"]),
                    "prospect_fv": random.randint(50, 65),
                    "age": random.randint(19, 24)
                }
            prospects.append({
                "player_name": p["player_name"],
                "mlb_team": p["mlb_team"],
                "position": p["position"],
                "minor_league_recent_ops": round(random.uniform(0.70, 0.90), 3),
                "prospect_fv": p["prospect_fv"],
                "age_at_callup": p["age"],
                "highest_level": random.choice(["AA", "AAA", "A+"]),
                "post_callup_first_year_approx_war": round(random.uniform(0.5, 3.5), 1),
                "injury_days_missed_last_2yrs": random.randint(0, 60),
                "surgery_history": random.choice([0, 0, 1]),
                "prior_mlb_stints": random.choice([0, 0, 1]),
                "defensive_tools": random.randint(45, 70),
                "team_market_score": random.randint(6, 10)
            })
        
        df = pd.DataFrame(prospects)
        csv = df.to_csv(index=False)
        
        st.success("✅ Top 100 Prospects Generated!")
        st.download_button(
            label="📥 Download top_100_prospects_2025.csv",
            data=csv,
            file_name="top_100_prospects_2025.csv",
            mime="text/csv"
        )
        
        st.dataframe(df.head(10))

st.caption("Download this CSV → Upload to algorithm for scoring & weight training")
