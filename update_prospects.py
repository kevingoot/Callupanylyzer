import pandas as pd
import os
from datetime import datetime

print("=== Prospects Updater ===")

def find_or_create_base_csv():
    possible_paths = [
        "data/top_100_prospects_full_stats.csv",
        "top_100_prospects_full_stats.csv",
        "data/top_100_prospects_2025.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found base CSV at {path}")
            return pd.read_csv(path)
    
    print("No base CSV found — creating default")
    data = {
        "player_name": ["Roman Anthony", "Colson Montgomery", "Jackson Jobe"],
        "mlb_team": ["BOS", "CHW", "DET"],
        "position": ["OF", "SS", "RHP"],
        "minor_league_recent_ops": [0.85, 0.78, 0.81],
        "prospect_fv": [70, 65, 65],
        "age_at_callup": [22, 23, 22],
        "highest_level": ["AAA", "AAA", "AAA"]
    }
    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/top_100_prospects_full_stats.csv", index=False)
    return df

df = find_or_create_base_csv()
print(f"Loaded {len(df)} prospects")

df["pull_date"] = datetime.now().strftime("%Y-%m-%d")

filename = f"data/top_100_prospects_{datetime.now().strftime('%Y%m%d')}.csv"
df.to_csv(filename, index=False)
print(f"✅ Saved {filename}")
