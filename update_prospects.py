import pandas as pd
import os
from datetime import datetime

print("=== Prospects Updater ===")

# Look for the base file
base_file = "top_100_prospects_full_stats.csv"
if not os.path.exists(base_file):
    print("Base CSV not found — creating default")
    data = {
        "player_name": ["Roman Anthony", "Colson Montgomery", "Jackson Jobe"],
        "mlb_team": ["BOS", "CHW", "DET"],
        "position": ["OF", "SS", "RHP"],
        "minor_league_recent_ops": [0.85, 0.78, 0.81],
        "prospect_fv": [70, 65, 65],
        "age_at_callup": [22, 23, 22],
        "highest_level": ["AAA", "AAA", "AAA"]
    }
    pd.DataFrame(data).to_csv(base_file, index=False)

df = pd.read_csv(base_file)
print(f"Loaded {len(df)} prospects")

df["pull_date"] = datetime.now().strftime("%Y-%m-%d")
df.to_csv(f"top_100_prospects_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
print("✅ Update complete")
