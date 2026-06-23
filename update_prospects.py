import pandas as pd
import os
from datetime import datetime

print("=== Prospects Updater ===")

base_file = "top_100_prospects_full_stats.csv"
if not os.path.exists(base_file):
    print("Creating default base CSV")
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

new_file = f"top_100_prospects_{datetime.now().strftime('%Y%m%d')}.csv"
df.to_csv(new_file, index=False)
print(f"✅ Saved {new_file}")

# Git commit (for Actions)
os.system('git add *.csv')
os.system('git commit -m "Daily prospects update" || echo "No changes"')
