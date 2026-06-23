import pandas as pd
import requests
import os
from datetime import datetime

print("=== Prospects Updater ===")

def get_fangraphs_top_100():
    url = "https://www.fangraphs.com/prospects/the-board"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    html = requests.get(url, headers=headers, timeout=30).text

    tables = pd.read_html(html)

    for table in tables:
        cols = [str(c) for c in table.columns]

        if (
            any("Top 100 Rank" in c for c in cols)
            and "Name Name" in cols
        ):
            df = table.copy()

            df.columns = [
                "rank" if "Top 100 Rank" in str(c) else
                "player_name" if "Name Name" in str(c) else
                "mlb_team" if "Org MLB Organization" in str(c) else
                "position" if "Pos" in str(c) else
                "eta" if "ETA" in str(c) else
                "prospect_fv" if "FV" in str(c) else
                str(c)
                for c in df.columns
            ]

            return df

    raise Exception("FanGraphs Top 100 table not found")

# Pull latest rankings
df = get_fangraphs_top_100()

print(f"Loaded {len(df)} prospects")

# Add pull date
df["pull_date"] = datetime.now().strftime("%Y-%m-%d")

# Save master file
df.to_csv("top_100_prospects_full_stats.csv", index=False)

# Save dated snapshot
new_file = f"top_100_prospects_{datetime.now():%Y%m%d}.csv"
df.to_csv(new_file, index=False)

print(f"✅ Saved {new_file}")

# Commit changes
os.system("git add *.csv")
os.system('git commit -m "Daily prospects update" || echo "No changes"')
