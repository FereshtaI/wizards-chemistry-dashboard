import json
import pandas as pd

IN_PATH = "data/wizards_lineups_2025_26.json"
OUT_PATH = "data/wizards_lineups_2025_26.csv"

with open(IN_PATH, "r") as f:
    data = json.load(f)

headers = data["resultSets"][0]["headers"]
rows = data["resultSets"][0]["rowSet"]

df = pd.DataFrame(rows, columns=headers)
df.to_csv(OUT_PATH, index=False)

print("Saved ✅", OUT_PATH)
print("Rows:", len(df), "Cols:", len(df.columns))
