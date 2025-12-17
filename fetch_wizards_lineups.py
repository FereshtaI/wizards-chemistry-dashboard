import time
import requests
import pandas as pd

URL = "https://stats.nba.com/stats/leaguedashlineups"

params = {
    "Conference": "",
    "DateFrom": "",
    "DateTo": "",
    "Division": "",
    "GameSegment": "",
    "GroupQuantity": "5",
    "ISTRound": "",
    "LastNGames": "0",
    "LeagueID": "00",
    "Location": "",
    "MeasureType": "Advanced",
    "Month": "0",
    "OpponentTeamID": "0",
    "Outcome": "",
    "PORound": "0",
    "PaceAdjust": "N",
    "PerMode": "PerGame",
    "Period": "0",
    "PlusMinus": "N",
    "Rank": "N",
    "Season": "2025-26",
    "SeasonSegment": "",
    "SeasonType": "Regular Season",
    "ShotClockRange": "",
    "TeamID": "1610612764",  # Wizards
    "VsConference": "",
    "VsDivision": "",
}

# These headers are the difference-maker for stats.nba.com
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/stats/lineups/advanced",
    "Connection": "keep-alive",
}

def fetch_with_retries(session: requests.Session, tries: int = 5) -> dict:
    for attempt in range(1, tries + 1):
        try:
            print(f"Requesting NBA lineup data... (attempt {attempt}/{tries})")
            r = session.get(URL, params=params, headers=headers, timeout=90)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            print("Request failed:", e)
            if attempt == tries:
                raise
            time.sleep(2 * attempt)  # small backoff

def main():
    session = requests.Session()
    data = fetch_with_retries(session)

    result = data["resultSets"][0]
    rows = result["rowSet"]
    columns = result["headers"]

    df = pd.DataFrame(rows, columns=columns)

    out_path = "data/wizards_lineups_2025_26.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved ✅ {out_path}")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

if __name__ == "__main__":
    main()
