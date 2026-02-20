"""
scrape.py
Columbus Crew xG Dashboard — 2025 MLS Season
-------------------------------------------------
Pulls xG data directly from FBref using pandas.read_html().
No ScraperFC needed — simpler and more reliable.

Run from the project root:
    python python/scrape.py
"""

import os
import json
import time
import requests
import pandas as pd

# ── Output paths ──────────────────────────────────────────
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
DASHBOARD_DIR = "dashboard/data"

for path in [RAW_DIR, PROCESSED_DIR, DASHBOARD_DIR]:
    os.makedirs(path, exist_ok=True)

# FBref blocks default Python user-agent, so we set a browser one
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_fbref_table(url, table_index=0):
    """Fetch a table from an FBref page, handling their HTML comment wrapping."""
    print(f"  Fetching: {url}")
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()

    # FBref wraps some tables in HTML comments — strip them so pandas can parse
    html = response.text.replace("<!--", "").replace("-->", "")

    tables = pd.read_html(html, header=1)
    df = tables[table_index].copy()

    # Drop repeated header rows FBref inserts mid-table
    if "Date" in df.columns:
        df = df[df["Date"] != "Date"].dropna(subset=["Date"])

    return df.reset_index(drop=True)


# ── 1. Player shooting stats (xG per player) ─────────────
print("\n[1/3] Fetching Columbus Crew player shooting stats...")
time.sleep(4)  # be polite to FBref — wait between requests

PLAYER_SHOOTING_URL = (
    "https://fbref.com/en/squads/529ba333/2024-2025/"
    "shooting/Columbus-Crew-Shooting"
)

try:
    player_df = fetch_fbref_table(PLAYER_SHOOTING_URL)
    player_df.to_csv(os.path.join(RAW_DIR, "crew_player_shooting_raw.csv"), index=False)
    print(f"  -> {len(player_df)} player rows saved")
    print(f"  -> Columns: {list(player_df.columns)}")
except Exception as e:
    print(f"  ERROR: {e}")
    player_df = pd.DataFrame()


# ── 2. Match log — shooting/xG per game ──────────────────
print("\n[2/3] Fetching Columbus Crew match log (xG per game)...")
time.sleep(4)

MATCH_LOG_URL = (
    "https://fbref.com/en/squads/529ba333/2024-2025/"
    "matchlogs/c22/shooting/Columbus-Crew-Match-Logs-Shooting"
)

try:
    match_df = fetch_fbref_table(MATCH_LOG_URL)
    match_df.to_csv(os.path.join(RAW_DIR, "crew_match_log_raw.csv"), index=False)
    print(f"  -> {len(match_df)} match rows saved")
    print(f"  -> Columns: {list(match_df.columns)}")
except Exception as e:
    print(f"  ERROR: {e}")
    match_df = pd.DataFrame()


# ── 3. Process match log ──────────────────────────────────
print("\n[3/3] Processing data...")

if not match_df.empty:
    # Flatten multi-level columns if present
    if isinstance(match_df.columns, pd.MultiIndex):
        match_df.columns = [" ".join(c).strip() for c in match_df.columns]

    print(f"  Columns after flatten: {list(match_df.columns)}")

    # Rename common FBref column names to our standard names
    rename_map = {}
    for col in match_df.columns:
        cl = col.lower().strip()
        if cl == "date":                                      rename_map[col] = "date"
        elif cl in ["opponent", "opp"]:                       rename_map[col] = "opponent"
        elif cl == "result":                                  rename_map[col] = "result"
        elif cl in ["gf", "for"]:                             rename_map[col] = "goals"
        elif cl in ["ga", "against"]:                         rename_map[col] = "goals_allowed"
        elif "xg" in cl and "xga" not in cl and "np" not in cl and "ps" not in cl:
            rename_map[col] = "xg"
        elif "xga" in cl or ("xg" in cl and "against" in cl):
            rename_map[col] = "xga"

    match_df = match_df.rename(columns=rename_map)

    # Coerce numeric columns
    for col in ["xg", "xga", "goals", "goals_allowed"]:
        if col in match_df.columns:
            match_df[col] = pd.to_numeric(match_df[col], errors="coerce")

    match_df = match_df.sort_values("date").reset_index(drop=True)

    # Cumulative totals for the trend chart
    for col, cum_col in [("xg", "cumulative_xg"), ("xga", "cumulative_xga"), ("goals", "cumulative_goals")]:
        if col in match_df.columns:
            match_df[cum_col] = match_df[col].cumsum()

    if "xg" in match_df.columns and "xga" in match_df.columns:
        match_df["xg_diff_match"] = match_df["xg"] - match_df["xga"]

    match_df.to_csv(os.path.join(PROCESSED_DIR, "crew_match_xg.csv"), index=False)
    print(f"  -> Saved: {PROCESSED_DIR}/crew_match_xg.csv")


# ── 4. Process player stats ───────────────────────────────
if not player_df.empty:
    if isinstance(player_df.columns, pd.MultiIndex):
        player_df.columns = [" ".join(c).strip() for c in player_df.columns]

    # Detect xG and goals columns
    xg_col   = next((c for c in player_df.columns if "xg" in c.lower()
                     and "xga" not in c.lower() and "np" not in c.lower()), None)
    goal_col = next((c for c in player_df.columns if c.lower() in ["gls", "goals", "g"]), None)

    print(f"  xG column   : {xg_col}")
    print(f"  Goals column: {goal_col}")

    if xg_col and goal_col:
        player_df[xg_col]   = pd.to_numeric(player_df[xg_col],   errors="coerce")
        player_df[goal_col] = pd.to_numeric(player_df[goal_col], errors="coerce")
        player_df["xg_diff"] = player_df[goal_col] - player_df[xg_col]

        def finishing_label(diff):
            if pd.isna(diff):  return "N/A"
            if diff >  0.5:    return "Overperforming"
            if diff < -0.5:    return "Underperforming"
            return "On Track"

        player_df["finishing"] = player_df["xg_diff"].apply(finishing_label)

    # Drop rows with no player name (FBref sometimes has blank rows)
    player_col = next((c for c in player_df.columns if "player" in c.lower()), None)
    if player_col:
        player_df = player_df.dropna(subset=[player_col])

    player_df.to_csv(os.path.join(PROCESSED_DIR, "crew_player_xg.csv"), index=False)
    print(f"  -> Saved: {PROCESSED_DIR}/crew_player_xg.csv")


# ── 5. Export JSON for the JS dashboard ───────────────────
print("\nExporting JSON files for dashboard...")

if not match_df.empty:
    export_cols = [c for c in [
        "date", "opponent", "xg", "xga", "goals", "goals_allowed",
        "result", "cumulative_xg", "cumulative_xga",
        "cumulative_goals", "xg_diff_match"
    ] if c in match_df.columns]

    with open(os.path.join(DASHBOARD_DIR, "match_xg.json"), "w") as f:
        json.dump(match_df[export_cols].to_dict(orient="records"), f, indent=2, default=str)
    print(f"  -> Saved: {DASHBOARD_DIR}/match_xg.json")

if not player_df.empty:
    player_export = [c for c in player_df.columns
                     if c in ["Player", "Pos", "90s", "Sh", goal_col, xg_col, "xg_diff", "finishing"]]
    with open(os.path.join(DASHBOARD_DIR, "player_xg.json"), "w") as f:
        json.dump(player_df[player_export].to_dict(orient="records"), f, indent=2, default=str)
    print(f"  -> Saved: {DASHBOARD_DIR}/player_xg.json")


# ── 6. Season summary ─────────────────────────────────────
print("\n-- Season Summary -------------------------------------------")
if not match_df.empty and "xg" in match_df.columns:
    print(f"  Matches     : {len(match_df)}")
    print(f"  xG For      : {match_df['xg'].sum():.2f}")
    if "xga"   in match_df.columns: print(f"  xG Against  : {match_df['xga'].sum():.2f}")
    if "goals" in match_df.columns: print(f"  Actual Goals: {int(match_df['goals'].sum())}")
    if "xg_diff_match" in match_df.columns:
        print(f"  xG Diff     : {match_df['xg_diff_match'].sum():.2f}")
    if "result" in match_df.columns:
        r = match_df["result"]
        print(f"  Record      : {r.str.startswith('W').sum()}W "
              f"{r.str.startswith('D').sum()}D "
              f"{r.str.startswith('L').sum()}L")
else:
    print("  Match log unavailable — check data/raw/ for the raw HTML")
    print("  If FBref blocked you, wait 10 min and try again")

print("-------------------------------------------------------------")
print("\nDone! Open data/processed/ and dashboard/data/ to see your files.")