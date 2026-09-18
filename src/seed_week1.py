import os
import csv
import pandas as pd
import nfl_data_py as nfl

print("Downloading historical 2026 schedule and odds...")
schedule = nfl.import_schedules([2026])
week1 = schedule[(schedule['week'] == 1) & (schedule['game_type'] == 'REG')]

# Make sure we target the root directory history.csv
history_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'history.csv')

if not os.path.exists(history_file):
    with open(history_file, 'w', newline='') as f:
        csv.writer(f).writerow([
            "Date", "Away_Team", "Home_Team", 
            "Pinnacle_Away_ML", "Pinnacle_Home_ML", 
            "Pinnacle_Away_Spread", "Pinnacle_Home_Spread", 
            "Pinnacle_Total_Line", 
            "Actual_Away_Score", "Actual_Home_Score"
        ])

existing = pd.read_csv(history_file)
existing_keys = set(zip(existing['Date'], existing['Away_Team'], existing['Home_Team']))

added = 0
with open(history_file, 'a', newline='') as f:
    writer = csv.writer(f)
    for _, row in week1.iterrows():
        date = row['gameday']
        away = "LAR" if row['away_team'] == "LA" else row['away_team']
        home = "LAR" if row['home_team'] == "LA" else row['home_team']
        
        if (date, away, home) in existing_keys: continue
            
        home_spread = row['spread_line']
        away_spread = -home_spread if pd.notna(home_spread) else "N/A"
        home_spread = home_spread if pd.notna(home_spread) else "N/A"
        away_ml = row.get('away_moneyline', "N/A")
        home_ml = row.get('home_moneyline', "N/A")
        away_ml = away_ml if pd.notna(away_ml) else "N/A"
        home_ml = home_ml if pd.notna(home_ml) else "N/A"
        total = row['total_line'] if pd.notna(row['total_line']) else "N/A"
        away_score = row['away_score'] if pd.notna(row['away_score']) else "N/A"
        home_score = row['home_score'] if pd.notna(row['home_score']) else "N/A"
        
        writer.writerow([date, away, home, away_ml, home_ml, away_spread, home_spread, total, away_score, home_score])
        added += 1

print(f"Successfully seeded {added} Week 1 games into history.csv!")
