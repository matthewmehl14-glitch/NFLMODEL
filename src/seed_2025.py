import os
import csv
import pandas as pd
import nfl_data_py as nfl

print("Downloading historical 2025 schedule and odds...")
schedule = nfl.import_schedules([2025])
reg_season = schedule[schedule['game_type'] == 'REG']

# Target a dedicated 2025 history file in the root folder
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
history_file = os.path.join(base_dir, 'history_2025.csv')

with open(history_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Date", "Away_Team", "Home_Team", 
        "Pinnacle_Away_ML", "Pinnacle_Home_ML", 
        "Pinnacle_Away_Spread", "Pinnacle_Home_Spread", 
        "Pinnacle_Total_Line", 
        "Actual_Away_Score", "Actual_Home_Score"
    ])
    
    added = 0
    for _, row in reg_season.iterrows():
        date = row['gameday']
        away = "LAR" if row['away_team'] == "LA" else row['away_team']
        home = "LAR" if row['home_team'] == "LA" else row['home_team']
        
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

print(f"Successfully seeded {added} games from 2025 into history_2025.csv!")
