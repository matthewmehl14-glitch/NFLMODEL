# src/grade_backtest.py

import os
import requests
import pandas as pd
import json

def grade_bets():
    csv_file = 'data/historical_log.csv'
    
    if not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0:
        print("No historical log found or file is empty. Skipping grading.")
        update_roi_dashboard(pd.DataFrame())
        return
        
    try:
        df = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError:
        print("CSV is empty despite size check. Skipping grading.")
        update_roi_dashboard(pd.DataFrame())
        return

    if df.empty or 'status' not in df.columns:
        print("No valid bet columns. Skipping grading.")
        update_roi_dashboard(df)
        return

    pending_mask = df['status'] == 'Pending'
    if not pending_mask.any():
        print("No pending bets to grade.")
        update_roi_dashboard(df)
        return
        
    api_key = os.environ.get("ODDS_API_KEY")
    
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/scores"
    params = {"apiKey": api_key, "daysFrom": 3}
    res = requests.get(url, params=params)
    scores_data = res.json()
    
    completed_games = {}
    for game in scores_data:
        if game.get('completed'):
            scores = game.get('scores')
            if scores:
                home_score = next((float(s['score']) for s in scores if s['name'] == game['home_team']), 0)
                away_score = next((float(s['score']) for s in scores if s['name'] == game['away_team']), 0)
                completed_games[game['id']] = {
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                    'home_score': home_score,
                    'away_score': away_score
                }

    for idx, row in df[pending_mask].iterrows():
        game_id = row['game_id']
        if game_id in completed_games:
            c_game = completed_games[game_id]
            target_team = row['target_team']
            target_line = float(row['target_line'])
            stake = float(row['stake'])
            odds_dec = float(row['odds_decimal'])
            
            if target_team == c_game['home_team']:
                target_score = c_game['home_score']
                opp_score = c_game['away_score']
            else:
                target_score = c_game['away_score']
                opp_score = c_game['home_score']
                
            margin = target_score - opp_score + target_line
            
            if margin > 0:
                df.at[idx, 'status'] = 'Win'
                df.at[idx, 'profit_loss'] = round(stake * (odds_dec - 1.0), 2)
            elif margin < 0:
                df.at[idx, 'status'] = 'Loss'
                df.at[idx, 'profit_loss'] = -stake
            else:
                df.at[idx, 'status'] = 'Push'
                df.at[idx, 'profit_loss'] = 0.0

    df.to_csv(csv_file, index=False)
    update_roi_dashboard(df)

def update_roi_dashboard(df):
    if df.empty or 'status' not in df.columns:
        stats = {"total_bets": 0, "wins": 0, "losses": 0, "pushes": 0, "roi": 0.0, "profit": 0.0}
    else:
        graded = df[df['status'].isin(['Win', 'Loss', 'Push'])]
        total_bets = len(graded)
        
        if total_bets == 0:
            stats = {"total_bets": 0, "wins": 0, "losses": 0, "pushes": 0, "roi": 0.0, "profit": 0.0}
        else:
            wins = len(graded[graded['status'] == 'Win'])
            losses = len(graded[graded['status'] == 'Loss'])
            pushes = len(graded[graded['status'] == 'Push'])
            total_profit = graded['profit_loss'].sum()
            total_staked = graded['stake'].sum()
            roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0.0
            
            stats = {
                "total_bets": total_bets,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "roi": round(roi, 2),
                "profit": round(total_profit, 2)
            }
        
    os.makedirs('data', exist_ok=True)
    with open('data/roi_stats.json', 'w') as f:
        json.dump(stats, f)

if __name__ == "__main__":
    grade_bets()
