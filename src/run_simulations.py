import os
import json
import requests
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime, timedelta, timezone

TEAM_MAPPING = {
    'Arizona Cardinals': 'ARI', 'Atlanta Falcons': 'ATL', 'Baltimore Ravens': 'BAL',
    'Buffalo Bills': 'BUF', 'Carolina Panthers': 'CAR', 'Chicago Bears': 'CHI',
    'Cincinnati Bengals': 'CIN', 'Cleveland Browns': 'CLE', 'Dallas Cowboys': 'DAL',
    'Denver Broncos': 'DEN', 'Detroit Lions': 'DET', 'Green Bay Packers': 'GB',
    'Houston Texans': 'HOU', 'Indianapolis Colts': 'IND', 'Jacksonville Jaguars': 'JAX',
    'Kansas City Chiefs': 'KC', 'Las Vegas Raiders': 'LV', 'Los Angeles Chargers': 'LAC',
    'Los Angeles Rams': 'LA', 'Miami Dolphins': 'MIA', 'Minnesota Vikings': 'MIN',
    'New England Patriots': 'NE', 'New Orleans Saints': 'NO', 'New York Giants': 'NYG',
    'New York Jets': 'NYJ', 'Philadelphia Eagles': 'PHI', 'Pittsburgh Steelers': 'PIT',
    'San Francisco 49ers': 'SF', 'Seattle Seahawks': 'SEA', 'Tampa Bay Buccaneers': 'TB',
    'Tennessee Titans': 'TEN', 'Washington Commanders': 'WAS'
}

def fetch_current_slate():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY is missing.")

    now = datetime.now(timezone.utc)
    end_date = now + timedelta(days=5)

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",
        "bookmakers": "pinnacle",
        "markets": "spreads,h2h",
        "oddsFormat": "decimal",
        "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Odds API returned an error: {response.text}")
    return response.json()

def calculate_advanced_metrics():
    print("Downloading NFLverse play-by-play database (this takes a minute)...")
    current_year = datetime.now().year
    years_to_pull = [current_year - 1, current_year]
    
    pbp = nfl.import_pbp_data(years_to_pull)
    plays = pbp[(pbp['play_type'].isin(['pass', 'run'])) & (pbp['epa'].notna())].copy()
    
    league_epa_per_play = plays['epa'].mean()
    
    off_stats = plays.groupby('posteam').agg(
        off_epa_per_play=('epa', 'mean'),
        total_plays=('play_id', 'count'),
        games_played=('game_id', 'nunique')
    ).reset_index()
    off_stats['off_plays_per_game'] = off_stats['total_plays'] / off_stats['games_played']
    
    def_games = plays.groupby('defteam')['game_id'].nunique()
    def_stats = plays.groupby('defteam').agg(
        def_epa_per_play=('epa', 'mean'),
        total_def_plays=('play_id', 'count')
    ).reset_index()
    def_stats['def_plays_per_game'] = def_stats['total_def_plays'] / def_stats['defteam'].map(def_games)

    scores = nfl.import_schedules(years_to_pull)
    completed = scores[scores['away_score'].notna()].copy()
    league_avg_points = (completed['home_score'].mean() + completed['away_score'].mean()) / 2

    team_metrics = {}
    teams = off_stats['posteam'].unique()
    
    for t in teams:
        off = off_stats[off_stats['posteam'] == t].iloc[0]
        defense = def_stats[def_stats['defteam'] == t].iloc[0]
        
        team_metrics[t] = {
            'off_epa_edge': off['off_epa_per_play'] - league_epa_per_play,
            'def_epa_edge': defense['def_epa_per_play'] - league_epa_per_play,
            'pace': (off['off_plays_per_game'] + defense['def_plays_per_game']) / 2
        }
        
    return team_metrics, league_avg_points

def run_monte_carlo_simulations(games_data, team_metrics, league_avg_points):
    slate_projections = []
    iterations = 10000
    
    # NFL games have high variance; standard deviation for a team's score is roughly 10 points.
    std_dev_nfl = 10.0 
    # Regression factor to stabilize noisy early-season EPA data
    regression_factor = 0.5 
    
    for game in games_data:
        home_name = game.get('home_team')
        away_name = game.get('away_team')
        home_abbr = TEAM_MAPPING.get(home_name)
        away_abbr = TEAM_MAPPING.get(away_name)
        
        if not home_abbr or not away_abbr or home_abbr not in team_metrics or away_abbr not in team_metrics:
            continue
            
        pinnacle_away_spread = 0.0
        pinnacle_away_price = 1.909 
        pinnacle_home_spread = 0.0
        pinnacle_home_price = 1.909 
        
        for bookmaker in game.get('bookmakers', []):
            if bookmaker.get('key') == 'pinnacle':
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == away_name:
                                pinnacle_away_spread = float(outcome.get('point', 0))
                                pinnacle_away_price = float(outcome.get('price', 1.909))
                            elif outcome.get('name') == home_name:
                                pinnacle_home_spread = float(outcome.get('point', 0))
                                pinnacle_home_price = float(outcome.get('price', 1.909))

        home_plays = (team_metrics[home_abbr]['pace'] + team_metrics[away_abbr]['pace']) / 2
        away_plays = home_plays 

        # Apply the regression factor to the EPA edge to tame the projections
        exp_home_score = league_avg_points + ((team_metrics[home_abbr]['off_epa_edge'] - team_metrics[away_abbr]['def_epa_edge']) * home_plays * regression_factor) + 1.5
        exp_away_score = league_avg_points + ((team_metrics[away_abbr]['off_epa_edge'] - team_metrics[home_abbr]['def_epa_edge']) * away_plays * regression_factor)
        
        exp_home_score = max(exp_home_score, 0.1)
        exp_away_score = max(exp_away_score, 0.1)

        # Revert to a Normal Distribution to properly capture NFL score variance
        sim_home_scores = np.random.normal(exp_home_score, std_dev_nfl, iterations)
        sim_away_scores = np.random.normal(exp_away_score, std_dev_nfl, iterations)
        
        away_covers = np.sum((sim_away_scores + pinnacle_away_spread) > sim_home_scores)
        home_covers = np.sum((sim_home_scores + pinnacle_home_spread) > sim_away_scores)
        
        model_away_cover_prob = away_covers / iterations
        model_home_cover_prob = home_covers / iterations

        if model_away_cover_prob > model_home_cover_prob:
            target_team = away_name
            target_side = "AWAY"
            target_line = f"{pinnacle_away_spread:+g}"
            win_prob = model_away_cover_prob
            odds_dec = pinnacle_away_price
        else:
            target_team = home_name
            target_side = "HOME"
            target_line = f"{pinnacle_home_spread:+g}"
            win_prob = model_home_cover_prob
            odds_dec = pinnacle_home_price

        ev_percent = (win_prob * odds_dec - 1.0) * 100

        slate_projections.append({
            "id": game.get('id'),
            "commence_time": game.get('commence_time'),
            "away_team": away_name,
            "home_team": home_name,
            "model_away_prob": round(model_away_cover_prob * 100, 1),
            "model_home_prob": round(model_home_cover_prob * 100, 1),
            "pinnacle_away_line": f"{away_name} {pinnacle_away_spread:+g}",
            "target_team": target_team,
            "target_side": target_side,
            "target_line": target_line,
            "odds_dec": odds_dec,
            "recommended_bet": f"{target_team} {target_line}",
            "ev": round(ev_percent, 1)
        })
        
    return slate_projections

def save_to_dashboard(slate_projections):
    os.makedirs('data', exist_ok=True)
    file_path = 'data/upcoming_slate.json'
    with open(file_path, 'w') as f:
        json.dump(slate_projections, f, indent=4)
        
    log_file = 'data/historical_log.csv'
    bets_to_log = [p for p in slate_projections if p['ev'] > 0]
    
    if bets_to_log:
        new_rows = []
        for p in bets_to_log:
            new_rows.append({
                'game_id': p['id'],
                'commence_time': p['commence_time'],
                'target_team': p['target_team'],
                'target_line': p['target_line'],
                'odds_decimal': p['odds_dec'],
                'stake': 25.0, 
                'ev': p['ev'],
                'status': 'Pending',
                'profit_loss': 0.0
            })
        df_new = pd.DataFrame(new_rows)
        
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
            try:
                df_existing = pd.read_csv(log_file)
                existing_ids = df_existing['game_id'].tolist() if 'game_id' in df_existing.columns else []
                df_new = df_new[~df_new['game_id'].isin(existing_ids)]
                if not df_new.empty:
                    pd.concat([df_existing, df_new], ignore_index=True).to_csv(log_file, index=False)
            except pd.errors.EmptyDataError:
                df_new.to_csv(log_file, index=False)
        else:
            df_new.to_csv(log_file, index=False)
            
    print(f"Saved {len(slate_projections)} simulations and logged new bets.")

if __name__ == "__main__":
    team_metrics, league_avg = calculate_advanced_metrics()
    raw_games = fetch_current_slate()
    projections = run_monte_carlo_simulations(raw_games, team_metrics, league_avg)
    save_to_dashboard(projections)
