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
        "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Odds API returned an error: {response.text}")
    return response.json()

def calculate_team_power_ratings():
    current_year = datetime.now().year
    schedule = nfl.import_schedules([current_year - 1, current_year])
    completed_games = schedule[schedule['away_score'].notna()].copy()
    
    team_stats = {}
    teams = pd.concat([completed_games['home_team'], completed_games['away_team']]).unique()
    league_avg_score = (completed_games['home_score'].mean() + completed_games['away_score'].mean()) / 2

    for team in teams:
        home_games = completed_games[completed_games['home_team'] == team]
        away_games = completed_games[completed_games['away_team'] == team]
        total_games = len(home_games) + len(away_games)
        
        if total_games == 0:
            continue
            
        points_scored = home_games['home_score'].sum() + away_games['away_score'].sum()
        points_allowed = home_games['away_score'].sum() + away_games['home_score'].sum()
        
        avg_scored = points_scored / total_games
        avg_allowed = points_allowed / total_games
        
        team_stats[team] = {
            'offense_rating': avg_scored / league_avg_score if league_avg_score > 0 else 1.0,
            'defense_rating': avg_allowed / league_avg_score if league_avg_score > 0 else 1.0
        }
        
    return team_stats, league_avg_score

def american_to_decimal(american_odds):
    if american_odds > 0:
        return (american_odds / 100.0) + 1.0
    elif american_odds < 0:
        return (100.0 / abs(american_odds)) + 1.0
    return 1.909 

def run_monte_carlo_simulations(games_data, power_ratings, league_avg_score):
    slate_projections = []
    iterations = 10000
    std_dev_nfl_score = 10.0 
    
    for game in games_data:
        home_name = game.get('home_team')
        away_name = game.get('away_team')
        home_abbr = TEAM_MAPPING.get(home_name)
        away_abbr = TEAM_MAPPING.get(away_name)
        
        if not home_abbr or not away_abbr or home_abbr not in power_ratings or away_abbr not in power_ratings:
            continue
            
        pinnacle_away_spread = 0.0
        pinnacle_away_price = -110
        pinnacle_home_spread = 0.0
        
        for bookmaker in game.get('bookmakers', []):
            if bookmaker.get('key') == 'pinnacle':
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == away_name:
                                pinnacle_away_spread = float(outcome.get('point', 0))
                                pinnacle_away_price = float(outcome.get('price', -110))
                            elif outcome.get('name') == home_name:
                                pinnacle_home_spread = float(outcome.get('point', 0))

        # Expected Matchup Scores
        exp_home_score = (power_ratings[home_abbr]['offense_rating'] * power_ratings[away_abbr]['defense_rating'] * league_avg_score) + 1.5
        exp_away_score = (power_ratings[away_abbr]['offense_rating'] * power_ratings[home_abbr]['defense_rating'] * league_avg_score)

        # Simulation Arrays
        sim_home_scores = np.random.normal(exp_home_score, std_dev_nfl_score, iterations)
        sim_away_scores = np.random.normal(exp_away_score, std_dev_nfl_score, iterations)
        
        away_covers = np.sum((sim_away_scores + pinnacle_away_spread) > sim_home_scores)
        home_covers = np.sum((sim_home_scores + pinnacle_home_spread) > sim_away_scores)
        
        model_away_cover_prob = away_covers / iterations
        model_home_cover_prob = home_covers / iterations

        # Target Identification
        if model_away_cover_prob > model_home_cover_prob:
            target_team = away_name
            target_side = "AWAY"
            target_line = f"{pinnacle_away_spread:+g}"
            win_prob = model_away_cover_prob
            odds_dec = american_to_decimal(pinnacle_away_price)
        else:
            target_team = home_name
            target_side = "HOME"
            target_line = f"{pinnacle_home_spread:+g}"
            win_prob = model_home_cover_prob
            odds_dec = american_to_decimal(-110 if pinnacle_away_price == -110 else (pinnacle_away_price * -1))

        # EV Math
        implied_prob = 1.0 / odds_dec
        ev_percent = (win_prob * odds_dec - 1.0) * 100

        slate_projections.append({
            "commence_time": game.get('commence_time'),
            "away_team": away_name,
            "home_team": home_name,
            "model_away_prob": round(model_away_cover_prob * 100, 1),
            "model_home_prob": round(model_home_cover_prob * 100, 1),
            "pinnacle_away_line": f"{away_name} {pinnacle_away_spread:+g}",
            "target_team": target_team,
            "target_side": target_side,
            "target_line": target_line,
            "recommended_bet": f"{target_team} {target_line}",
            "ev": round(ev_percent, 1)
        })
        
    return slate_projections

def save_to_dashboard(slate_projections):
    os.makedirs('data', exist_ok=True)
    file_path = 'data/upcoming_slate.json'
    with open(file_path, 'w') as f:
        json.dump(slate_projections, f, indent=4)
    print(f"Saved {len(slate_projections)} simulations to {file_path}")

if __name__ == "__main__":
    power_ratings, league_avg = calculate_team_power_ratings()
    raw_games = fetch_current_slate()
    projections = run_monte_carlo_simulations(raw_games, power_ratings, league_avg)
    save_to_dashboard(projections)
