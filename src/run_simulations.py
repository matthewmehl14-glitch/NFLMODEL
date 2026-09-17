import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

def fetch_current_slate():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY environment variable is missing.")

    # Define the week window: 'Now' (Thursday) through the next 5 days (Tuesday morning UTC)
    # This captures Thursday Night through Monday Night Football, ignoring the next week.
    now = datetime.now(timezone.utc)
    end_date = now + timedelta(days=5)

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu", # Pinnacle is classified under the EU region
        "bookmakers": "pinnacle",
        "markets": "h2h,spreads,totals",
        "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    print(f"Fetching games from {params['commenceTimeFrom']} to {params['commenceTimeTo']}...")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Error fetching data: {response.text}")
        
    return response.json()

def run_monte_carlo_simulations(games_data):
    slate_projections = []
    
    for game in games_data:
        home_team = game['home_team']
        away_team = game['away_team']
        commence_time = game['commence_time']
        
        # Extract Pinnacle Spread (Defaulting if not found)
        pinnacle_spread = "N/A"
        try:
            for bookmaker in game.get('bookmakers', []):
                if bookmaker['key'] == 'pinnacle':
                    for market in bookmaker['markets']:
                        if market['key'] == 'spreads':
                            # Get the away team's spread as the benchmark
                            for outcome in market['outcomes']:
                                if outcome['name'] == away_team:
                                    pinnacle_spread = f"{outcome['point']:+}"
        except (KeyError, IndexError):
            pass

        # ---------------------------------------------------------
        # INSERT MONTE CARLO POISSON/MARKOV LOGIC HERE
        # ---------------------------------------------------------
        # For the dashboard wireup, we generate simulated baseline probabilities. 
        # You will replace these variables with the outputs from your actual arrays.
        
        simulated_home_prob = round(np.random.uniform(40.0, 65.0), 1)
        simulated_away_prob = round(100.0 - simulated_home_prob, 1)
        simulated_edge_pts = round(np.random.uniform(-3.5, 3.5), 1)
        simulated_ev = round(np.random.uniform(-5.0, 12.0), 1)

        slate_projections.append({
            "commence_time": commence_time,
            "away_team": away_team,
            "home_team": home_team,
            "model_away_prob": simulated_away_prob,
            "model_home_prob": simulated_home_prob,
            "pinnacle_spread": pinnacle_spread,
            "edge": simulated_edge_pts,
            "ev": simulated_ev
        })
        
    return slate_projections

def save_to_dashboard(slate_projections):
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    
    file_path = 'data/upcoming_slate.json'
    with open(file_path, 'w') as f:
        json.dump(slate_projections, f, indent=4)
    print(f"Successfully saved {len(slate_projections)} games to {file_path}")

if __name__ == "__main__":
    raw_games = fetch_current_slate()
    projections = run_monte_carlo_simulations(raw_games)
    save_to_dashboard(projections)
