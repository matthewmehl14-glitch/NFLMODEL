import os
import json
import requests
import numpy as np
from datetime import datetime, timedelta, timezone

def fetch_current_slate():
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise ValueError("ODDS_API_KEY environment variable is missing.")

    # Rolling window: Now through the next 5 days (captures Thursday night through Monday night)
    now = datetime.now(timezone.utc)
    end_date = now + timedelta(days=5)

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": api_key,
        "regions": "eu",  # Pinnacle is categorized under the EU region
        "bookmakers": "pinnacle",
        "markets": "spreads,h2h",
        "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commenceTimeTo": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
    }

    print(f"Fetching games from {params['commenceTimeFrom']} to {params['commenceTimeTo']}...")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Odds API returned an error: {response.text}")
        
    return response.json()

def run_monte_carlo_simulations(games_data):
    slate_projections = []
    
    for game in games_data:
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        commence_time = game.get('commence_time')
        
        # Extract Pinnacle spreads and prices
        pinnacle_away_spread = None
        pinnacle_home_spread = None
        pinnacle_away_price = None
        pinnacle_home_price = None
        
        for bookmaker in game.get('bookmakers', []):
            if bookmaker.get('key') == 'pinnacle':
                for market in bookmaker.get('markets', []):
                    if market.get('key') == 'spreads':
                        for outcome in market.get('outcomes', []):
                            if outcome.get('name') == away_team:
                                pinnacle_away_spread = outcome.get('point')
                                pinnacle_away_price = outcome.get('price')
                            elif outcome.get('name') == home_team:
                                pinnacle_home_spread = outcome.get('point')
                                pinnacle_home_price = outcome.get('price')

        # Fallback if Pinnacle lines are not posted yet
        if pinnacle_away_spread is None:
            pinnacle_away_spread = 0.0
            pinnacle_home_spread = 0.0

        # Format display strings for the market lines
        away_spread_str = f"{pinnacle_away_spread:+g}"
        home_spread_str = f"{pinnacle_home_spread:+g}"

        # -----------------------------------------------------------------
        # SIMULATION / MODEL PLACEHOLDER
        # -----------------------------------------------------------------
        # Replace these values with your true Monte Carlo distribution outputs:
        model_home_prob = round(float(np.random.uniform(42.0, 68.0)), 1)
        model_away_prob = round(100.0 - model_home_prob, 1)

        # Model projected spread difference (negative means away favored)
        # Compare model's fair line against the available Pinnacle line
        model_fair_spread_away = round(float(np.random.uniform(-7.5, 7.5)), 1)
        spread_edge = round(pinnacle_away_spread - model_fair_spread_away, 1)

        # -----------------------------------------------------------------
        # DETERMINE TARGET PLAY & EDGE
        # -----------------------------------------------------------------
        # If the model spread edge is positive, taking away points has value.
        # Otherwise, the value lies on the home side.
        if spread_edge >= 0:
            target_team = away_team
            target_side = "AWAY"
            target_line = away_spread_str
            recommended_bet = f"{away_team} {away_spread_str}"
            edge_pts = abs(spread_edge)
            calculated_ev = round(edge_pts * 2.1, 1)
        else:
            target_team = home_team
            target_side = "HOME"
            target_line = home_spread_str
            recommended_bet = f"{home_team} {home_spread_str}"
            edge_pts = abs(spread_edge)
            calculated_ev = round(edge_pts * 2.1, 1)

        slate_projections.append({
            "commence_time": commence_time,
            "away_team": away_team,
            "home_team": home_team,
            "model_away_prob": model_away_prob,
            "model_home_prob": model_home_prob,
            "pinnacle_away_line": f"{away_team} {away_spread_str}",
            "pinnacle_home_line": f"{home_team} {home_spread_str}",
            "target_team": target_team,
            "target_side": target_side,
            "target_line": target_line,
            "recommended_bet": recommended_bet,
            "edge_pts": edge_pts,
            "ev": calculated_ev
        })
        
    return slate_projections

def save_to_dashboard(slate_projections):
    os.makedirs('data', exist_ok=True)
    file_path = 'data/upcoming_slate.json'
    with open(file_path, 'w') as f:
        json.dump(slate_projections, f, indent=4)
    print(f"Successfully processed and saved {len(slate_projections)} games to {file_path}")

if __name__ == "__main__":
    raw_games = fetch_current_slate()
    projections = run_monte_carlo_simulations(raw_games)
    save_to_dashboard(projections)
