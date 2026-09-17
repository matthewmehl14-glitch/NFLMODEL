import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# Pull the secret key GitHub securely provides
api_key = os.environ["ODDS_API_KEY"]

# Set the window to only grab the next 6 days of upcoming games
now = datetime.utcnow()
end_date = now + timedelta(days=6)

url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
params = {
    "apiKey": api_key,
    "regions": "us,eu",
    "bookmakers": "pinnacle",
    "markets": "h2h,spreads,totals",
    "commenceTimeFrom": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "commenceTimeTo": end_date.strftime("%Y-%m-%dT%H:%M:%SZ")
}

response = requests.get(url, params=params)
upcoming_games = response.json()

# ... Your Monte Carlo simulation arrays go here ...

# Example saving logic (this updates the dashboard data)
# df.to_json('data/upcoming_slate.json', orient='records')
# df.to_csv('data/historical_log.csv', mode='a', header=False)
