import csv
import sys
import os
import numpy as np
import pandas as pd
import nfl_data_py as nfl

# Ensure Python can import from the same folder regardless of working directory
sys.path.append(os.path.dirname(__file__))

from run_simulations import (
    is_regular_season,
    aggregate_team_epa_qb_isolated,
    get_scoring_environment,
    fit_points_calibration,
    weighted_current_value,
    simulate_nfl_game,
    proportional_devig,
    calculate_ev,
    american_to_decimal,
    ML_MARKET_WEIGHT,
    SPREAD_MARKET_WEIGHT,
    NUM_SIMS,
    DEFAULT_EPA_TO_POINTS_PER_PLAY,
    EPA_PRIOR_PLAYS
)

np.random.seed(42)

def backtest_safe_float(val):
    if val is None: return None
    val_str = str(val).strip().lower()
    if val_str in ['', 'n/a', 'nan', 'none']: return None
    try: return float(val_str)
    except ValueError: return None

def format_odds(odds):
    if odds is None: return "N/A"
    return f"{odds:+.0f}"

def evaluate_bet(market_type, pick, row, bet_amount=25.0):
    away_score = backtest_safe_float(row.get('Actual_Away_Score'))
    home_score = backtest_safe_float(row.get('Actual_Home_Score'))
    
    if away_score is None or home_score is None:
        return 0.0, 'push'
        
    actual_total = away_score + home_score

    if market_type == 'ML':
        odds = backtest_safe_float(row.get('Pinnacle_Away_ML') if pick == 'away' else row.get('Pinnacle_Home_ML'))
        won = (away_score > home_score) if pick == 'away' else (home_score > away_score)
        if away_score == home_score: return 0.0, 'push'
        profit = bet_amount * (american_to_decimal(odds) - 1.0) if won else -bet_amount
        return profit, ('win' if won else 'loss')

    elif market_type == 'SPREAD':
        line = backtest_safe_float(row.get('Pinnacle_Away_Spread') if pick == 'away' else row.get('Pinnacle_Home_Spread'))
        margin = (away_score - home_score) if pick == 'away' else (home_score - away_score)
        if margin + line == 0: return 0.0, 'push'
        won = (margin + line) > 0
        profit = bet_amount * (100 / 110) if won else -bet_amount  
        return profit, ('win' if won else 'loss')

def get_walk_forward_stats(prior_pbp, current_pbp):
    prior_stats, prior_league_stats, _ = aggregate_team_epa_qb_isolated(prior_pbp)
    
    if current_pbp.empty:
        current_stats, current_league_stats = {}, prior_league_stats
    else:
        current_stats, current_league_stats, _ = aggregate_team_epa_qb_isolated(current_pbp)

    blended = {}
    all_teams = sorted(set(prior_stats) | set(current_stats))

    for team in all_teams:
        prior = prior_stats.get(team, {})
        current = current_stats.get(team, {})
        current_off_plays = int(current.get("plays", 0))
        current_def_plays = int(current.get("def_plays", 0))

        blended[team] = {
            "primary_qb": current.get("primary_qb", prior.get("primary_qb", "Unknown")),
            "off_epa_per_play": weighted_current_value(current.get("off_epa_per_play"), prior.get("off_epa_per_play"), current_off_plays),
            "off_pass_epa": weighted_current_value(current.get("off_pass_epa"), prior.get("off_pass_epa"), current_off_plays),
            "off_rush_epa": weighted_current_value(current.get("off_rush_epa"), prior.get("off_rush_epa"), current_off_plays),
            "def_epa_per_play": weighted_current_value(current.get("def_epa_per_play"), prior.get("def_epa_per_play"), current_def_plays),
            "def_pass_epa": weighted_current_value(current.get("def_pass_epa"), prior.get("def_pass_epa"), current_def_plays),
            "def_rush_epa": weighted_current_value(current.get("def_rush_epa"), prior.get("def_rush_epa"), current_def_plays),
            "plays": current_off_plays or int(prior.get("plays", 0)),
            "def_plays": current_def_plays or int(prior.get("def_plays", 0)),
            "pace": float(current.get("pace", prior.get("pace", 63.0))),
        }

    prior_points, prior_games = get_scoring_environment(prior_pbp)
    current_points, current_games = get_scoring_environment(current_pbp)

    league_points = prior_points
    if current_games > 0:
        league_weight = current_games / (current_games + 32.0)
        league_points = (league_weight * current_points + (1.0 - league_weight) * prior_points)

    prior_epa = float(prior_league_stats.get("off_epa_per_play", 0.0))
    current_epa = float(current_league_stats.get("off_epa_per_play", prior_epa))
    
    league_epa = prior_epa
    if current_games > 0:
        epa_weight = current_games / (current_games + 32.0)
        league_epa = (epa_weight * current_epa + (1.0 - epa_weight) * prior_epa)

    calibration = fit_points_calibration(prior_pbp, prior_stats, prior_epa, prior_points)
    
    return blended, league_points, league_epa, {"calibration": calibration}

def get_ev_bucket(ev):
    if ev < 3.0: return "1.5% to 3.0%"
    elif ev < 5.0: return "3.0% to 5.0%"
    elif ev < 7.0: return "5.0% to 7.0%"
    else: return "7.0%+"

def run_walk_forward_backtest(flat_bet=25.0):
    ml_min, ml_max = 3.0, 10.0
    sp_min, sp_max = 1.5, 7.0

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_file = os.path.join(base_dir, 'history_2025.csv')
    
    if not os.path.exists(history_file):
        print(f"Error: {history_file} not found. Run seed_2025.py first.")
        return

    print("Loading massive Play-By-Play datasets (2024 Prior & 2025 Current)...")
    prior_raw = nfl.import_pbp_data([2024])
    prior_pbp = is_regular_season(prior_raw)
    
    current_raw = nfl.import_pbp_data([2025])
    current_pbp = is_regular_season(current_raw)

    print("\n--- INITIATING 2025 WALK-FORWARD BACKTEST (ML & SPREADS ONLY) ---")
    
    with open(history_file, mode='r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))

    games_evaluated = 0
    total_staked = 0.0
    total_profit = 0.0
    
    results_summary = {
        'ML': {'W': 0, 'L': 0, 'P': 0},
        'SPREAD': {'W': 0, 'L': 0, 'P': 0}
    }

    # Initialize Edge Buckets
    edge_buckets = {
        "1.5% to 3.0%": {'W': 0, 'L': 0, 'P': 0, 'staked': 0.0, 'profit': 0.0},
        "3.0% to 5.0%": {'W': 0, 'L': 0, 'P': 0, 'staked': 0.0, 'profit': 0.0},
        "5.0% to 7.0%": {'W': 0, 'L': 0, 'P': 0, 'staked': 0.0, 'profit': 0.0},
        "7.0%+":        {'W': 0, 'L': 0, 'P': 0, 'staked': 0.0, 'profit': 0.0}
    }

    cached_stats_by_date = {}

    for row in reader:
        date = row.get('Date')
        away_score = backtest_safe_float(row.get('Actual_Away_Score'))
        home_score = backtest_safe_float(row.get('Actual_Home_Score'))
        
        if away_score is None or home_score is None:
            continue

        if date not in cached_stats_by_date:
            print(f"-> Time-traveling to {date}: Recalculating league baselines based strictly on past plays...")
            current_pbp_filtered = current_pbp[current_pbp['game_date'] < date].copy()
            cached_stats_by_date[date] = get_walk_forward_stats(prior_pbp, current_pbp_filtered)
            
        epa_stats, league_points, league_epa, diagnostics = cached_stats_by_date[date]

        away = row['Away_Team']
        home = row['Home_Team']
        if away not in epa_stats or home not in epa_stats:
            continue

        games_evaluated += 1
        
        spread_line = backtest_safe_float(row.get('Pinnacle_Home_Spread'))
        away_sp = backtest_safe_float(row.get('Pinnacle_Away_Spread'))
        home_sp = backtest_safe_float(row.get('Pinnacle_Home_Spread'))
        away_ml = backtest_safe_float(row.get('Pinnacle_Away_ML'))
        home_ml = backtest_safe_float(row.get('Pinnacle_Home_ML'))

        sim_res = simulate_nfl_game(
            epa_stats[away], 
            epa_stats[home], 
            league_points=league_points,
            league_epa=league_epa,
            hfa_points=diagnostics["calibration"]["hfa_points"],
            epa_to_points=diagnostics["calibration"]["epa_to_points_per_play"],
            spread_line=spread_line if spread_line is not None else -3.0
        )

        # Moneyline Check
        if away_ml is not None and home_ml is not None:
            try:
                t_away, t_home = proportional_devig(away_ml, home_ml)
                b_away = ((1.0 - ML_MARKET_WEIGHT) * (sim_res["away_win_prob"] / 100.0)) + (ML_MARKET_WEIGHT * t_away)
                b_home = ((1.0 - ML_MARKET_WEIGHT) * (sim_res["home_win_prob"] / 100.0)) + (ML_MARKET_WEIGHT * t_home)
                away_ml_ev = calculate_ev(b_away * 100.0, away_ml)
                home_ml_ev = calculate_ev(b_home * 100.0, home_ml)

                if away_ml_ev and ml_min <= away_ml_ev <= ml_max:
                    p, res = evaluate_bet('ML', 'away', row, flat_bet)
                    if res != 'push':
                        bkt = get_ev_bucket(away_ml_ev)
                        edge_buckets[bkt]['W' if res == 'win' else 'L' if res == 'loss' else 'P'] += 1
                        edge_buckets[bkt]['staked'] += flat_bet
                        edge_buckets[bkt]['profit'] += p
                        
                        total_staked += flat_bet; total_profit += p; results_summary['ML'][res[0].upper()] += 1
                        print(f"[{date}] BET ML: {away} ({format_odds(away_ml)}) vs {home} | {res.upper()} | EV: {away_ml_ev:+.1f}% | Stake: ${flat_bet:.2f} | Profit: ${p:+.2f}")

                elif home_ml_ev and ml_min <= home_ml_ev <= ml_max:
                    p, res = evaluate_bet('ML', 'home', row, flat_bet)
                    if res != 'push':
                        bkt = get_ev_bucket(home_ml_ev)
                        edge_buckets[bkt]['W' if res == 'win' else 'L' if res == 'loss' else 'P'] += 1
                        edge_buckets[bkt]['staked'] += flat_bet
                        edge_buckets[bkt]['profit'] += p
                        
                        total_staked += flat_bet; total_profit += p; results_summary['ML'][res[0].upper()] += 1
                        print(f"[{date}] BET ML: {home} ({format_odds(home_ml)}) vs {away} | {res.upper()} | EV: {home_ml_ev:+.1f}% | Stake: ${flat_bet:.2f} | Profit: ${p:+.2f}")
            except: pass

        # Spread Check
        if spread_line is not None and away_sp is not None:
            try:
                t_sp_a, t_sp_h = proportional_devig(-110, -110)
                b_sp_a = ((1.0 - SPREAD_MARKET_WEIGHT) * sim_res["spread_probs"]["away"]) + (SPREAD_MARKET_WEIGHT * t_sp_a)
                b_sp_h = ((1.0 - SPREAD_MARKET_WEIGHT) * sim_res["spread_probs"]["home"]) + (SPREAD_MARKET_WEIGHT * t_sp_h)
                b_sp_push = sim_res["spread_probs"]["push"]

                away_sp_ev = calculate_ev(b_sp_a * 100.0, -110, b_sp_push * 100.0)
                home_sp_ev = calculate_ev(b_sp_h * 100.0, -110, b_sp_push * 100.0)

                if away_sp_ev and sp_min <= away_sp_ev <= sp_max:
                    p, res = evaluate_bet('SPREAD', 'away', row, flat_bet)
                    if res != 'push':
                        bkt = get_ev_bucket(away_sp_ev)
                        edge_buckets[bkt]['W' if res == 'win' else 'L' if res == 'loss' else 'P'] += 1
                        edge_buckets[bkt]['staked'] += flat_bet
                        edge_buckets[bkt]['profit'] += p
                        
                        total_staked += flat_bet; total_profit += p; results_summary['SPREAD'][res[0].upper()] += 1
                        print(f"[{date}] BET SPREAD: {away} {away_sp:+.1f} (-110) vs {home} | {res.upper()} | EV: {away_sp_ev:+.1f}% | Stake: ${flat_bet:.2f} | Profit: ${p:+.2f}")

                elif home_sp_ev and sp_min <= home_sp_ev <= sp_max:
                    p, res = evaluate_bet('SPREAD', 'home', row, flat_bet)
                    if res != 'push':
                        bkt = get_ev_bucket(home_sp_ev)
                        edge_buckets[bkt]['W' if res == 'win' else 'L' if res == 'loss' else 'P'] += 1
                        edge_buckets[bkt]['staked'] += flat_bet
                        edge_buckets[bkt]['profit'] += p
                        
                        total_staked += flat_bet; total_profit += p; results_summary['SPREAD'][res[0].upper()] += 1
                        print(f"[{date}] BET SPREAD: {home} {home_sp:+.1f} (-110) vs {away} | {res.upper()} | EV: {home_sp_ev:+.1f}% | Stake: ${flat_bet:.2f} | Profit: ${p:+.2f}")
            except: pass

    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
    total_bets = sum(results_summary['ML'].values()) + sum(results_summary['SPREAD'].values())
    
    print("\n==================================================")
    print("             EDGE BUCKET PERFORMANCE              ")
    print("==================================================")
    for bkt in ["1.5% to 3.0%", "3.0% to 5.0%", "5.0% to 7.0%", "7.0%+"]:
        s = edge_buckets[bkt]
        total_bkt_bets = s['W'] + s['L'] + s['P']
        if total_bkt_bets > 0:
            win_pct = (s['W'] / (s['W'] + s['L']) * 100) if (s['W'] + s['L']) > 0 else 0.0
            bkt_roi = (s['profit'] / s['staked'] * 100) if s['staked'] > 0 else 0.0
            print(f"[{bkt:<12}] Record: {s['W']:>2}-{s['L']:>2}-{s['P']:>2} ({win_pct:>5.1f}%) | ROI: {bkt_roi:>6.1f}% | Profit: ${s['profit']:>7.2f}")
        else:
            print(f"[{bkt:<12}] No bets placed in this range.")

    print("\n--------------------------------------------------")
    print("           FINAL 2025 SUMMARY (NO TOTALS)         ")
    print("--------------------------------------------------")
    print(f"Games Evaluated:  {games_evaluated}")
    print(f"Total Bets Placed: {total_bets}")
    print(f"Moneyline Record: {results_summary['ML']['W']}-{results_summary['ML']['L']}-{results_summary['ML']['P']}")
    print(f"Spread Record:    {results_summary['SPREAD']['W']}-{results_summary['SPREAD']['L']}-{results_summary['SPREAD']['P']}")
    print(f"Total Staked:     ${total_staked:.2f}")
    print(f"Net Profit:       ${total_profit:+.2f}")
    print(f"Strategy ROI:     {roi:.2f}%")
    print("==================================================")

if __name__ == '__main__':
    run_walk_forward_backtest()
