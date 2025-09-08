#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def logistic(x):
    import numpy as _np
    return 1.0 / (1.0 + _np.exp(-x))






def generate_synthetic_nba_games(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    home_elo = rng.normal(1500, 100, size = n)
    away_elo = rng.normal(1500, 100, size = n)
    elo_diff = home_elo - away_elo
    home_off_5 = rng.normal(112, 5, size = n)
    away_off_5 = rng.normal(112, 5, size = n)
    home_def_5 = rng.normal(112, 5, size = n)
    away_def_5 = rng.normal(112, 5, size = n)
    home_rest = rng.integers(0, 4, size = n)
    away_rest = rng.integers(0, 4, size = n)
    home_b2b = (home_rest == 0).astype(int)
    away_b2b = (away_rest == 0).astype(int)
    home_travel_km = rng.uniform(0, 500, size=n)
    away_travel_km = rng.uniform(200, 4000, size=n)
    inj_home = rng.integers(0, 3, size = n)
    inj_away = rng.integers(0, 3, size = n)
    spread_noise = rng.normal(0, 2.0, size = n)
    vegas_spread = (
        0.03 * elo_diff
        + 0.20 * (home_off_5 - away_off_5)
        - 0.20 * (home_def_5 - away_def_5)
        + 0.50 * (home_rest - away_rest)
        - 0.80 * (home_b2b - away_b2b)
        - 0.60 * (inj_home - inj_away)
        + spread_noise
    )

    intercept = 0.25
    latent = (
        intercept
        + 0.004 * elo_diff 
        + 0.020 * (home_off_5 - away_off_5)
        - 0.020 * (home_def_5 - away_def_5)
        + 0.18 * (home_rest - away_rest)
        - 0.30 * (inj_home - inj_away)
        + 0.00015 * (away_travel_km - home_travel_km)
        + 0.06 * vegas_spread
        + rng.normal(0, 0.75, size = n)
    )
    p_home_win = np.clip(logistic(latent), 0.05, 0.95)
    home_win = rng.binomial(1, p_home_win)
    seasons = rng.integers(2014, 2025, size = n)
    dates = pd.to_datetime(
        rng.integers(pd.Timestamp("2014-10-01").value//10**9, pd.Timestamp("2024-06-30").value//10**9, size = n),
        unit = "s"
    )
    game_id = np.arange(1, n + 1)
    df = pd.DataFrame({
        "season": seasons,
        "date": dates,
        "game_id": game_id,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
        "home_off_rating_5": home_off_5,
        "away_off_rating_5": away_off_5,
        "home_def_rating_5": home_def_5,
        "away_def_rating_5": away_def_5,
        "home_rest": home_rest,
        "away_rest": away_rest,
        "home_b2b": home_b2b,
        "away_b2b": away_b2b,
        "home_travel_km": home_travel_km,
        "away_travel_km": away_travel_km,
        "injured_starters_home": inj_home,
        "injured_starters_away": inj_away,
        "vegas_spread": vegas_spread,
        "home_win": home_win,
    })
    return df

def generate_synthetic_nfl_games(n=4000, seed=21):
    rng = np.random.default_rng(seed)
    home_elo = rng.normal(1500, 120, size = n)
    away_elo = rng.normal(1500, 120, size = n)
    elo_diff = home_elo - away_elo
    home_qb_rating = rng.normal(95, 12, size = n)
    away_qb_rating = rng.normal(95, 12, size = n)
    qb_rating_diff = home_qb_rating - away_qb_rating
    home_rush_off = rng.normal(0, 1, size=n)
    away_rush_off = rng.normal(0, 1, size=n)
    home_rush_def = rng.normal(0, 1, size=n)
    away_rush_def = rng.normal(0, 1, size=n)
    home_pass_off = rng.normal(0, 1, size=n)
    away_pass_off = rng.normal(0, 1, size=n)
    home_pass_def = rng.normal(0, 1, size=n)
    away_pass_def = rng.normal(0, 1, size=n)
    home_rest = rng.integers(3, 10, size=n)
    away_rest = rng.integers(3, 10, size=n)
    home_b2b = (home_rest <= 4).astype(int)
    away_b2b = (away_rest <= 4).astype(int)
    home_travel_km = rng.uniform(0, 3000, size = n)
    away_travel_km = rng.uniform(200, 6000, size = n)
    inj_home = rng.integers(0, 5, size=n)
    inj_away = rng.integers(0, 5, size=n)
    weather_index = rng.uniform(-1, 1, size=n)
    spread_noise = rng.normal(0, 2.5, size=n)
    vegas_spread = (
        0.035 * elo_diff
        + 0.04 * qb_rating_diff
        + 0.40 * (home_pass_off - away_pass_off - (home_pass_def - away_pass_def))
        + 0.30 * (home_rush_off - away_rush_off - (home_rush_def - away_rush_def))
        + 0.30 * (home_rest - away_rest)
        - 0.60 * (inj_home - inj_away)
        + 0.0002 * (home_travel_km - away_travel_km)
        + 0.50 * weather_index
        + spread_noise
    )
    intercept = 0.40
    latent = (
        intercept
        + 0.005 * elo_diff
        + 0.030 * qb_rating_diff
        + 0.25 * (home_pass_off - away_pass_off) - 0.25 * (home_pass_def - away_pass_def)
        + 0.20 * (home_rush_off - away_rush_off) - 0.20 * (home_rush_def - away_rush_def)
        + 0.20 * (home_rest - away_rest)
        - 0.25 * (inj_home - inj_away)
        + 0.00015 * (home_travel_km - away_travel_km)
        + 0.08 * vegas_spread
        + 0.15 * weather_index
        + rng.normal(0, 0.9, size = n)
    )
    p_home_win = np.clip(logistic(latent), 0.05, 0.95)
    home_win = rng.binomial(1, p_home_win)
    seasons = rng.integers(2014, 2025, size=n)
    dates = pd.to_datetime(
        rng.integers(pd.Timestamp("2014-09-01").value//10**9, pd.Timestamp("2024-02-15").value//10**9, size=n)
        unit = "s"
    )
    game_id = np.arange(1, n + 1)
    df = pd.DataFrame({
        "season": seasons,
        "date": dates,
        "game_id": game_id,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_diff": elo_diff,
        "home_qb_rating": home_qb_rating,
        "away_qb_rating": away_qb_rating,
        "qb_rating_diff": qb_rating_diff,
        "home_rush_off": home_rush_off,
        "away_rush_off": away_rush_off,
        "home_rush_def": home_rush_def,
        "away_rush_def": away_rush_def,
        "home_pass_off": home_pass_off,
        "away_pass_off": away_pass_off,
        "home_pass_def": away_pass_def,
        "away_pass_def": away_pass_def,
        "home_rest": home_rest,
        "away_rest": away_rest,
        "home_b2b": home_b2b,
        "away_b2b": away_b2b,
        "home_travel_km": home_travel_km,
        "away_travel_km": away_travel_km,
        "injured_starters_home": inj_home,
        "injured_starters_away": inj_away,
        "weather_index": weather_index,
        "vegas_spread": vegas_spread,
        "home_win": home_win,
    })
    return df

def main():
    parser = argparse.ArgumentParser(description = "Generate or validate multi-sport (NBA/NFL) datasets.")
    parser.add_argument("--generate", type=int, default=0, help="Rows of synthetic data to generate (0 to skip).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for synthetic generation.")
    parser.add_argument("--sport", type=str, default="nba", choices = ["nba", "nfl"], help = "Which synthetic generator to use")
    parser.add_argument("--out", type = str, default="data/simulated_games.csv", help = "Output CSV path for synthetic data.")
    parser.add_argument("--validate", type=str, default="", help="Optional path to validate columns of an existing CSV (must include home_win).")
    args = parser.parse_args()

    if args.generate > 0:
        if args.sport == "nba":
            df = generate_synthetic_nba_games(n=args.generate, seed=args.seed)
        else:
            df = generate_synthetic_nfl_games(n=args.generate, seed=args.seed)
        out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok = True)
        df.to_csv(out_path, index=False)
        print(f"Wrote synthetic {args.sport.upper()} dataset to {out_path.resolve()} with shape = {df.shape}")

    if args.validate:
        df = pd.read_csv(args.validate)
        if "home_win" not in df.columns:
            raise SystemExit("Missing required column: home_win")
        print(f"Validated columns; found {len(df.columns)} columns & {len(df):,} rows.")

if __name__ == "__main__":
    main()
