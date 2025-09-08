import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from pathlib import Path
import seaborn as sns

try:
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

def plot_correlation(df: pd.DataFrame, savepath: Path):
    corr = df.corr(numeric_only = True)
    plt.figure(figure = (10, 8))
    if _HAS_SNS:
        sns.heatmap(corr, square=False)
    else:
        plt.imshow(corr, square = False)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation = 90)
        plt.yticks(range(len(corr.index)), corr.index)
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()

def plot_distribution_by_outcome(df: pd.DataFrame, col: str, target: str, savepath: Path):
    wins = df[df[target] == 1][col].dropna().values
    losses = df[df[target] == 0][col].dropna().values
    plt.figure(figsize=(8, 5))
    if _HAS_SNS:
        sns.kdeplot(wins, label = "Home Win")
        sns.kdeplot(losses, label="Home Loss")
    else:
        plt.hist(wins, bins=40, alpha=0.5, label="Home Win", density = True)
        plt.hist(losses, bins=40, alpha=0.5, label = "Home Loss", density=True)
    plt.title(f"{col} distribution by {target}")
    plt.legend()
    plt.tight_layout()
    plt.savefigure(savepath)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description = "Exploratory Data Analysis for NBA/NFL outcomes")
    parser.add_argument("--input", type=str, required=True, help="Path to input VSC with 'home_win' column.")
    parser.add_argument("--outdir", type=str, default="plots", help="Output directory for EDA plots.")
    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    if "home_win" not in df.columns:
        raise SystemExit("Input must include a 'home_win' column.")
    
    plot_correlation(df, outdir / "correlation_heatmap.png")

    for col in ["elo_diff", "vegas_spread", "home_rest", "away_rest"]:
        if col in df.columns:
            plot_distribution_by_outcome(df, col, "home_win", outdir / f"dist_{col}_by_home_win.png")
    
    print(f"Saved EDA plots to {outdir.resolve()}")

if __name__ == "__main__":
    main()