# Court & Gridiron Outcome Lab — Project Writeup

## The question

Can a small set of well-chosen game-context features — team strength, recent form,
rest, travel, injuries, and the market line — predict the **home team's chance of
winning** an NBA or NFL game well enough to beat a coin flip by a meaningful margin?

I built this lab to find out, and to practice the full ML loop end-to-end:
data → EDA → preprocessing → tuned models → evaluation → interactive demo.

## Why synthetic data

Real play-by-play datasets are messy and expensive to clean before you can model
anything. I wanted to control the **signal-to-noise ratio** so I could focus on the
modeling pipeline itself: feature scaling, leakage-safe cross-validation,
hyperparameter search, and probability-calibrated reporting.

The data generator in [`src/data.py`](src/data.py) builds each game from a
parameterized latent score:

```
latent = α + β₁·elo_diff + β₂·form_diff + β₃·rest_diff − β₄·injury_diff + ...
p_home_win = sigmoid(latent)
home_win    ~ Bernoulli(p_home_win)
```

The coefficients are deliberately modest, and Gaussian noise is added before the
sigmoid, so a perfect classifier is impossible — the Bayes-optimal accuracy lives
somewhere in the 70s. That's the right regime: hard enough to make modeling
choices matter, easy enough that good choices show up in the metrics.

The NFL generator adds QB rating, pass/rush offense and defense z-scores, a weather
index, and longer rest windows. The NBA generator emphasizes Elo, last-5 offensive
and defensive ratings, back-to-backs, and travel.

## EDA: what the features actually tell you

Two patterns dominated the EDA plots in [`plots/`](plots/):

- **Vegas spread is the single most informative feature.** Its distribution shifts
  cleanly between home wins and losses in both sports — the market already prices
  in most of the signal the other features carry.
- **Elo difference is a strong but noisier second.** Useful, but visibly overlapped
  between win/loss distributions, which is why no single feature gets you above the
  mid-70s.

Rest days mattered more in the NBA than in the NFL (B2Bs are a real penalty;
Sunday-vs-Sunday rest is mostly equal), and injuries had a larger marginal effect
in NFL than NBA (NFL rosters are bigger but losing 2+ starters hurts more).

## Modeling choices

Everything runs through a `Pipeline(preprocess → classifier)` so the scaler and
encoder fit **inside each CV fold**. This is the single biggest source of subtle
leakage in beginner ML projects, and it matters most when you tune hyperparameters
on the same data you scale on.

| Model | Search | Why I picked the grid I did |
|-------|--------|-----------------------------|
| Logistic Regression | GridSearchCV over `penalty`, `C`, `class_weight` | Small grid, fast, gives a calibrated baseline and tells you how much the trees actually buy you. |
| Random Forest | RandomizedSearchCV over depth, splits, features, weights | Captures non-linear interactions like "rested AND healthy AND home." |
| XGBoost | RandomizedSearchCV over depth, lr, subsample, reg_lambda | Usually wins on tabular; `reg_lambda` controls overfit on the noisy latent signal. |

Scoring metric: **F1** during search (balances precision/recall on the slightly
home-favored class), with **average precision** and **ROC AUC** reported on the
held-out 20% test split for a more complete picture.

## Results

See the README for the full results table and embedded plots. The short version:

- All three models converge to a similar **F1 in the mid-0.7s** on both sports —
  roughly where the Bayes-optimal classifier should sit given the noise in the
  data generator.
- On **NBA**, the three models are within a percentage point of each other on F1;
  Random Forest edges ahead on AP/AUC, which is what you'd expect when there are
  modest interactions (rest × b2b, injuries × form) to capture.
- On **NFL**, **Logistic Regression actually wins** on all three metrics
  (F1 0.757, AP 0.834, AUC 0.814). That was the most interesting result of the
  project. After staring at the data generator, it makes sense: the NFL latent
  score is built from a large number of additive z-scored terms (pass off/def,
  rush off/def, QB rating, weather, spread, …). Linear models *love* that shape.
  XGBoost has more capacity than it needs and slightly overfits the noise.

The lesson I'd carry into a real-data version: **always train a regularized linear
baseline.** When it wins, it tells you something true about the data — that your
features are doing most of the work and the model is mostly along for the ride.

## What I'd do differently with real data

1. **Time-aware CV.** Random k-fold leaks the future into the past — for sports
   I'd switch to a forward-chaining `TimeSeriesSplit` by season.
2. **Probability calibration.** I'd add a `CalibratedClassifierCV` wrap or
   isotonic regression so the predicted 0.62 actually means a 62% win rate.
3. **Per-team adjustments.** Elo updates online and is generated independently
   here — in production it'd be a recursive feature.
4. **Market-aware loss.** If the goal is betting rather than prediction, the
   objective should be **profit vs the closing line**, not F1.

## What I learned

- Pipeline-internal preprocessing matters more than picking the "best" model.
- Vegas is hard to beat because it's already integrating most of your features.
- Plot the per-feature distributions before fitting anything — you'll know within
  five minutes whether your data has signal.
- A small, hand-curated set of context features beats a kitchen-sink one-hot encode
  every time.
