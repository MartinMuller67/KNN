"""
Does the KNN parameter k overfit the same way trading strategy parameters do?

Follow-up to the moving-average crossover overfitting project. Same
walk-forward protocol, applied this time to a K-Nearest Neighbours
regression strategy instead of a moving-average crossover.

Question: naive k selection (grid search on in-sample data) is known to
overfit for MA crossover parameters. Does the same happen for KNN's k?
And does distance-weighted KNN (DWKNN), which is supposed to be more
robust to the choice of k, actually generalise better in a walk-forward
setting?

Inspired by the DWKNN literature applied to exchange rate forecasting
in emerging markets (weighting neighbours by inverse distance to reduce
sensitivity to k).

Run on your own machine (needs internet for yfinance):
    pip install -r requirements.txt
    python knn_overfitting.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

USE_REAL_DATA = True
TICKER = "AAXJ"

np.random.seed(1)


def generate_price_series(n_days=2600, s0=100):
    n_regimes = 5
    regime_len = n_days // n_regimes
    all_returns = []
    regime_params = [
        (0.0003, 0.014, 0.15),
        (0.0005, 0.010, 0.0),
        (-0.0002, 0.018, 0.10),
        (0.0004, 0.011, 0.0),
        (0.0002, 0.015, 0.12),
    ]
    for mu, sigma, autocorr_strength in regime_params:
        r = np.random.normal(mu, sigma, regime_len)
        if autocorr_strength > 0:
            noise = np.random.normal(0, sigma * 0.5, regime_len)
            r += autocorr_strength * np.roll(noise, 1)
        all_returns.append(r)
    returns = np.concatenate(all_returns)
    prices = s0 * np.exp(np.cumsum(returns))
    dates = pd.date_range("2016-01-01", periods=len(prices), freq="B")
    return pd.Series(prices, index=dates)


if USE_REAL_DATA:
    import yfinance as yf
    data = yf.download(TICKER, start="2016-01-01", end="2025-01-01", progress=False)
    prices = data["Close"].dropna()
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0]
    prices.index = pd.to_datetime(prices.index)
else:
    prices = generate_price_series()

print(f"Total price history: {len(prices)} trading days "
      f"({prices.index[0].date()} -> {prices.index[-1].date()})")


# --- feature engineering ---

def build_features(price_series):
    df = pd.DataFrame({"price": price_series})
    df["ret_1"] = df["price"].pct_change()
    df["ret_5"] = df["price"].pct_change(5)
    df["ret_20"] = df["price"].pct_change(20)
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["target"] = df["ret_1"].shift(-1)  # next day's return
    df = df.dropna()
    return df

features_df = build_features(prices)
FEATURE_COLS = ["ret_5", "ret_20", "vol_20"]


# --- strategy backtest for a given k and weighting scheme ---

TRANSACTION_COST = 0.0010

def backtest_knn(train_df, test_df, k, weights, cost=0.0):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS])
    y_train = train_df["target"].values

    model = KNeighborsRegressor(n_neighbors=k, weights=weights)
    model.fit(X_train, y_train)

    X_test = scaler.transform(test_df[FEATURE_COLS])
    predictions = model.predict(X_test)

    signal = (predictions > 0).astype(int)
    actual_returns = test_df["target"].values

    strategy_returns = signal * actual_returns

    trades = np.abs(np.diff(np.concatenate([[0], signal])))
    strategy_returns_net = strategy_returns - trades * cost

    cumulative_return = float(np.prod(1 + strategy_returns_net) - 1)
    return cumulative_return, strategy_returns_net


def sharpe_ratio(returns, periods_per_year=252, risk_free=0.025):
    if len(returns) == 0 or np.std(returns) == 0:
        return np.nan
    mean_annual = np.mean(returns) * periods_per_year
    vol_annual = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    return (mean_annual - risk_free) / vol_annual


# --- walk-forward ---

TRAIN_LEN = 500
TEST_LEN = 250
STEP = TEST_LEN
K_GRID = [3, 5, 7, 10, 15, 20, 30, 40, 50]

windows = []
start = 0
while start + TRAIN_LEN + TEST_LEN <= len(features_df):
    train = features_df.iloc[start: start + TRAIN_LEN]
    test = features_df.iloc[start + TRAIN_LEN: start + TRAIN_LEN + TEST_LEN]
    windows.append((train, test))
    start += STEP

print(f"Number of walk-forward windows: {len(windows)}")


def run_walk_forward(windows, weights, cost):
    records = []
    for w_idx, (train, test) in enumerate(windows):
        # naive grid search: split the training window itself into a
        # sub-train / sub-validation split to pick k "naively" (a
        # realistic mistake: no proper cross-validation, just one split)
        split = int(len(train) * 0.8)
        sub_train, sub_val = train.iloc[:split], train.iloc[split:]

        best_k, best_val_ret, best_val_series = None, -np.inf, None
        for k in K_GRID:
            ret, series = backtest_knn(sub_train, sub_val, k, weights, cost=cost)
            if ret > best_val_ret:
                best_val_ret, best_k, best_val_series = ret, k, series

        # refit on the FULL training window with the selected k, then
        # test on the genuinely unseen out-of-sample window
        test_ret, test_series = backtest_knn(train, test, best_k, weights, cost=cost)

        # FIX 1 (look-ahead in the "train" Sharpe): the previous version
        # refit on the full training window and then predicted on
        # train.iloc[split:], i.e. on points the model had already seen.
        # Each point was its own nearest neighbour at distance 0, so with
        # weights="distance" the prediction was the true next-day return
        # (perfect foresight). The in-sample Sharpe is now computed on the
        # validation series that was actually used to select k.
        #
        # FIX 2 (horizon mismatch in the generalisation ratio): the
        # previous ratio divided a 250-day cumulative test return by a
        # 100-day cumulative validation return, so identical daily
        # performance gave a ratio of ~2.5, not 1. The ratio now compares
        # mean daily net returns, so 1.0 really means "same performance".
        val_mean = np.mean(best_val_series)
        test_mean = np.mean(test_series)
        gen_ratio = test_mean / val_mean if val_mean != 0 else np.nan

        records.append({
            "window": w_idx + 1,
            "best_k": best_k,
            "val_return": best_val_ret,          # cumulative, 100 days
            "test_return": test_ret,             # cumulative, 250 days
            "val_mean_daily": val_mean,
            "test_mean_daily": test_mean,
            "gen_ratio": gen_ratio,
            "both_negative": (val_mean < 0) and (test_mean < 0),
            "train_sharpe": sharpe_ratio(best_val_series),
            "test_sharpe": sharpe_ratio(test_series),
        })
    return pd.DataFrame(records)


print("\nVanilla KNN (uniform weights):")
results_uniform = run_walk_forward(windows, weights="uniform", cost=TRANSACTION_COST)
print(results_uniform.to_string(index=False))

print("\nDistance-weighted KNN (DWKNN):")
results_distance = run_walk_forward(windows, weights="distance", cost=TRANSACTION_COST)
print(results_distance.to_string(index=False))

print("\n--- summary ---")
print(f"Vanilla KNN  - mean gen ratio: {results_uniform['gen_ratio'].mean():.3f}, "
      f"median: {results_uniform['gen_ratio'].median():.3f}, "
      f"std: {results_uniform['gen_ratio'].std():.3f}")
print(f"DWKNN        - mean gen ratio: {results_distance['gen_ratio'].mean():.3f}, "
      f"median: {results_distance['gen_ratio'].median():.3f}, "
      f"std: {results_distance['gen_ratio'].std():.3f}")

print("\n--- Sharpe ratios (annualised, per-window average) ---")
print(f"Vanilla KNN  - validation Sharpe: {results_uniform['train_sharpe'].mean():.2f}, "
      f"test Sharpe: {results_uniform['test_sharpe'].mean():.2f}")
print(f"DWKNN        - validation Sharpe: {results_distance['train_sharpe'].mean():.2f}, "
      f"test Sharpe: {results_distance['test_sharpe'].mean():.2f}")
print("\nPer-window Sharpe detail:")
print("Vanilla:")
print(results_uniform[["window", "train_sharpe", "test_sharpe"]].to_string(index=False))
print("DWKNN:")
print(results_distance[["window", "train_sharpe", "test_sharpe"]].to_string(index=False))

print("\nPer-window daily mean returns (validation vs test):")
for name, res in [("Vanilla", results_uniform), ("DWKNN", results_distance)]:
    print(name)
    print(res[["window", "val_mean_daily", "test_mean_daily", "gen_ratio", "both_negative"]].to_string(index=False))

for name, res in [("Vanilla KNN", results_uniform), ("DWKNN", results_distance)]:
    n_pos_sharpe = (res["test_sharpe"] > 0).sum()
    n_misleading = res["both_negative"].sum()
    print(f"{name}: positive test-Sharpe windows {n_pos_sharpe}/{len(res)}, "
          f"positive ratio but loss-making (both negative) {n_misleading}/{len(res)}")

n_neg_uniform = (results_uniform["test_return"] < 0).sum()
n_neg_distance = (results_distance["test_return"] < 0).sum()
print(f"\nNegative out-of-sample windows: vanilla {n_neg_uniform}/{len(results_uniform)}, "
      f"DWKNN {n_neg_distance}/{len(results_distance)}")

print(f"\nSelected k per window (vanilla): {results_uniform['best_k'].tolist()}")
print(f"Selected k per window (DWKNN):   {results_distance['best_k'].tolist()}")


# --- plots ---

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

x = results_uniform["window"]
width = 0.35
bars_u = axes[0].bar(x - width/2, results_uniform["gen_ratio"], width, label="Vanilla KNN (uniform)", color="#E74C3C")
bars_d = axes[0].bar(x + width/2, results_distance["gen_ratio"], width, label="DWKNN (distance-weighted)", color="#27AE60")
for bars, res in [(bars_u, results_uniform), (bars_d, results_distance)]:
    for bar, flag in zip(bars, res["both_negative"]):
        if flag:
            bar.set_hatch("///")
            bar.set_edgecolor("black")
from matplotlib.patches import Patch
handles, labels = axes[0].get_legend_handles_labels()
handles.append(Patch(facecolor="white", edgecolor="black", hatch="///"))
labels.append("Positive ratio, but loss-making\n(validation & test returns negative)")
axes[0].axhline(1.0, color="black", linestyle="--", linewidth=1, label="Perfect generalisation")
axes[0].axhline(0.0, color="grey", linewidth=0.8)
axes[0].set_xlabel("Walk-forward window")
axes[0].set_ylabel("Generalisation ratio (mean daily return, test / validation)")
axes[0].set_title("KNN Generalisation Ratio per Window")
axes[0].legend(handles, labels)
axes[0].set_xticks(x)

methods = ["Vanilla KNN\nmedian", "DWKNN\nmedian"]
values = [results_uniform["gen_ratio"].median(), results_distance["gen_ratio"].median()]
axes[1].bar(methods, values, color=["#E74C3C", "#27AE60"])
axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
axes[1].axhline(0.0, color="grey", linewidth=0.8)
axes[1].set_ylabel("Median generalisation ratio")
axes[1].set_title("Does Distance Weighting Help k Generalise Better?")
for i, v in enumerate(values):
    axes[1].text(i, v + 0.02*np.sign(v if v != 0 else 1), f"{v:.2f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("knn_overfitting.png", dpi=130)
plt.close()
print("\nSaved knn_overfitting.png")
