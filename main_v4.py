import json
import os
import random
from collections import Counter
from statistics import mean, stdev
from math import sqrt

import requests
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# AI VIETLOTT MEGA 6/45 - V4
# Strict walk-forward evaluation + simulated random benchmark
# Research only: lottery draws are independent and unpredictable.
# =====================================================

URL = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl"
BACKTEST_DRAWS = 300
TRAIN_MAX_DRAWS = 500
MIN_HISTORY = 200
REFIT_INTERVAL = 10
RANDOM_SIMULATIONS = 1000
RNG_SEED = 6452026
WINDOWS = [10, 30, 50, 100, 200]


def download_data():
    print("DOWNLOADING MEGA 6/45 DATA")
    response = requests.get(URL, timeout=60)
    response.raise_for_status()
    draws, seen = [], set()
    for line in response.text.splitlines():
        try:
            item = json.loads(line)
            draw_id = str(item["id"])
            numbers = [int(n) for n in item["result"]]
            date = str(item["date"])
            if len(numbers) != 6 or len(set(numbers)) != 6:
                continue
            if any(n < 1 or n > 45 for n in numbers) or draw_id in seen:
                continue
            seen.add(draw_id)
            draws.append({"id": draw_id, "date": date, "result": sorted(numbers)})
        except (ValueError, KeyError, TypeError):
            continue
    draws.sort(key=lambda x: (x["date"], int(x["id"])))
    if len(draws) < MIN_HISTORY + BACKTEST_DRAWS:
        raise ValueError("Not enough valid historical draws for V4 backtest")
    os.makedirs("data", exist_ok=True)
    with open("data/mega645.jsonl", "w", encoding="utf-8") as f:
        for draw in draws:
            f.write(json.dumps(draw, ensure_ascii=False) + "\n")
    print("Valid draws:", len(draws))
    print("Latest draw:", draws[-1]["id"])
    print("Latest date:", draws[-1]["date"])
    return draws


def extract_features(history, number):
    length = len(history)
    positions = [i for i, draw in enumerate(history) if number in draw["result"]]
    features = []
    for window in WINDOWS:
        sample = history[-window:]
        features.append(sum(number in d["result"] for d in sample) / max(len(sample), 1))
    features.append(len(positions) / max(length, 1))
    gap = length - 1 - positions[-1] if positions else length
    features.append(min(gap, 300) / 300.0)
    intervals = [positions[j] - positions[j - 1] for j in range(1, len(positions))]
    features.append(min(mean(intervals), 100) / 100.0 if intervals else 0.0)
    features.append(min(stdev(intervals), 100) / 100.0 if len(intervals) > 1 else 0.0)
    recent, previous = history[-30:], history[-60:-30]
    recent_rate = sum(number in d["result"] for d in recent) / max(len(recent), 1)
    previous_rate = sum(number in d["result"] for d in previous) / max(len(previous), 1)
    features.append(recent_rate - previous_rate)
    last10 = history[-10:]
    features.append(sum(number in d["result"] for d in last10) / max(len(last10), 1))
    # Exponentially decayed frequency; only past draws are included.
    decay_weights = np.exp(-np.arange(min(length, 100), dtype=float) / 25.0)
    recent100 = history[-len(decay_weights):]
    hits = np.array([number in d["result"] for d in recent100], dtype=float)[::-1]
    features.append(float(np.dot(hits, decay_weights) / max(decay_weights.sum(), 1e-12)))
    return features


def statistical_scores(history):
    weights = {30: 0.20, 50: 0.15, 100: 0.25, 200: 0.20}
    scores = {}
    for number in range(1, 46):
        score = 0.0
        for window, weight in weights.items():
            sample = history[-window:]
            score += weight * sum(number in d["result"] for d in sample) / max(len(sample), 1)
        score += 0.20 * sum(number in d["result"] for d in history) / max(len(history), 1)
        scores[number] = score
    return scores


def rank_scores(scores):
    return sorted(scores, key=lambda n: (-scores[n], n))


def percentile_scores(scores):
    ordered = rank_scores(scores)
    return {number: (45 - rank) / 44.0 for rank, number in enumerate(ordered)}


def create_models(seed_offset=0):
    seed = RNG_SEED + seed_offset
    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed)
    )
    forest = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1
    )
    return logistic, forest


def build_training_data(draws, start, end, cache):
    # Leakage guard: each feature row at t sees draws[:t] only;
    # its label is draw[t], and end is exclusive.
    X, y = [], []
    for t in range(start, end):
        if t not in cache:
            past_only = draws[:t]
            cache[t] = [extract_features(past_only, n) for n in range(1, 46)]
        actual = set(draws[t]["result"])
        for idx, number in enumerate(range(1, 46)):
            X.append(cache[t][idx])
            y.append(1 if number in actual else 0)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def predict_all(history, logistic, forest):
    features = np.asarray([extract_features(history, n) for n in range(1, 46)], dtype=float)
    lp = logistic.predict_proba(features)[:, list(logistic.classes_).index(1)]
    fp = forest.predict_proba(features)[:, list(forest.classes_).index(1)]
    logistic_scores = {n: float(lp[n - 1]) for n in range(1, 46)}
    forest_scores = {n: float(fp[n - 1]) for n in range(1, 46)}
    stats = statistical_scores(history)
    sr, lr, fr = percentile_scores(stats), percentile_scores(logistic_scores), percentile_scores(forest_scores)
    ensemble = {n: 0.40 * sr[n] + 0.30 * lr[n] + 0.30 * fr[n] for n in range(1, 46)}
    return {
        "STATISTICS": set(rank_scores(stats)[:6]),
        "LOGISTIC REGRESSION": set(rank_scores(logistic_scores)[:6]),
        "RANDOM FOREST": set(rank_scores(forest_scores)[:6]),
        "V4 ENSEMBLE": set(rank_scores(ensemble)[:6]),
    }


def ci95(values):
    avg = mean(values)
    if len(values) < 2:
        return avg, avg
    margin = 1.96 * stdev(values) / sqrt(len(values))
    return avg - margin, avg + margin


def random_strategy_benchmark(actual_sets, simulations=RANDOM_SIMULATIONS):
    rng = random.Random(RNG_SEED + 999)
    means = []
    for _ in range(simulations):
        total = 0
        for actual in actual_sets:
            ticket = set(rng.sample(range(1, 46), 6))
            total += len(ticket & actual)
        means.append(total / len(actual_sets))
    return means


def backtest(draws):
    test_start = len(draws) - BACKTEST_DRAWS
    results = {name: [] for name in ["STATISTICS", "LOGISTIC REGRESSION", "RANDOM FOREST", "V4 ENSEMBLE"]}
    actual_sets = []
    cache, models = {}, None
    print("\n===== V4 STRICT WALK-FORWARD BACKTEST =====")
    print("Test draws:", len(draws) - test_start)
    print("Training history limit:", TRAIN_MAX_DRAWS)
    print("Refit interval:", REFIT_INTERVAL)
    print("Leakage rule: features at draw t use draws before t only")

    for i in range(test_start, len(draws)):
        history = draws[:i]
        actual = set(draws[i]["result"])
        actual_sets.append(actual)
        if models is None or (i - test_start) % REFIT_INTERVAL == 0:
            train_start = max(MIN_HISTORY, i - TRAIN_MAX_DRAWS)
            X, y = build_training_data(draws, train_start, i, cache)
            logistic, forest = create_models()
            logistic.fit(X, y)
            forest.fit(X, y)
            models = (logistic, forest)
            print("Refit at draw:", draws[i]["id"], "| samples:", len(y))
        predictions = predict_all(history, models[0], models[1])
        for name, predicted in predictions.items():
            results[name].append(len(predicted & actual))

    random_means = random_strategy_benchmark(actual_sets)
    random_avg = mean(random_means)
    random_low, random_high = ci95(random_means)
    print("\n===== V4 RESULTS =====")
    print("Theoretical random expectation:", "0.8000")
    print("Simulated random strategy mean:", round(random_avg, 4))
    print("Random strategy 95% range of means:", round(float(np.percentile(random_means, 2.5)), 4), round(float(np.percentile(random_means, 97.5)), 4))

    for name, hits in results.items():
        avg = mean(hits)
        low, high = ci95(hits)
        diff = [x - 0.8 for x in hits]
        dlow, dhigh = ci95(diff)
        p_value = (1 + sum(x >= avg for x in random_means)) / (len(random_means) + 1)
        print("\n", name)
        print("Average hits:", round(avg, 4))
        print("95% CI:", round(low, 4), round(high, 4))
        print("Difference vs theoretical random:", round(avg - 0.8, 4))
        print("Difference 95% CI:", round(dlow, 4), round(dhigh, 4))
        print("Random strategy tail probability:", round(p_value, 4))
        for k in range(7):
            print(f"Hits {k}: {hits.count(k)}")
    return cache


def final_prediction(draws, cache):
    start = max(MIN_HISTORY, len(draws) - TRAIN_MAX_DRAWS)
    X, y = build_training_data(draws, start, len(draws), cache)
    logistic, forest = create_models()
    logistic.fit(X, y)
    forest.fit(X, y)
    predictions = predict_all(draws, logistic, forest)
    print("\n===== V4 NEXT-DRAW CANDIDATES =====")
    for name, numbers in predictions.items():
        print(name, ":", " - ".join(f"{n:02d}" for n in sorted(numbers)))


def main():
    draws = download_data()
    cache = backtest(draws)
    final_prediction(draws, cache)
    print("\n===== AI MEGA 6/45 V4 COMPLETE =====")
    print("Research only. No guarantee of winning.")


if __name__ == "__main__":
    main()
