
import json
import os
import random
import requests
import numpy as np

from collections import Counter
from statistics import mean, stdev
from math import sqrt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# =====================================================
# AI VIETLOTT MEGA 6/45 - V3.0 FAST BUILD
# Statistical + Machine Learning + Ensemble
# Strict walk-forward backtest
# =====================================================

URL = (
    "https://raw.githubusercontent.com/"
    "vietvudanh/vietlott-data/main/data/power645.jsonl"
)

START_TEST = 400
BACKTEST_DRAWS = 120
TRAIN_MAX_DRAWS = 500
REFIT_INTERVAL = 10

RANDOM_TRIALS = 200
RNG_SEED = 6452026

WINDOWS = [10, 30, 50, 100, 200]


# ---------------- DATA ENGINE ----------------

def download_data():
    print("DOWNLOADING MEGA 6/45 DATA")

    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    draws = []
    seen = set()

    for line in response.text.splitlines():
        try:
            item = json.loads(line)

            draw_id = str(item["id"])
            numbers = [int(n) for n in item["result"]]

            if (
                len(numbers) != 6
                or len(set(numbers)) != 6
                or any(n < 1 or n > 45 for n in numbers)
            ):
                continue

            if draw_id in seen:
                continue

            seen.add(draw_id)

            draws.append({
                "id": draw_id,
                "date": item["date"],
                "result": sorted(numbers)
            })

        except (ValueError, KeyError, TypeError):
            continue

    draws.sort(key=lambda x: (x["date"], int(x["id"])))

    if len(draws) < START_TEST + 50:
        raise ValueError("Insufficient historical data")

    os.makedirs("data", exist_ok=True)

    with open("data/mega645.jsonl", "w") as f:
        for draw in draws:
            f.write(json.dumps(draw) + "\n")

    print("Valid draws:", len(draws))
    print("Latest draw:", draws[-1]["id"])
    print("Latest date:", draws[-1]["date"])

    return draws


# ---------------- FEATURE ENGINE ----------------

def extract_features(history, number):

    length = len(history)
    counts = Counter()

    positions = []

    for i, draw in enumerate(history):
        if number in draw["result"]:
            counts.update([number])
            positions.append(i)

    features = []

    # Frequency across different windows
    for window in WINDOWS:

        sample = history[-window:]

        count = sum(
            number in draw["result"]
            for draw in sample
        )

        features.append(count / min(window, length))

    # Long-term frequency
    features.append(counts[number] / length)

    # Gap since last appearance
    gap = (
        length - 1 - positions[-1]
        if positions else length
    )

    features.append(min(gap, 300) / 300)

    # Average and variation of historical intervals
    intervals = [
        positions[i] - positions[i - 1]
        for i in range(1, len(positions))
    ]

    if intervals:
        features.append(min(mean(intervals), 100) / 100)

        features.append(
            min(stdev(intervals), 100) / 100
            if len(intervals) > 1 else 0
        )
    else:
        features.extend([0, 0])

    # Recent trend versus previous period
    recent = history[-30:]
    previous = history[-60:-30]

    recent_rate = sum(
        number in draw["result"] for draw in recent
    ) / max(len(recent), 1)

    previous_rate = sum(
        number in draw["result"] for draw in previous
    ) / max(len(previous), 1)

    features.append(recent_rate - previous_rate)

    # Frequency in the last 10 draws
    last10 = history[-10:]

    features.append(
        sum(number in draw["result"] for draw in last10)
        / max(len(last10), 1)
    )

    return features


# ---------------- STATISTICAL ENGINE ----------------

def statistical_scores(history):

    scores = {}

    for number in range(1, 46):

        score = 0

        weights = {
            30: 0.20,
            50: 0.15,
            100: 0.25,
            200: 0.20
        }

        for window, weight in weights.items():

            sample = history[-window:]

            count = sum(
                number in draw["result"]
                for draw in sample
            )

            score += weight * count / len(sample)

        long_count = sum(
            number in draw["result"]
            for draw in history
        )

        score += 0.20 * long_count / len(history)

        scores[number] = score

    return scores


def rank_scores(scores):

    return sorted(
        scores,
        key=lambda n: (-scores[n], n)
    )


def predict_v2(history):

    scores = statistical_scores(history)

    return set(rank_scores(scores)[:6])


# ---------------- MACHINE LEARNING ENGINE ----------------

def create_models():

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=300,
            class_weight="balanced",
            random_state=RNG_SEED
        )
    )

    forest = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RNG_SEED,
        n_jobs=-1
    )

    return logistic, forest


def probability_scores(model, features):

    probabilities = model.predict_proba(features)

    class_index = list(model.classes_).index(1)

    return probabilities[:, class_index]


def build_training_data(draws, start, end, cache):

    X = []
    y = []

    for t in range(start, end):

        if t not in cache:

            history = draws[:t]

            cache[t] = [
                extract_features(history, number)
                for number in range(1, 46)
            ]

        actual = set(draws[t]["result"])

        for index, number in enumerate(range(1, 46)):

            X.append(cache[t][index])
            y.append(1 if number in actual else 0)

    return np.asarray(X), np.asarray(y)


# ---------------- ENSEMBLE ENGINE ----------------

def percentile_scores(scores):

    ordered = sorted(
        scores,
        key=lambda n: (-scores[n], n)
    )

    return {
        number: (45 - rank) / 44
        for rank, number in enumerate(ordered)
    }


def predict_all(history, logistic, forest):

    features = np.asarray([
        extract_features(history, number)
        for number in range(1, 46)
    ])

    logistic_probs = probability_scores(
        logistic, features
    )

    forest_probs = probability_scores(
        forest, features
    )

    logistic_scores = {
        n: float(logistic_probs[n - 1])
        for n in range(1, 46)
    }

    forest_scores = {
        n: float(forest_probs[n - 1])
        for n in range(1, 46)
    }

    stats = statistical_scores(history)

    stats_rank = percentile_scores(stats)
    logistic_rank = percentile_scores(logistic_scores)
    forest_rank = percentile_scores(forest_scores)

    # Equal rank-weight ensemble.
    ensemble = {
        n: (
            0.40 * stats_rank[n]
            + 0.30 * logistic_rank[n]
            + 0.30 * forest_rank[n]
        )
        for n in range(1, 46)
    }

    return {
        "V2 STATISTICS": set(rank_scores(stats)[:6]),
        "LOGISTIC REGRESSION": set(rank_scores(logistic_scores)[:6]),
        "RANDOM FOREST": set(rank_scores(forest_scores)[:6]),
        "V3 ENSEMBLE": set(rank_scores(ensemble)[:6])
    }


# ---------------- BACKTEST ENGINE ----------------

def confidence_interval(values):

    avg = mean(values)

    if len(values) < 2:
        return avg, avg

    margin = 1.96 * stdev(values) / sqrt(len(values))

    return avg - margin, avg + margin


def backtest(draws):

    test_start = max(
        START_TEST,
        len(draws) - BACKTEST_DRAWS
    )

    results = {
        "V2 STATISTICS": [],
        "LOGISTIC REGRESSION": [],
        "RANDOM FOREST": [],
        "V3 ENSEMBLE": []
    }

    random_results = []

    cache = {}
    models = None
    rng = random.Random(RNG_SEED)

    print("\n===== V3 WALK-FORWARD BACKTEST =====")
    print("Test period:", len(draws) - test_start)
    print("Training history limit:", TRAIN_MAX_DRAWS)
    print("Model refit interval:", REFIT_INTERVAL)

    for i in range(test_start, len(draws)):

        history = draws[:i]
        actual = set(draws[i]["result"])

        # Retrain only at fixed intervals.
        if models is None or (i - test_start) % REFIT_INTERVAL == 0:

            train_start = max(200, i - TRAIN_MAX_DRAWS)

            X, y = build_training_data(
                draws,
                train_start,
                i,
                cache
            )

            logistic, forest = create_models()

            logistic.fit(X, y)
            forest.fit(X, y)

            models = (logistic, forest)

            print(
                "Model trained at draw:",
                draws[i]["id"],
                "| Training samples:",
                len(y)
            )

        predictions = predict_all(
            history,
            models[0],
            models[1]
        )

        for name, predicted in predictions.items():

            results[name].append(
                len(predicted & actual)
            )

        # Random baseline
        random_hits = []

        for _ in range(RANDOM_TRIALS):

            ticket = set(
                rng.sample(range(1, 46), 6)
            )

            random_hits.append(
                len(ticket & actual)
            )

        random_results.append(mean(random_hits))

    baseline = mean(random_results)

    print("\n===== V3 RESULTS =====")
    print("Random baseline:", round(baseline, 4))
    print("Theoretical baseline: 0.8000")

    for name, hits in results.items():

        avg = mean(hits)

        low, high = confidence_interval(hits)

        differences = [
            a - b
            for a, b in zip(hits, random_results)
        ]

        diff_low, diff_high = confidence_interval(
            differences
        )

        print("\n", name)
        print("Average hits:", round(avg, 4))
        print(
            "95% CI:",
            round(low, 4),
            round(high, 4)
        )

        print(
            "Difference vs random:",
            round(avg - baseline, 4)
        )

        print(
            "Paired difference CI:",
            round(diff_low, 4),
            round(diff_high, 4)
        )

        for k in range(7):
            print(f"Hits {k}: {hits.count(k)}")

    return cache


# ---------------- FINAL PREDICTION ----------------

def final_prediction(draws, cache):

    X, y = build_training_data(
        draws,
        max(200, len(draws) - TRAIN_MAX_DRAWS),
        len(draws),
        cache
    )

    logistic, forest = create_models()

    logistic.fit(X, y)
    forest.fit(X, y)

    predictions = predict_all(
        draws,
        logistic,
        forest
    )

    print("\n===== V3 CURRENT PREDICTIONS =====")

    for name, numbers in predictions.items():

        print(
            name,
            ":",
            " - ".join(
                f"{n:02d}" for n in sorted(numbers)
            )
        )


# ---------------- MAIN ----------------

def main():

    draws = download_data()

    cache = backtest(draws)

    final_prediction(draws, cache)

    print("\n===== AI MEGA 6/45 V3 COMPLETE =====")
    print("Research only. No guarantee of winning.")


if __name__ == "__main__":
    main()
