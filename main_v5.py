
import json
import requests
import numpy as np

from collections import Counter
from statistics import mean, stdev
from math import sqrt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ==========================================
# AI VIETLOTT MEGA 6/45 - VERSION 5
# Strict walk-forward + Monte Carlo
# Research only. No guarantee of winning.
# ==========================================

URL = "https://raw.githubusercontent.com/vietvudanh/vietlott-data/main/data/power645.jsonl"

BACKTEST_DRAWS = 300
TRAIN_MAX_DRAWS = 500
MIN_HISTORY = 200
REFIT_INTERVAL = 10
RANDOM_SIMULATIONS = 5000
SEED = 6452026

WINDOWS = [10, 30, 50, 100, 200]

MODEL_NAMES = [
    "STATISTICS",
    "LOGISTIC REGRESSION",
    "RANDOM FOREST",
    "V5 ENSEMBLE"
]


# ==========================================
# 1. DOWNLOAD AND VALIDATE DATA
# ==========================================

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
            date = str(item["date"])

            if len(numbers) != 6:
                continue

            if len(set(numbers)) != 6:
                continue

            if any(n < 1 or n > 45 for n in numbers):
                continue

            if draw_id in seen:
                continue

            seen.add(draw_id)

            draws.append({
                "id": draw_id,
                "date": date,
                "result": sorted(numbers)
            })

        except (ValueError, KeyError, TypeError):
            continue

    draws.sort(
        key=lambda x: (x["date"], int(x["id"]))
    )

    if len(draws) < MIN_HISTORY + BACKTEST_DRAWS:
        raise ValueError("Not enough historical data")

    print("Valid draws:", len(draws))
    print("Latest draw:", draws[-1]["id"])
    print("Latest date:", draws[-1]["date"])

    return draws


# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================

def extract_features(history, number):

    length = len(history)

    positions = [
        i for i, draw in enumerate(history)
        if number in draw["result"]
    ]

    features = []

    # Frequency windows
    for window in WINDOWS:

        sample = history[-window:]

        rate = sum(
            number in d["result"]
            for d in sample
        ) / max(len(sample), 1)

        features.append(rate)

    # Long-term frequency
    features.append(
        len(positions) / max(length, 1)
    )

    # Gap
    gap = (
        length - 1 - positions[-1]
        if positions else length
    )

    features.append(min(gap, 300) / 300)

    # Interval statistics
    intervals = [
        positions[i] - positions[i - 1]
        for i in range(1, len(positions))
    ]

    features.append(
        min(mean(intervals), 100) / 100
        if intervals else 0
    )

    features.append(
        min(stdev(intervals), 100) / 100
        if len(intervals) > 1 else 0
    )

    # Recent trend
    recent = history[-30:]
    previous = history[-60:-30]

    recent_rate = sum(
        number in d["result"] for d in recent
    ) / max(len(recent), 1)

    previous_rate = sum(
        number in d["result"] for d in previous
    ) / max(len(previous), 1)

    features.append(recent_rate - previous_rate)

    # Last 10 draws
    last10 = history[-10:]

    features.append(
        sum(number in d["result"] for d in last10)
        / max(len(last10), 1)
    )

    # Exponential decay
    size = min(length, 100)

    recent100 = history[-size:]

    weights = np.exp(
        -np.arange(size, dtype=float) / 25
    )

    hits = np.array([
        number in d["result"]
        for d in recent100
    ], dtype=float)[::-1]

    features.append(
        float(np.dot(hits, weights) / weights.sum())
    )

    return features


# ==========================================
# 3. STATISTICAL MODEL
# ==========================================

def statistical_scores(history):

    weights = {
        30: 0.20,
        50: 0.15,
        100: 0.25,
        200: 0.20
    }

    scores = {}

    for number in range(1, 46):

        score = 0

        for window, weight in weights.items():

            sample = history[-window:]

            frequency = sum(
                number in d["result"]
                for d in sample
            ) / max(len(sample), 1)

            score += weight * frequency

        long_frequency = sum(
            number in d["result"]
            for d in history
        ) / max(len(history), 1)

        score += 0.20 * long_frequency

        scores[number] = score

    return scores


def rank_scores(scores):

    return sorted(
        scores,
        key=lambda n: (-scores[n], n)
    )


def percentile_scores(scores):

    ordered = rank_scores(scores)

    return {
        number: (45 - rank) / 44
        for rank, number in enumerate(ordered)
    }


# ==========================================
# 4. MACHINE LEARNING
# ==========================================

def create_models():

    logistic = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=SEED
        )
    )

    forest = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=SEED,
        n_jobs=-1
    )

    return logistic, forest


# ==========================================
# 5. BUILD TRAINING DATA
# ==========================================

def build_training_data(draws, start, end, cache):

    X = []
    y = []

    for t in range(start, end):

        if t not in cache:

            past_only = draws[:t]

            cache[t] = [
                extract_features(past_only, n)
                for n in range(1, 46)
            ]

        actual = set(draws[t]["result"])

        for index, number in enumerate(range(1, 46)):

            X.append(cache[t][index])

            y.append(
                1 if number in actual else 0
            )

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int)
    )


# ==========================================
# 6. PREDICTION
# ==========================================

def predict_all(history, logistic, forest):

    features = np.asarray([
        extract_features(history, n)
        for n in range(1, 46)
    ], dtype=float)

    lp = logistic.predict_proba(features)
    fp = forest.predict_proba(features)

    logistic_index = list(
        logistic.classes_
    ).index(1)

    forest_index = list(
        forest.classes_
    ).index(1)

    logistic_scores = {
        n: float(lp[n - 1][logistic_index])
        for n in range(1, 46)
    }

    forest_scores = {
        n: float(fp[n - 1][forest_index])
        for n in range(1, 46)
    }

    stats = statistical_scores(history)

    sr = percentile_scores(stats)
    lr = percentile_scores(logistic_scores)
    fr = percentile_scores(forest_scores)

    ensemble = {
        n: (
            0.3333 * sr[n]
            + 0.3333 * lr[n]
            + 0.3334 * fr[n]
        )
        for n in range(1, 46)
    }

    return {
        "STATISTICS": set(rank_scores(stats)[:6]),

        "LOGISTIC REGRESSION":
            set(rank_scores(logistic_scores)[:6]),

        "RANDOM FOREST":
            set(rank_scores(forest_scores)[:6]),

        "V5 ENSEMBLE":
            set(rank_scores(ensemble)[:6])
    }


# ==========================================
# 7. CONFIDENCE INTERVAL
# ==========================================

def confidence_interval(values):

    avg = mean(values)

    if len(values) < 2:
        return avg, avg

    margin = (
        1.96 * stdev(values)
        / sqrt(len(values))
    )

    return avg - margin, avg + margin


# ==========================================
# 8. MONTE CARLO RANDOM BASELINE
# ==========================================

def monte_carlo_random(actual_sets):

    rng = np.random.default_rng(SEED + 505)

    totals = np.zeros(
        RANDOM_SIMULATIONS,
        dtype=np.int32
    )

    population = np.arange(1, 46)

    for actual in actual_sets:

        mask = np.zeros(46, dtype=np.uint8)

        mask[list(actual)] = 1

        # Each row is an independent random ticket.
        # Sorting random scores selects 6 unique numbers.
        random_scores = rng.random(
            (RANDOM_SIMULATIONS, 45)
        )

        tickets = (
            np.argpartition(
                random_scores,
                6,
                axis=1
            )[:, :6] + 1
        )

        totals += np.asarray(mask[tickets].sum(axis=1), dtype=np.int32)

    return totals


# ==========================================
# 9. STABILITY TEST
# ==========================================

def stability_test(results):

    print(
        "\n===== V5 PERIOD STABILITY ====="
    )

    for name, hits in results.items():

        print("\n", name)

        for block in range(3):

            segment = hits[
                block * 100:
                (block + 1) * 100
            ]

            print(
                "Period", block + 1,
                "| Average:", round(mean(segment), 4),
                "| Zero hits:", segment.count(0),
                "| 2+ hits:", sum(x >= 2 for x in segment)
            )


# ==========================================
# 10. STRICT WALK-FORWARD BACKTEST
# ==========================================

def backtest(draws):

    test_start = len(draws) - BACKTEST_DRAWS

    results = {
        name: [] for name in MODEL_NAMES
    }

    actual_sets = []
    cache = {}
    models = None

    print(
        "\n===== V5 STRICT WALK-FORWARD BACKTEST ====="
    )

    print("Test draws:", BACKTEST_DRAWS)
    print("Training history limit:", TRAIN_MAX_DRAWS)
    print("Refit interval:", REFIT_INTERVAL)
    print("Random simulations:", RANDOM_SIMULATIONS)

    print(
        "Leakage rule: features for draw t use draws before t only"
    )

    for i in range(test_start, len(draws)):

        history = draws[:i]

        actual = set(draws[i]["result"])

        actual_sets.append(actual)

        if (
            models is None
            or (i - test_start) % REFIT_INTERVAL == 0
        ):

            train_start = max(
                MIN_HISTORY,
                i - TRAIN_MAX_DRAWS
            )

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
                "Refit at draw:",
                draws[i]["id"],
                "| Samples:",
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

    # Random benchmark
    random_totals = monte_carlo_random(actual_sets)

    random_means = (
        random_totals / BACKTEST_DRAWS
    )

    random_average = float(
        np.mean(random_means)
    )

    random_range = np.percentile(
        random_means,
        [2.5, 97.5]
    )

    print("\n===== V5 RESULTS =====")

    print("Theoretical random expectation: 0.8000")

    print(
        "Simulated random mean:",
        round(random_average, 4)
    )

    print(
        "Random strategy 95% range:",
        round(float(random_range[0]), 4),
        round(float(random_range[1]), 4)
    )

    for name, hits in results.items():

        total = sum(hits)

        avg = mean(hits)

        low, high = confidence_interval(hits)

        differences = [
            x - 0.8 for x in hits
        ]

        dlow, dhigh = confidence_interval(
            differences
        )

        # Monte Carlo upper-tail probability
        p_value = (
            1 + int(np.sum(random_totals >= total))
        ) / (RANDOM_SIMULATIONS + 1)

        adjusted_p = min(
            1.0,
            p_value * len(MODEL_NAMES)
        )

        print("\n", name)

        print("Total hits:", total)

        print("Average hits:", round(avg, 4))

        print(
            "95% CI:",
            round(low, 4),
            round(high, 4)
        )

        print(
            "Difference vs random:",
            round(avg - 0.8, 4)
        )

        print(
            "Difference CI:",
            round(dlow, 4),
            round(dhigh, 4)
        )

        print(
            "Random upper-tail p:",
            round(p_value, 4)
        )

        print(
            "Adjusted p:",
            round(adjusted_p, 4)
        )

        for k in range(7):

            print(
                f"Hits {k}: {hits.count(k)}"
            )

    stability_test(results)

    return cache


# ==========================================
# 11. NEXT DRAW PREDICTION
# ==========================================

def final_prediction(draws, cache):

    start = max(
        MIN_HISTORY,
        len(draws) - TRAIN_MAX_DRAWS
    )

    X, y = build_training_data(
        draws,
        start,
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

    print("\n===== V5 NEXT-DRAW CANDIDATES =====")

    for name, numbers in predictions.items():

        print(
            name,
            ":",
            " - ".join(
                f"{n:02d}"
                for n in sorted(numbers)
            )
        )

    votes = Counter()

    for model in [
        "STATISTICS",
        "LOGISTIC REGRESSION",
        "RANDOM FOREST"
    ]:

        for number in predictions[model]:

            votes[number] += 1

    print("\n===== MODEL CONSENSUS =====")

    for number, count in sorted(
        votes.items(),
        key=lambda item: (-item[1], item[0])
    ):

        print(
            f"{number:02d}: {count}/3 models"
        )


# ==========================================
# 12. MAIN
# ==========================================

def main():

    draws = download_data()

    cache = backtest(draws)

    final_prediction(draws, cache)

    print(
        "\n===== AI MEGA 6/45 V5 COMPLETE ====="
    )

    print(
        "Research only. No guarantee of winning."
    )


if __name__ == "__main__":
    main()
