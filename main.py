
import json
import os
import random
import requests
from collections import Counter
from statistics import mean, stdev

URL = (
    "https://raw.githubusercontent.com/"
    "vietvudanh/vietlott-data/main/data/power645.jsonl"
)

START_TEST = 200
WINDOWS = [30, 50, 100, 200]
RANDOM_TRIALS = 20
RNG_SEED = 6452026


def download_data():
    print("DOWNLOADING MEGA 6/45 DATA")

    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    draws = []
    seen = set()

    for line in response.text.splitlines():
        try:
            item = json.loads(line)
            numbers = [int(n) for n in item["result"]]

            if (
                len(numbers) != 6
                or len(set(numbers)) != 6
                or any(n < 1 or n > 45 for n in numbers)
            ):
                continue

            draw_id = str(item["id"])

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

    draws.sort(key=lambda x: (x["date"], x["id"]))

    if len(draws) <= START_TEST:
        raise ValueError("Insufficient historical data")

    os.makedirs("data", exist_ok=True)

    with open("data/mega645.jsonl", "w") as f:
        for draw in draws:
            f.write(json.dumps(draw) + "\n")

    return draws


def frequency(history):
    counts = Counter()

    for draw in history:
        counts.update(draw["result"])

    return counts


def rank_scores(scores):
    return sorted(
        range(1, 46),
        key=lambda n: (-scores[n], n)
    )


def predict_window(history, window):
    sample = history[-window:]
    counts = frequency(sample)

    scores = {
        n: counts[n] / len(sample)
        for n in range(1, 46)
    }

    return set(rank_scores(scores)[:6])


def predict_ensemble(history):
    weights = {
        "long": 0.25,
        30: 0.15,
        50: 0.15,
        100: 0.25,
        200: 0.20
    }

    scores = {n: 0.0 for n in range(1, 46)}

    long_counts = frequency(history)

    for n in range(1, 46):
        scores[n] += (
            weights["long"]
            * long_counts[n] / len(history)
        )

    for window in WINDOWS:
        sample = history[-window:]
        counts = frequency(sample)

        for n in range(1, 46):
            scores[n] += (
                weights[window]
                * counts[n] / len(sample)
            )

    return set(rank_scores(scores)[:6])


def backtest(draws):
    methods = {
        "30 ky": lambda h: predict_window(h, 30),
        "50 ky": lambda h: predict_window(h, 50),
        "100 ky": lambda h: predict_window(h, 100),
        "200 ky": lambda h: predict_window(h, 200),
        "ENSEMBLE V2": predict_ensemble
    }

    results = {name: [] for name in methods}
    random_results = []

    rng = random.Random(RNG_SEED)

    for i in range(START_TEST, len(draws)):
        history = draws[:i]
        actual = set(draws[i]["result"])

        for name, predictor in methods.items():
            predicted = predictor(history)
            results[name].append(
                len(predicted & actual)
            )

        for _ in range(RANDOM_TRIALS):
            ticket = set(rng.sample(range(1, 46), 6))
            random_results.append(
                len(ticket & actual)
            )

    print("\n===== V2 BACKTEST =====")
    print("Test draws:", len(results["ENSEMBLE V2"]))

    baseline = mean(random_results)

    print(f"Random baseline: {baseline:.4f}")
    print("Theoretical baseline: 0.8000")

    for name, hits in results.items():
        avg = mean(hits)
        sd = stdev(hits) if len(hits) > 1 else 0

        print("\n", name)
        print(f"Average hits: {avg:.4f}")
        print(f"Standard deviation: {sd:.4f}")
        print(f"Difference vs random: {avg-baseline:+.4f}")

        for k in range(7):
            print(f"Hits {k}: {hits.count(k)}")

    print("\n===== CURRENT PREDICTIONS =====")

    for name, predictor in methods.items():
        numbers = sorted(predictor(draws))
        print(name, ":", " - ".join(
            f"{n:02d}" for n in numbers
        ))

    print("\n===== V2 COMPLETE =====")


def main():
    draws = download_data()

    print("\n===== DATA =====")
    print("Total draws:", len(draws))
    print("Latest draw:", draws[-1]["id"])
    print("Latest date:", draws[-1]["date"])

    backtest(draws)

    print("\nResearch only. No guarantee of winning.")


if __name__ == "__main__":
    main()
