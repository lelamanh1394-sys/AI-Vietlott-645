
import numpy as np
from main_v5 import download_data

# ================= CONFIG =================
BACKTEST_DRAWS = 300
TRAIN_MAX_DRAWS = 500
MIN_HISTORY = 200
RANDOM_SIMULATIONS = 5000
SEED = 6452026

WINDOWS = [30, 100, 300, 500]
P = 6 / 45


# ================= DATA =================
def load_matrix():
    draws = download_data()

    matrix = []
    valid_draws = []

    for draw in draws:
        numbers = draw.get("result", [])

        try:
            nums = sorted(set(int(x) for x in numbers))
        except (TypeError, ValueError):
            continue

        if len(nums) != 6 or any(x < 1 or x > 45 for x in nums):
            continue

        row = np.zeros(45, dtype=np.int8)
        row[np.array(nums) - 1] = 1

        matrix.append(row)
        valid_draws.append(draw)

    if not matrix:
        raise ValueError("No valid Mega 6/45 draws found.")

    return np.asarray(matrix), valid_draws


# ================= SCORING =================
def zscore(values):
    values = np.asarray(values, dtype=float)
    sd = values.std()

    if sd < 1e-12:
        return np.zeros_like(values)

    return (values - values.mean()) / sd


def multi_window_score(history):
    scores = np.zeros(45)

    weights = {
        30: 0.10,
        100: 0.20,
        300: 0.30,
        500: 0.40,
    }

    for window, weight in weights.items():
        sample = history[-min(window, len(history)):]
        count = sample.sum(axis=0)
        n = len(sample)

        expected = n * P
        variance = n * P * (1 - P)

        z = (count - expected) / np.sqrt(max(variance, 1e-9))
        scores += weight * z

    return scores


def ewma_score(history, half_life=50):
    n = len(history)

    ages = np.arange(n - 1, -1, -1)
    weights = np.exp(-np.log(2) * ages / half_life)

    weighted_count = (history * weights[:, None]).sum(axis=0)
    total_weight = weights.sum()

    weighted_rate = weighted_count / total_weight
    effective_n = total_weight ** 2 / np.sum(weights ** 2)

    variance = P * (1 - P) / max(effective_n, 1)

    return (weighted_rate - P) / np.sqrt(max(variance, 1e-9))


def trend_score(history):
    if len(history) < 60:
        return np.zeros(45)

    recent = history[-30:].mean(axis=0)
    previous = history[-60:-30].mean(axis=0)

    difference = recent - previous

    # Shrink noisy 30-draw differences toward zero.
    shrinkage = 30 / (30 + 45)

    return zscore(difference * shrinkage)


def rank_percentile(scores):
    order = np.argsort(np.argsort(scores))
    return order / 44.0


def all_scores(history):
    history = history[-TRAIN_MAX_DRAWS:]

    frequency = multi_window_score(history)
    recency = ewma_score(history)
    trend = trend_score(history)

    ensemble = (
        0.40 * rank_percentile(frequency)
        + 0.35 * rank_percentile(recency)
        + 0.25 * rank_percentile(trend)
    )

    return {
        "Multi-window": frequency,
        "EWMA": recency,
        "Trend": trend,
        "V9 Ensemble": ensemble,
    }


def choose_six(scores):
    indices = np.argsort(scores)[-6:]
    return sorted((indices + 1).tolist())


# ================= STATISTICS =================
def confidence_interval(hits):
    hits = np.asarray(hits, dtype=float)

    mean = hits.mean()
    if len(hits) < 2:
        return mean, mean

    se = hits.std(ddof=1) / np.sqrt(len(hits))
    return mean - 1.96 * se, mean + 1.96 * se


def evaluate_strategy(hits, random_totals):
    hits = np.asarray(hits, dtype=int)

    total = int(hits.sum())
    average = float(hits.mean())
    low, high = confidence_interval(hits)

    p_value = (
        np.count_nonzero(random_totals >= total) + 1
    ) / (len(random_totals) + 1)

    return {
        "total": total,
        "average": average,
        "low": low,
        "high": high,
        "difference": average - 0.8,
        "p": p_value,
    }


# ================= WALK FORWARD =================
def walk_forward(matrix):
    n = len(matrix)
    test_count = min(BACKTEST_DRAWS, n - MIN_HISTORY)

    if test_count <= 0:
        raise ValueError("Not enough history for backtest.")

    start = n - test_count

    names = [
        "Multi-window",
        "EWMA",
        "Trend",
        "V9 Ensemble",
    ]

    results = {name: [] for name in names}

    print("\n===== V9 STRICT WALK-FORWARD =====")
    print("Test draws:", test_count)
    print("Training cap:", TRAIN_MAX_DRAWS)
    print("Only previous draws are used.")

    for target in range(start, n):
        history = matrix[max(0, target - TRAIN_MAX_DRAWS):target]
        actual = matrix[target]

        scores = all_scores(history)

        for name in names:
            ticket = choose_six(scores[name])
            hits = int(actual[ticket - np.ones(6, dtype=int)].sum())
            results[name].append(hits)

    rng = np.random.default_rng(SEED)

    random_totals = rng.hypergeometric(
        ngood=6,
        nbad=39,
        nsample=6,
        size=(RANDOM_SIMULATIONS, test_count),
    ).sum(axis=1)

    print("\nTheoretical random expectation: 0.8000")
    print("Monte Carlo random mean:",
          round(random_totals.mean() / test_count, 4))

    reports = {}

    for name in names:
        report = evaluate_strategy(results[name], random_totals)
        reports[name] = report

        print(f"\n===== {name} =====")
        print("Total hits:", report["total"])
        print("Average hits:", round(report["average"], 4))
        print("95% CI:",
              round(report["low"], 4),
              round(report["high"], 4))
        print("Difference vs random:",
              round(report["difference"], 4))
        print("Random upper-tail p:",
              round(report["p"], 4))

        for block in range(3):
            segment = results[name][
                block * test_count // 3:
                (block + 1) * test_count // 3
            ]
            if segment:
                print(
                    f"Period {block + 1}: "
                    f"mean={np.mean(segment):.4f}, "
                    f"zero={sum(x == 0 for x in segment)}, "
                    f"2+={sum(x >= 2 for x in segment)}"
                )

    # Bonferroni correction for four tested strategies.
    print("\n===== MULTIPLE TEST CORRECTION =====")
    for name, report in reports.items():
        adjusted = min(1.0, report["p"] * len(names))
        print(name, "adjusted p =", round(adjusted, 4))

    return reports


# ================= STABILITY =================
def stability_report(matrix):
    print("\n===== V9 RANK STABILITY =====")

    block_size = len(matrix) // 4
    top_sets = []

    for i in range(4):
        start = i * block_size
        end = (i + 1) * block_size if i < 3 else len(matrix)

        block = matrix[start:end]
        scores = multi_window_score(block[-min(500, len(block)):])
        top = set((np.argsort(scores)[-10:] + 1).tolist())
        top_sets.append(top)

        print(f"Block {i + 1} top 10:",
              " ".join(f"{x:02d}" for x in sorted(top)))

    print("\nTop-10 overlap:")
    for i in range(4):
        for j in range(i + 1, 4):
            print(
                f"Block {i + 1} vs {j + 1}:",
                f"{len(top_sets[i] & top_sets[j])}/10"
            )


# ================= LATEST CANDIDATE =================
def latest_candidate(matrix):
    scores = all_scores(matrix[-TRAIN_MAX_DRAWS:])

    print("\n===== V9 LATEST CANDIDATES =====")

    for name, values in scores.items():
        ticket = choose_six(values)
        print(name + ":", " - ".join(f"{x:02d}" for x in ticket))

    ensemble_ticket = choose_six(scores["V9 Ensemble"])

    print("\nV9 EXPERIMENTAL CANDIDATE:")
    print(" - ".join(f"{x:02d}" for x in ensemble_ticket))

    print("\nRanking only. No proven predictive advantage.")


# ================= MAIN =================
def main():
    print("DOWNLOADING MEGA 6/45 DATA")

    matrix, draws = load_matrix()

    print("Valid draws:", len(matrix))

    latest = draws[-1]
    print("Latest draw ID:",
          latest.get("draw_id", latest.get("id", "unknown")))
    print("Latest date:", latest.get("date", "unknown"))

    print("\n===== V9 RESEARCH =====")
    print("Numbers analyzed: 45")
    print("Theoretical appearance rate:", round(P, 4))

    stability_report(matrix)
    walk_forward(matrix)
    latest_candidate(matrix)

    print("\n===== AI MEGA 6/45 V9 COMPLETE =====")
    print("Research only.")
    print("Historical patterns do not establish predictive advantage.")


if __name__ == "__main__":
    main()
