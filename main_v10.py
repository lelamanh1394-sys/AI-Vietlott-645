
import numpy as np

from main_v5 import download_data
from main_v9 import all_scores, choose_six

# ================= CONFIG =================
BACKTEST_DRAWS = 300
TRAIN_MAX_DRAWS = 500
MIN_HISTORY = 200
RANDOM_SIMULATIONS = 5000
SEED = 6452026

STRATEGIES = [
    "Multi-window",
    "EWMA",
    "Trend",
    "V9 Ensemble",
]


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
        raise ValueError("No valid Mega 6/45 draws.")

    return np.asarray(matrix), valid_draws


# ================= BACKTEST =================
def run_walk_forward(matrix):
    n = len(matrix)
    test_count = min(BACKTEST_DRAWS, n - MIN_HISTORY)

    if test_count <= 0:
        raise ValueError("Not enough historical data.")

    start = n - test_count

    results = {name: [] for name in STRATEGIES}
    predictions = {name: [] for name in STRATEGIES}

    print("\n===== V10 STRICT WALK-FORWARD =====")
    print("Test draws:", test_count)
    print("Training cap:", TRAIN_MAX_DRAWS)
    print("Only past data used for predictions.")

    for target in range(start, n):
        history = matrix[
            max(0, target - TRAIN_MAX_DRAWS):target
        ]

        actual = matrix[target]
        scores = all_scores(history)

        for name in STRATEGIES:
            ticket = choose_six(scores[name])
            indices = np.array(ticket, dtype=int) - 1

            hits = int(actual[indices].sum())

            results[name].append(hits)
            predictions[name].append(ticket)

    return results, predictions, start


# ================= RANDOM CONTROL =================
def make_random_control(test_count):
    rng = np.random.default_rng(SEED)

    # Every random ticket contains exactly 6 distinct numbers.
    # Number of hits against a fair 6/45 draw follows Hypergeometric(45, 6, 6).
    random_hits = rng.hypergeometric(
        ngood=6,
        nbad=39,
        nsample=6,
        size=(RANDOM_SIMULATIONS, test_count),
    )

    return random_hits


# ================= METRICS =================
def distribution(hits):
    hits = np.asarray(hits, dtype=int)

    return {
        k: int(np.count_nonzero(hits == k))
        for k in range(7)
    }


def confidence_interval(hits):
    hits = np.asarray(hits, dtype=float)

    mean = hits.mean()

    if len(hits) < 2:
        return mean, mean

    se = hits.std(ddof=1) / np.sqrt(len(hits))

    return mean - 1.96 * se, mean + 1.96 * se


def evaluate(name, hits, random_control):
    hits = np.asarray(hits, dtype=int)

    total = int(hits.sum())
    average = float(hits.mean())

    low, high = confidence_interval(hits)

    random_totals = random_control.sum(axis=1)

    p_value = (
        np.count_nonzero(random_totals >= total) + 1
    ) / (len(random_totals) + 1)

    actual_dist = distribution(hits)

    print(f"\n===== {name} =====")
    print("Total hits:", total)
    print("Average hits:", round(average, 4))
    print("95% CI:", round(low, 4), round(high, 4))
    print("Difference vs theoretical 0.8000:",
          round(average - 0.8, 4))
    print("Random upper-tail p:", round(p_value, 4))

    print("\nHit distribution:")
    for k in range(7):
        print(f"{k} hits: {actual_dist[k]}")

    print("2+ hits:", sum(actual_dist[k] for k in range(2, 7)))
    print("3+ hits:", sum(actual_dist[k] for k in range(3, 7)))
    print("4+ hits:", sum(actual_dist[k] for k in range(4, 7)))
    print("5+ hits:", actual_dist[5] + actual_dist[6])
    print("6 hits:", actual_dist[6])

    return {
        "total": total,
        "average": average,
        "p": p_value,
        "distribution": actual_dist,
    }


# ================= RANDOM BENCHMARK =================
def print_random_benchmark(random_control):
    all_hits = random_control.flatten()

    print("\n===== RANDOM TICKET BENCHMARK =====")
    print("Simulations:", RANDOM_SIMULATIONS)
    print("Simulated average hits:",
          round(all_hits.mean(), 4))
    print("Theoretical average hits: 0.8000")

    random_dist = distribution(all_hits)

    print("\nRandom hit distribution:")
    for k in range(7):
        rate = random_dist[k] / len(all_hits)
        print(
            f"{k} hits: {random_dist[k]} "
            f"({rate:.4%})"
        )

    print("Random 2+ rate:",
          round(np.mean(all_hits >= 2), 4))
    print("Random 3+ rate:",
          round(np.mean(all_hits >= 3), 4))


# ================= PERIOD ANALYSIS =================
def period_analysis(results):
    print("\n===== THREE-PERIOD VALIDATION =====")

    for name, values in results.items():
        print(f"\n{name}")

        size = len(values)

        for i in range(3):
            start = i * size // 3
            end = (i + 1) * size // 3

            segment = np.asarray(values[start:end])

            print(
                f"Period {i + 1}: "
                f"draws={len(segment)}, "
                f"mean={segment.mean():.4f}, "
                f"zero={np.sum(segment == 0)}, "
                f"2+={np.sum(segment >= 2)}, "
                f"3+={np.sum(segment >= 3)}"
            )


# ================= MULTIPLE TESTING =================
def correction_report(reports):
    print("\n===== MULTIPLE TEST CORRECTION =====")

    number_of_tests = len(reports)

    for name, report in reports.items():
        adjusted = min(
            1.0,
            report["p"] * number_of_tests
        )

        print(
            name,
            "raw p =", round(report["p"], 4),
            "Bonferroni adjusted p =", round(adjusted, 4)
        )


# ================= LATEST CANDIDATES =================
def latest_candidates(matrix):
    history = matrix[-TRAIN_MAX_DRAWS:]
    scores = all_scores(history)

    print("\n===== V10 LATEST CANDIDATES =====")

    for name in STRATEGIES:
        ticket = choose_six(scores[name])

        print(
            name + ":",
            " - ".join(f"{x:02d}" for x in ticket)
        )

    ensemble = choose_six(scores["V9 Ensemble"])

    print("\nV10 EXPERIMENTAL CANDIDATE:")
    print(" - ".join(f"{x:02d}" for x in ensemble))

    print("Research ranking only. No proven predictive advantage.")


# ================= MAIN =================
def main():
    print("===== AI MEGA 6/45 V10 =====")
    print("Loading historical data...")

    matrix, draws = load_matrix()

    print("\nValid draws:", len(matrix))
    print("Latest draw:",
          draws[-1].get("draw_id", draws[-1].get("id", "unknown")))
    print("Latest date:", draws[-1].get("date", "unknown"))

    results, predictions, start = run_walk_forward(matrix)

    test_count = len(next(iter(results.values())))
    random_control = make_random_control(test_count)

    print_random_benchmark(random_control)

    reports = {}

    print("\n===== V10 STRATEGY COMPARISON =====")

    for name in STRATEGIES:
        reports[name] = evaluate(
            name,
            results[name],
            random_control,
        )

    period_analysis(results)
    correction_report(reports)
    latest_candidates(matrix)

    print("\n===== V10 COMPLETE =====")
    print("Historical patterns do not establish predictive advantage.")
    print("No strategy is proven to beat random selection.")


if __name__ == "__main__":
    main()
