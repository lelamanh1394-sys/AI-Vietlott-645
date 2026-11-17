
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
        row[np.array(nums, dtype=int) - 1] = 1

        matrix.append(row)
        valid_draws.append(draw)

    if not matrix:
        raise ValueError("No valid Mega 6/45 draws.")

    return np.asarray(matrix), valid_draws


def check_data(draws):
    print("\n===== DATA INTEGRITY CHECK =====")

    ids = [
        str(d.get("draw_id", d.get("id", "")))
        for d in draws
    ]

    known_ids = [x for x in ids if x]

    duplicate_ids = len(known_ids) - len(set(known_ids))

    print("Draw records:", len(draws))
    print("Duplicate draw IDs:", duplicate_ids)

    dates = [
        str(d.get("date", ""))
        for d in draws
        if d.get("date")
    ]

    if len(dates) >= 2:
        print("First date:", dates[0])
        print("Last date:", dates[-1])

        if dates[0] > dates[-1]:
            print("WARNING: Dates may be newest-first.")
        else:
            print("Date order appears oldest-first.")

    if duplicate_ids:
        print("WARNING: Duplicate IDs detected.")

    if len(draws) >= 2 and known_ids:
        print("Check source ordering before interpreting backtest.")


# ================= BACKTEST =================
def run_walk_forward(matrix):
    n = len(matrix)
    test_count = min(BACKTEST_DRAWS, n - MIN_HISTORY)

    if test_count <= 0:
        raise ValueError("Not enough history.")

    start = n - test_count

    results = {name: [] for name in STRATEGIES}
    candidates = {}

    print("\n===== V11 STRICT WALK-FORWARD =====")
    print("Test draws:", test_count)
    print("Training cap:", TRAIN_MAX_DRAWS)
    print("Predictions use previous draws only.")

    for target in range(start, n):
        history = matrix[
            max(0, target - TRAIN_MAX_DRAWS):target
        ]

        actual = matrix[target]
        scores = all_scores(history)

        for name in STRATEGIES:
            ticket = choose_six(scores[name])
            indices = np.asarray(ticket, dtype=int) - 1

            hits = int(actual[indices].sum())
            results[name].append(hits)

    return results, start, test_count


# ================= PAIRED RANDOM CONTROL =================
def make_paired_random_control(test_count):
    """
    For every actual draw, simulate 5,000 random six-number tickets.

    The number of matches of a random Mega 6/45 ticket against
    any fixed six-number winning draw follows Hypergeometric(45, 6, 6).

    This is mathematically equivalent to generating random tickets
    and counting their matches, without storing the ticket numbers.
    """
    rng = np.random.default_rng(SEED)

    control = rng.hypergeometric(
        ngood=6,
        nbad=39,
        nsample=6,
        size=(RANDOM_SIMULATIONS, test_count),
    )

    return control


# ================= METRICS =================
def hit_distribution(values):
    values = np.asarray(values, dtype=int)

    return {
        k: int(np.sum(values == k))
        for k in range(7)
    }


def mean_ci(values):
    values = np.asarray(values, dtype=float)

    if len(values) < 2:
        return float(values.mean()), float(values.mean())

    mean = values.mean()
    se = values.std(ddof=1) / np.sqrt(len(values))

    return (
        float(mean - 1.96 * se),
        float(mean + 1.96 * se),
    )


def evaluate_strategy(name, hits, random_control):
    hits = np.asarray(hits, dtype=int)

    random_totals = random_control.sum(axis=1)
    strategy_total = int(hits.sum())

    # Monte Carlo upper-tail probability.
    p_value = (
        np.count_nonzero(random_totals >= strategy_total) + 1
    ) / (len(random_totals) + 1)

    # Paired comparison: compare strategy with random tickets
    # on the same target draws.
    random_mean_by_draw = random_control.mean(axis=0)

    paired_difference = hits - random_mean_by_draw

    diff_low, diff_high = mean_ci(paired_difference)

    dist = hit_distribution(hits)

    print(f"\n===== {name} =====")
    print("Total hits:", strategy_total)
    print("Average hits:", round(hits.mean(), 4))
    print("95% CI of average:", *[
        round(x, 4) for x in mean_ci(hits)
    ])

    print(
        "Paired difference vs random:",
        round(paired_difference.mean(), 4)
    )
    print(
        "95% CI paired difference:",
        round(diff_low, 4),
        round(diff_high, 4)
    )

    print("Random upper-tail p:", round(p_value, 4))

    print("\nHit distribution:")
    for k in range(7):
        print(f"{k} hits: {dist[k]}")

    print("2+ hits:", sum(dist[k] for k in range(2, 7)))
    print("3+ hits:", sum(dist[k] for k in range(3, 7)))
    print("4+ hits:", sum(dist[k] for k in range(4, 7)))
    print("5+ hits:", dist[5] + dist[6])
    print("6 hits:", dist[6])

    return {
        "total": strategy_total,
        "average": float(hits.mean()),
        "p": float(p_value),
        "paired_difference": float(paired_difference.mean()),
        "paired_ci": (diff_low, diff_high),
        "distribution": dist,
    }


# ================= RANDOM BENCHMARK =================
def random_report(control):
    flat = control.flatten()
    dist = hit_distribution(flat)

    print("\n===== PAIRED RANDOM BENCHMARK =====")
    print("Random tickets per target draw:", RANDOM_SIMULATIONS)
    print("Number of target draws:", control.shape[1])
    print("Total simulated random tickets:", len(flat))
    print("Random average hits:", round(flat.mean(), 4))
    print("Theoretical average hits: 0.8000")

    print("\nRandom hit distribution:")

    for k in range(7):
        rate = dist[k] / len(flat)
        print(
            f"{k} hits: {dist[k]} ({rate:.4%})"
        )

    print("Random 2+ rate:",
          round(np.mean(flat >= 2), 4))
    print("Random 3+ rate:",
          round(np.mean(flat >= 3), 4))


# ================= PERIOD VALIDATION =================
def period_report(results, control):
    print("\n===== THREE-PERIOD PAIRED VALIDATION =====")

    test_count = control.shape[1]

    for name, values in results.items():
        print(f"\n{name}")

        values = np.asarray(values, dtype=int)

        for i in range(3):
            start = i * test_count // 3
            end = (i + 1) * test_count // 3

            segment = values[start:end]
            random_segment = control[:, start:end]

            paired = (
                segment - random_segment.mean(axis=0)
            )

            low, high = mean_ci(paired)

            print(
                f"Period {i + 1}: "
                f"draws={len(segment)}, "
                f"model_mean={segment.mean():.4f}, "
                f"random_mean={random_segment.mean():.4f}, "
                f"paired_diff={paired.mean():.4f}, "
                f"paired_CI=[{low:.4f}, {high:.4f}], "
                f"zero={np.sum(segment == 0)}, "
                f"2+={np.sum(segment >= 2)}, "
                f"3+={np.sum(segment >= 3)}"
            )


# ================= MULTIPLE TEST CORRECTION =================
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
            "adjusted p =", round(adjusted, 4)
        )


# ================= LATEST CANDIDATES =================
def latest_candidates(matrix):
    history = matrix[-TRAIN_MAX_DRAWS:]
    scores = all_scores(history)

    print("\n===== V11 LATEST CANDIDATES =====")

    for name in STRATEGIES:
        ticket = choose_six(scores[name])

        print(
            name + ":",
            " - ".join(f"{x:02d}" for x in ticket)
        )

    candidate = choose_six(scores["V9 Ensemble"])

    print("\nV11 EXPERIMENTAL CANDIDATE:")
    print(" - ".join(f"{x:02d}" for x in candidate))

    print("Research ranking only.")


# ================= MAIN =================
def main():
    print("===== AI MEGA 6/45 V11 =====")
    print("Loading historical data...")

    matrix, draws = load_matrix()

    print("\nValid draws:", len(matrix))
    print("Latest draw:",
          draws[-1].get("draw_id", draws[-1].get("id", "unknown")))
    print("Latest date:", draws[-1].get("date", "unknown"))

    check_data(draws)

    results, start, test_count = run_walk_forward(matrix)

    control = make_paired_random_control(test_count)

    random_report(control)

    reports = {}

    print("\n===== V11 STRATEGY RESULTS =====")

    for name in STRATEGIES:
        reports[name] = evaluate_strategy(
            name,
            results[name],
            control,
        )

    period_report(results, control)
    correction_report(reports)
    latest_candidates(matrix)

    print("\n===== V11 COMPLETE =====")
    print("No strategy is proven to beat random selection.")
    print("Research only. Do not interpret rankings as better odds.")


if __name__ == "__main__":
    main()
