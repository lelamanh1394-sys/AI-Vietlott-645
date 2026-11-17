
import math
import numpy as np

from main_v5 import download_data

# ================= CONFIG =================
BACKTEST_DRAWS = 300
TRAIN_MAX_DRAWS = 500
MIN_HISTORY = 200

RANDOM_SIMULATIONS = 5000
SEED = 6452026

WINDOWS = [30, 100, 300, 500]
WINDOW_WEIGHTS = [0.10, 0.20, 0.30, 0.40]

P = 6 / 45
NUMBER_OF_STRATEGIES = 1


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
        raise ValueError("No valid Mega 6/45 data.")

    return np.asarray(matrix), valid_draws


def integrity_check(matrix, draws):
    print("\n===== V12 DATA INTEGRITY =====")

    ids = [
        str(d.get("draw_id", d.get("id", "")))
        for d in draws
    ]

    ids = [x for x in ids if x]

    print("Valid draws:", len(matrix))
    print("Duplicate IDs:", len(ids) - len(set(ids)))

    dates = [
        str(d.get("date", ""))
        for d in draws
        if d.get("date")
    ]

    if len(dates) >= 2:
        print("First date:", dates[0])
        print("Last date:", dates[-1])
        print("Oldest-first:", dates == sorted(dates))

        if dates != sorted(dates):
            raise ValueError(
                "Date order is not oldest-first. "
                "Stop before running backtest."
            )

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate draw IDs detected.")

    if np.any(matrix.sum(axis=1) != 6):
        raise ValueError("Invalid draw row detected.")


# ================= INDIVIDUAL NUMBER ANALYSIS =================
def individual_statistics(matrix):
    n = len(matrix)
    counts = matrix.sum(axis=0)

    expected = n * P
    variance = n * P * (1 - P)
    sd = math.sqrt(variance)

    print("\n===== V12 INDIVIDUAL NUMBER STATISTICS =====")
    print("Total draws:", n)
    print("Expected appearances per number:", round(expected, 2))
    print("Theoretical appearance rate:", round(P, 6))

    records = []

    for i in range(45):
        count = int(counts[i])
        rate = count / n

        z = (count - expected) / sd

        # Approximate two-sided normal p-value.
        raw_p = math.erfc(abs(z) / math.sqrt(2))

        adjusted_p = min(1.0, raw_p * 45)

        records.append({
            "number": i + 1,
            "count": count,
            "rate": rate,
            "z": z,
            "raw_p": raw_p,
            "adjusted_p": adjusted_p,
        })

    print("\nTOP 15 HISTORICAL FREQUENCY")
    for r in sorted(records, key=lambda x: x["count"], reverse=True)[:15]:
        print(
            f'{r["number"]:02d}: '
            f'count={r["count"]}, '
            f'rate={r["rate"]:.4f}, '
            f'z={r["z"]:.3f}, '
            f'adjusted_p={r["adjusted_p"]:.4f}'
        )

    print("\n15 LOWEST HISTORICAL FREQUENCY")
    for r in sorted(records, key=lambda x: x["count"])[:15]:
        print(
            f'{r["number"]:02d}: '
            f'count={r["count"]}, '
            f'rate={r["rate"]:.4f}, '
            f'z={r["z"]:.3f}, '
            f'adjusted_p={r["adjusted_p"]:.4f}'
        )

    significant = [
        r for r in records
        if r["adjusted_p"] < 0.05
    ]

    print("\nMULTIPLE TESTING")
    print("Numbers tested:", 45)
    print("Bonferroni threshold: 0.05")
    print("Numbers significant after correction:", len(significant))

    if not significant:
        print("No individual number passed the adjusted test.")

    return records


# ================= BAYESIAN MULTI-WINDOW SCORE =================
def number_scores(history):
    """
    Beta-binomial shrinkage toward theoretical probability 6/45.

    Prior:
        alpha = 6
        beta = 39

    The prior has mean 6/45 and reduces extreme estimates
    from short historical windows.
    """
    scores = np.zeros(45)

    for window, weight in zip(WINDOWS, WINDOW_WEIGHTS):
        sample = history[-min(window, len(history)):]

        n = len(sample)
        counts = sample.sum(axis=0)

        alpha = counts + 6
        beta = n - counts + 39

        posterior_mean = alpha / (alpha + beta)

        posterior_variance = (
            alpha * beta
            / (
                (alpha + beta) ** 2
                * (alpha + beta + 1)
            )
        )

        posterior_sd = np.sqrt(posterior_variance)

        z = (posterior_mean - P) / np.maximum(
            posterior_sd, 1e-9
        )

        scores += weight * z

    return scores


def choose_six(scores):
    indices = np.argsort(scores)[-6:]
    return sorted((indices + 1).tolist())


# ================= BLOCK STABILITY =================
def stability_analysis(matrix):
    print("\n===== V12 HISTORICAL STABILITY =====")

    block_size = len(matrix) // 4
    top_sets = []

    for i in range(4):
        start = i * block_size
        end = (i + 1) * block_size if i < 3 else len(matrix)

        block = matrix[start:end]

        counts = block.sum(axis=0)
        top = set((np.argsort(counts)[-10:] + 1).tolist())

        top_sets.append(top)

        print(
            f"Block {i + 1}: draws={len(block)}, "
            f"top10=",
            " ".join(f"{x:02d}" for x in sorted(top))
        )

    print("\nTOP-10 OVERLAP")

    for i in range(4):
        for j in range(i + 1, 4):
            print(
                f"Block {i + 1} vs {j + 1}: "
                f"{len(top_sets[i] & top_sets[j])}/10"
            )


# ================= WALK FORWARD =================
def walk_forward(matrix):
    n = len(matrix)
    test_count = min(BACKTEST_DRAWS, n - MIN_HISTORY)

    if test_count <= 0:
        raise ValueError("Not enough historical data.")

    start = n - test_count

    hits = []
    tickets = []

    print("\n===== V12 STRICT WALK-FORWARD =====")
    print("Test draws:", test_count)
    print("Training cap:", TRAIN_MAX_DRAWS)
    print("Only past data used.")

    for target in range(start, n):
        history = matrix[
            max(0, target - TRAIN_MAX_DRAWS):target
        ]

        actual = matrix[target]

        scores = number_scores(history)
        ticket = choose_six(scores)

        indices = np.array(ticket, dtype=int) - 1
        hit_count = int(actual[indices].sum())

        hits.append(hit_count)
        tickets.append(ticket)

    return np.asarray(hits), tickets


# ================= RANDOM CONTROL =================
def random_control(test_count):
    rng = np.random.default_rng(SEED)

    return rng.hypergeometric(
        ngood=6,
        nbad=39,
        nsample=6,
        size=(RANDOM_SIMULATIONS, test_count),
    )


# ================= EVALUATION =================
def confidence_interval(values):
    values = np.asarray(values, dtype=float)

    mean = values.mean()

    if len(values) < 2:
        return mean, mean

    se = values.std(ddof=1) / np.sqrt(len(values))

    return (
        mean - 1.96 * se,
        mean + 1.96 * se
    )


def evaluate(hits, control):
    hits = np.asarray(hits, dtype=int)

    random_totals = control.sum(axis=1)

    p_value = (
        np.count_nonzero(random_totals >= hits.sum()) + 1
    ) / (len(random_totals) + 1)

    paired_difference = hits - control.mean(axis=0)

    ci_low, ci_high = confidence_interval(paired_difference)

    print("\n===== V12 OUT-OF-SAMPLE RESULTS =====")

    print("Total hits:", int(hits.sum()))
    print("Average hits:", round(hits.mean(), 4))

    low, high = confidence_interval(hits)

    print(
        "95% CI average:",
        round(low, 4),
        round(high, 4)
    )

    print(
        "Random average:",
        round(control.mean(), 4)
    )

    print(
        "Paired difference:",
        round(paired_difference.mean(), 4)
    )

    print(
        "95% CI paired difference:",
        round(ci_low, 4),
        round(ci_high, 4)
    )

    print("Random upper-tail p:", round(p_value, 4))

    print("\nHIT DISTRIBUTION")

    for k in range(7):
        print(f"{k} hits:", int(np.sum(hits == k)))

    print("2+ hits:", int(np.sum(hits >= 2)))
    print("3+ hits:", int(np.sum(hits >= 3)))
    print("4+ hits:", int(np.sum(hits >= 4)))
    print("5+ hits:", int(np.sum(hits >= 5)))
    print("6 hits:", int(np.sum(hits == 6)))

    return p_value


# ================= PERIOD VALIDATION =================
def period_analysis(hits, control):
    print("\n===== V12 PERIOD VALIDATION =====")

    size = len(hits)

    for i in range(3):
        start = i * size // 3
        end = (i + 1) * size // 3

        segment = hits[start:end]
        random_segment = control[:, start:end]

        difference = segment - random_segment.mean(axis=0)

        low, high = confidence_interval(difference)

        print(
            f"Period {i + 1}: "
            f"draws={len(segment)}, "
            f"model_mean={segment.mean():.4f}, "
            f"random_mean={random_segment.mean():.4f}, "
            f"paired_diff={difference.mean():.4f}, "
            f"CI=[{low:.4f}, {high:.4f}], "
            f"zero={np.sum(segment == 0)}, "
            f"2+={np.sum(segment >= 2)}"
        )


# ================= LATEST CANDIDATE =================
def latest_candidate(matrix):
    history = matrix[-TRAIN_MAX_DRAWS:]

    scores = number_scores(history)
    ticket = choose_six(scores)

    print("\n===== V12 LATEST NUMBER RANKING =====")

    ranking = np.argsort(scores)[::-1]

    for index in ranking[:15]:
        print(
            f"{index + 1:02d}: score={scores[index]:.5f}"
        )

    print("\nV12 EXPERIMENTAL CANDIDATE:")
    print(" - ".join(f"{x:02d}" for x in ticket))

    print("Ranking only. No proven predictive advantage.")


# ================= MAIN =================
def main():
    print("===== AI MEGA 6/45 V12 =====")
    print("Loading historical data...")

    matrix, draws = load_matrix()

    print("Valid draws:", len(matrix))
    print(
        "Latest draw:",
        draws[-1].get("draw_id", draws[-1].get("id", "unknown"))
    )
    print("Latest date:", draws[-1].get("date", "unknown"))

    integrity_check(matrix, draws)

    individual_statistics(matrix)
    stability_analysis(matrix)

    hits, tickets = walk_forward(matrix)

    control = random_control(len(hits))

    print("\n===== RANDOM CONTROL =====")
    print("Simulations:", RANDOM_SIMULATIONS)
    print("Random mean:", round(control.mean(), 4))
    print("Theoretical mean: 0.8000")

    p_value = evaluate(hits, control)

    period_analysis(hits, control)

    print("\n===== V12 MULTIPLE TEST NOTE =====")
    print("One primary walk-forward strategy.")
    print("Individual historical tests use Bonferroni correction.")
    print("Do not treat historical significance as predictive proof.")

    latest_candidate(matrix)

    print("\n===== AI MEGA 6/45 V12 COMPLETE =====")
    print("Research only.")
    print("Historical patterns do not establish predictive advantage.")


if __name__ == "__main__":
    main()
