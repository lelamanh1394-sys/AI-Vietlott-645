
import math
import numpy as np

from main_v5 import download_data

# =====================================================
# AI MEGA 6/45 - V13
# Exact tests + probability calibration + walk-forward
# =====================================================

BACKTEST_DRAWS = 300
TRAIN_MAX_DRAWS = 500
MIN_HISTORY = 200

RANDOM_SIMULATIONS = 5000
SEED = 6452026

WINDOWS = [30, 100, 300, 500]
WINDOW_WEIGHTS = [0.10, 0.20, 0.30, 0.40]

P0 = 6 / 45
EPS = 1e-12


# ===================== DATA =====================

def load_matrix():
    draws = download_data()

    matrix = []
    valid_draws = []

    for draw in draws:
        numbers = draw.get("result", [])

        try:
            nums = [int(x) for x in numbers]
        except (TypeError, ValueError):
            continue

        if len(nums) != 6:
            continue

        if len(set(nums)) != 6:
            continue

        if any(x < 1 or x > 45 for x in nums):
            continue

        row = np.zeros(45, dtype=np.int8)
        row[np.asarray(nums, dtype=int) - 1] = 1

        matrix.append(row)
        valid_draws.append(draw)

    if not matrix:
        raise ValueError("No valid Mega 6/45 draws found.")

    return np.asarray(matrix), valid_draws


def check_data(matrix, draws):
    print("\n===== V13 DATA INTEGRITY =====")

    ids = [
        str(d.get("draw_id", d.get("id", "")))
        for d in draws
    ]
    ids = [x for x in ids if x]

    dates = [
        str(d.get("date", ""))
        for d in draws
        if d.get("date")
    ]

    print("Valid draws:", len(matrix))
    print("Draw IDs available:", len(ids))

    if ids:
        duplicate_ids = len(ids) - len(set(ids))
        print("Duplicate IDs:", duplicate_ids)

        if duplicate_ids:
            raise ValueError("Duplicate draw IDs detected.")

    if dates:
        print("First date:", dates[0])
        print("Last date:", dates[-1])
        print("Chronological order:", dates == sorted(dates))

        if dates != sorted(dates):
            raise ValueError("Draw dates are not oldest-first.")

    if np.any(matrix.sum(axis=1) != 6):
        raise ValueError("A draw does not contain exactly six numbers.")

    if len(matrix) < MIN_HISTORY + 1:
        raise ValueError("Not enough historical data.")


# ===================== EXACT BINOMIAL TEST =====================

def log_binomial_pmf(k, n, p):
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log(1 - p)
    )


def exact_binomial_two_sided(k, n, p):
    """
    Probability-ordering two-sided exact binomial test.
    Sums probabilities of outcomes no more likely
    than the observed outcome.
    """
    observed_log_p = log_binomial_pmf(k, n, p)

    total = 0.0

    for value in range(n + 1):
        lp = log_binomial_pmf(value, n, p)

        if lp <= observed_log_p + EPS:
            total += math.exp(lp)

    return min(1.0, total)


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)

    order = np.argsort(p_values)
    adjusted = np.ones(m, dtype=float)

    running_min = 1.0

    for rank_index in range(m - 1, -1, -1):
        original_index = order[rank_index]
        rank = rank_index + 1

        candidate = p_values[original_index] * m / rank
        running_min = min(running_min, candidate)

        adjusted[original_index] = min(1.0, running_min)

    return adjusted


def individual_number_tests(matrix):
    print("\n===== V13 EXACT INDIVIDUAL NUMBER TESTS =====")

    n = len(matrix)
    counts = matrix.sum(axis=0)

    expected = n * P0
    sd = math.sqrt(n * P0 * (1 - P0))

    raw_p = np.array([
        exact_binomial_two_sided(int(count), n, P0)
        for count in counts
    ])

    q_values = benjamini_hochberg(raw_p)

    records = []

    for i in range(45):
        count = int(counts[i])

        records.append({
            "number": i + 1,
            "count": count,
            "rate": count / n,
            "z": (count - expected) / sd,
            "p": raw_p[i],
            "q": q_values[i],
        })

    print("Draws:", n)
    print("Expected appearances:", round(expected, 3))
    print("Theoretical probability:", round(P0, 6))

    print("\nTOP 15 FREQUENCIES")

    for r in sorted(records, key=lambda x: x["count"], reverse=True)[:15]:
        print(
            f'{r["number"]:02d}: '
            f'count={r["count"]}, '
            f'rate={r["rate"]:.4f}, '
            f'z={r["z"]:.3f}, '
            f'p={r["p"]:.5f}, '
            f'BH_q={r["q"]:.5f}'
        )

    print("\n15 LOWEST FREQUENCIES")

    for r in sorted(records, key=lambda x: x["count"])[:15]:
        print(
            f'{r["number"]:02d}: '
            f'count={r["count"]}, '
            f'rate={r["rate"]:.4f}, '
            f'z={r["z"]:.3f}, '
            f'p={r["p"]:.5f}, '
            f'BH_q={r["q"]:.5f}'
        )

    significant = [
        r for r in records
        if r["q"] < 0.05
    ]

    print("\nBENJAMINI-HOCHBERG")
    print("Tests:", 45)
    print("FDR threshold: 0.05")
    print("Significant numbers:", len(significant))

    if significant:
        for r in significant:
            print(
                f'Number {r["number"]:02d}: '
                f'BH q={r["q"]:.6f}'
            )
    else:
        print("No number passed the BH threshold.")

    return records


# ===================== PROBABILITY MODEL =====================

def predict_probabilities(history):
    """
    Multi-window Beta-Binomial estimates.

    Prior mean = 6/45.
    Prior strength = 45 observations.

    The result estimates each number's marginal
    probability; it does not assume a proven bias.
    """
    probabilities = np.zeros(45, dtype=float)

    for window, weight in zip(WINDOWS, WINDOW_WEIGHTS):
        sample = history[-min(window, len(history)):]

        n = len(sample)
        counts = sample.sum(axis=0)

        alpha = counts + 6
        beta = n - counts + 39

        posterior_mean = alpha / (alpha + beta)

        probabilities += weight * posterior_mean

    return probabilities


def choose_six(probabilities):
    indices = np.argsort(probabilities)[-6:]
    return sorted((indices + 1).tolist())


# ===================== CONFIDENCE INTERVAL =====================

def confidence_interval(values):
    values = np.asarray(values, dtype=float)

    mean = float(np.mean(values))

    if len(values) < 2:
        return mean, mean

    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))

    return mean - 1.96 * se, mean + 1.96 * se


# ===================== WALK-FORWARD =====================

def walk_forward(matrix):
    n = len(matrix)
    test_count = min(BACKTEST_DRAWS, n - MIN_HISTORY)

    if test_count <= 0:
        raise ValueError("Not enough data for walk-forward.")

    start = n - test_count

    hits = []
    model_brier = []
    baseline_brier = []
    model_logloss = []
    baseline_logloss = []
    all_probabilities = []
    all_outcomes = []
    tickets = []

    print("\n===== V13 STRICT WALK-FORWARD =====")
    print("Test draws:", test_count)
    print("Training cap:", TRAIN_MAX_DRAWS)
    print("Features use past draws only.")

    for target in range(start, n):
        history = matrix[
            max(0, target - TRAIN_MAX_DRAWS):target
        ]

        actual = matrix[target].astype(float)

        probabilities = predict_probabilities(history)
        ticket = choose_six(probabilities)

        ticket_indices = np.asarray(ticket, dtype=int) - 1
        hit_count = int(actual[ticket_indices].sum())

        probabilities = np.clip(
            probabilities, 1e-9, 1 - 1e-9
        )

        baseline_probabilities = np.full(45, P0)

        brier = np.mean((probabilities - actual) ** 2)
        baseline_brier_score = np.mean(
            (baseline_probabilities - actual) ** 2
        )

        logloss = -np.mean(
            actual * np.log(probabilities)
            + (1 - actual) * np.log(1 - probabilities)
        )

        baseline_logloss_score = -np.mean(
            actual * np.log(P0)
            + (1 - actual) * np.log(1 - P0)
        )

        hits.append(hit_count)
        model_brier.append(brier)
        baseline_brier.append(baseline_brier_score)
        model_logloss.append(logloss)
        baseline_logloss.append(baseline_logloss_score)

        all_probabilities.extend(probabilities.tolist())
        all_outcomes.extend(actual.tolist())
        tickets.append(ticket)

    return {
        "hits": np.asarray(hits, dtype=int),
        "model_brier": np.asarray(model_brier),
        "baseline_brier": np.asarray(baseline_brier),
        "model_logloss": np.asarray(model_logloss),
        "baseline_logloss": np.asarray(baseline_logloss),
        "probabilities": np.asarray(all_probabilities),
        "outcomes": np.asarray(all_outcomes),
        "tickets": tickets,
    }


# ===================== RANDOM CONTROL =====================

def random_control(test_count):
    rng = np.random.default_rng(SEED)

    return rng.hypergeometric(
        ngood=6,
        nbad=39,
        nsample=6,
        size=(RANDOM_SIMULATIONS, test_count),
    )


# ===================== CALIBRATION =====================

def calibration_report(probabilities, outcomes):
    print("\n===== V13 PROBABILITY CALIBRATION =====")

    probabilities = np.asarray(probabilities)
    outcomes = np.asarray(outcomes)

    edges = np.linspace(0, 1, 6)

    print("Bin | Count | Mean predicted | Actual rate")

    for i in range(len(edges) - 1):
        if i == len(edges) - 2:
            mask = (
                (probabilities >= edges[i])
                & (probabilities <= edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= edges[i])
                & (probabilities < edges[i + 1])
            )

        if not np.any(mask):
            continue

        print(
            f"{edges[i]:.1f}-{edges[i + 1]:.1f} | "
            f"{int(mask.sum())} | "
            f"{probabilities[mask].mean():.4f} | "
            f"{outcomes[mask].mean():.4f}"
        )


# ===================== EVALUATION =====================

def evaluate(results, control):
    hits = results["hits"]

    random_totals = control.sum(axis=1)

    p_value = (
        np.count_nonzero(random_totals >= hits.sum()) + 1
    ) / (len(random_totals) + 1)

    paired_difference = hits - control.mean(axis=0)

    diff_low, diff_high = confidence_interval(
        paired_difference
    )

    brier_difference = (
        results["model_brier"] - results["baseline_brier"]
    )

    logloss_difference = (
        results["model_logloss"] - results["baseline_logloss"]
    )

    brier_low, brier_high = confidence_interval(
        brier_difference
    )

    logloss_low, logloss_high = confidence_interval(
        logloss_difference
    )

    print("\n===== V13 OUT-OF-SAMPLE RESULTS =====")

    print("Test draws:", len(hits))
    print("Total ticket hits:", int(hits.sum()))
    print("Average ticket hits:", round(hits.mean(), 4))

    low, high = confidence_interval(hits)

    print(
        "95% CI average hits:",
        round(low, 4),
        round(high, 4)
    )

    print("Random mean:", round(control.mean(), 4))
    print("Theoretical random mean: 0.8000")

    print("Paired hit difference:", round(paired_difference.mean(), 4))
    print(
        "Paired difference CI:",
        round(diff_low, 4),
        round(diff_high, 4)
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

    print("\nBrier score: lower is better")
    print("Model:", round(results["model_brier"].mean(), 6))
    print("Baseline:", round(results["baseline_brier"].mean(), 6))
    print(
        "Model - baseline:",
        round(brier_difference.mean(), 6)
    )
    print(
        "Difference CI:",
        round(brier_low, 6),
        round(brier_high, 6)
    )

    print("\nLog loss: lower is better")
    print("Model:", round(results["model_logloss"].mean(), 6))
    print("Baseline:", round(results["baseline_logloss"].mean(), 6))
    print(
        "Model - baseline:",
        round(logloss_difference.mean(), 6)
    )
    print(
        "Difference CI:",
        round(logloss_low, 6),
        round(logloss_high, 6)
    )

    calibration_report(
        results["probabilities"],
        results["outcomes"]
    )

    return {
        "p_value": p_value,
        "hit_difference": paired_difference.mean(),
        "hit_ci": (diff_low, diff_high),
        "brier_difference": brier_difference.mean(),
        "logloss_difference": logloss_difference.mean(),
    }


# ===================== PERIOD VALIDATION =====================

def period_analysis(results, control):
    hits = results["hits"]

    print("\n===== V13 PERIOD VALIDATION =====")

    n = len(hits)

    for i in range(3):
        start = i * n // 3
        end = (i + 1) * n // 3

        segment = hits[start:end]
        random_segment = control[:, start:end]

        difference = segment - random_segment.mean(axis=0)

        low, high = confidence_interval(difference)

        print(
            f"Period {i + 1}: "
            f"draws={len(segment)}, "
            f"model={segment.mean():.4f}, "
            f"random={random_segment.mean():.4f}, "
            f"diff={difference.mean():.4f}, "
            f"CI=[{low:.4f}, {high:.4f}], "
            f"2+={np.sum(segment >= 2)}, "
            f"3+={np.sum(segment >= 3)}"
        )


# ===================== LATEST CANDIDATE =====================

def latest_candidate(matrix):
    history = matrix[-TRAIN_MAX_DRAWS:]

    probabilities = predict_probabilities(history)
    ranking = np.argsort(probabilities)[::-1]

    ticket = choose_six(probabilities)

    print("\n===== V13 LATEST PROBABILITY RANKING =====")

    for index in ranking[:15]:
        print(
            f"{index + 1:02d}: "
            f"estimated_probability={probabilities[index]:.6f}"
        )

    print("\nV13 EXPERIMENTAL CANDIDATE:")
    print(" - ".join(f"{x:02d}" for x in ticket))

    print(
        "Reference probability for each number:",
        round(P0, 6)
    )

    print("Candidate is a model ranking, not a prediction guarantee.")


# ===================== MAIN =====================

def main():
    print("===== AI MEGA 6/45 V13 =====")
    print("Loading historical data...")

    matrix, draws = load_matrix()

    print("Valid draws:", len(matrix))
    print(
        "Latest draw:",
        draws[-1].get("draw_id", draws[-1].get("id", "unknown"))
    )
    print("Latest date:", draws[-1].get("date", "unknown"))

    check_data(matrix, draws)

    individual_number_tests(matrix)

    results = walk_forward(matrix)

    control = random_control(len(results["hits"]))

    print("\n===== RANDOM CONTROL =====")
    print("Simulations:", RANDOM_SIMULATIONS)
    print("Random mean:", round(control.mean(), 4))
    print("Theoretical mean: 0.8000")

    summary = evaluate(results, control)

    period_analysis(results, control)

    latest_candidate(matrix)

    print("\n===== V13 FINAL RESEARCH CHECK =====")

    print(
        "Ticket comparison p-value:",
        round(summary["p_value"], 4)
    )

    if (
        summary["p_value"] < 0.05
        and summary["hit_ci"][0] > 0
    ):
        print(
            "Potential signal detected. "
            "Independent replication is still required."
        )
    else:
        print(
            "No statistically convincing ticket advantage detected."
        )

    if summary["brier_difference"] < 0:
        print("Brier score: model numerically better than baseline.")
    else:
        print("Brier score: model not better than baseline.")

    if summary["logloss_difference"] < 0:
        print("Log loss: model numerically better than baseline.")
    else:
        print("Log loss: model not better than baseline.")

    print("\n===== AI MEGA 6/45 V13 COMPLETE =====")
    print("Research only. No guaranteed prediction.")


if __name__ == "__main__":
    main()
