
"""
AI Vietlott Mega 6/45 V8
Individual-number research:
- Frequency
- Current gap
- Appearance intervals
- Recent trends
- Previous-draw transition
- Time-block stability
- Strict walk-forward backtest

Research only. No proven predictive advantage.
"""

from collections import Counter
from statistics import mean
import numpy as np

from main_v5 import (
    download_data,
    statistical_scores,
    rank_scores,
    monte_carlo_random,
    confidence_interval,
    BACKTEST_DRAWS,
    RANDOM_SIMULATIONS,
    SEED,
)

TRAIN_MAX_DRAWS = 500

WINDOWS = [10, 30, 50, 100, 200, 500]

# Longer windows receive more weight.
SCORE_WINDOWS = [30, 100, 300, 500]
SCORE_WEIGHTS = [0.10, 0.20, 0.30, 0.40]

EXPECTED_RATE = 6 / 45


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

def number_matrix(draws):
    """Convert draws into a binary matrix: rows=draws, columns=01..45."""
    matrix = np.zeros((len(draws), 45), dtype=np.int8)

    for i, draw in enumerate(draws):
        for number in draw["result"]:
            matrix[i, int(number) - 1] = 1

    return matrix


# --------------------------------------------------
# 1. INDIVIDUAL NUMBER STATISTICS
# --------------------------------------------------

def calculate_number_statistics(matrix):
    total_draws = len(matrix)
    stats = {}

    for number in range(1, 46):
        column = matrix[:, number - 1]
        positions = np.flatnonzero(column)

        total_hits = len(positions)

        # Number of draws since its most recent appearance.
        if total_hits == 0:
            current_gap = total_draws
        else:
            current_gap = total_draws - 1 - int(positions[-1])

        # Intervals between appearances.
        if total_hits >= 2:
            intervals = np.diff(positions)
            average_interval = float(np.mean(intervals))
            median_interval = float(np.median(intervals))
            interval_std = float(np.std(intervals))
        else:
            average_interval = None
            median_interval = None
            interval_std = None

        window_counts = {}
        window_rates = {}

        for window in WINDOWS:
            actual_window = min(window, total_draws)
            count = int(column[-actual_window:].sum())

            window_counts[window] = count
            window_rates[window] = count / actual_window

        # Recent trend: compare latest 30 with the previous 30.
        if total_draws >= 60:
            recent_30 = float(column[-30:].mean())
            previous_30 = float(column[-60:-30].mean())
            trend_30 = recent_30 - previous_30
        else:
            recent_30 = None
            previous_30 = None
            trend_30 = None

        # Appearance in the immediately previous draw.
        previous_draw = int(column[-1]) if total_draws else 0

        # Appearance rate after the previous draw contained this number.
        if total_draws >= 2:
            previous_present = column[:-1] == 1
            next_values = column[1:]

            if previous_present.sum() > 0:
                next_rate_if_present = float(
                    next_values[previous_present].mean()
                )
            else:
                next_rate_if_present = None

            if (~previous_present).sum() > 0:
                next_rate_if_absent = float(
                    next_values[~previous_present].mean()
                )
            else:
                next_rate_if_absent = None
        else:
            next_rate_if_present = None
            next_rate_if_absent = None

        stats[number] = {
            "total": total_hits,
            "rate": total_hits / total_draws,
            "gap": current_gap,
            "average_interval": average_interval,
            "median_interval": median_interval,
            "interval_std": interval_std,
            "window_counts": window_counts,
            "window_rates": window_rates,
            "recent_30": recent_30,
            "previous_30": previous_30,
            "trend_30": trend_30,
            "previous_draw": previous_draw,
            "next_rate_if_present": next_rate_if_present,
            "next_rate_if_absent": next_rate_if_absent,
        }

    return stats


def print_number_report(matrix):
    stats = calculate_number_statistics(matrix)
    total_draws = len(matrix)

    print("\n===== V8 INDIVIDUAL NUMBER ANALYSIS =====")
    print("Total draws:", total_draws)
    print("Numbers analyzed: 45")
    print("Theoretical appearance rate:", round(EXPECTED_RATE, 4))

    print("\n===== TOP 15 HISTORICAL FREQUENCY =====")
    ranked = sorted(
        stats.items(),
        key=lambda item: item[1]["total"],
        reverse=True,
    )

    for number, data in ranked[:15]:
        print(
            f"{number:02d}: total={data['total']} "
            f"rate={data['rate']:.4f} "
            f"gap={data['gap']}"
        )

    print("\n===== 15 NUMBERS WITH LOWEST FREQUENCY =====")
    for number, data in sorted(
        stats.items(),
        key=lambda item: item[1]["total"],
    )[:15]:
        print(
            f"{number:02d}: total={data['total']} "
            f"rate={data['rate']:.4f} "
            f"gap={data['gap']}"
        )

    print("\n===== CURRENT GAP RANKING =====")
    for number, data in sorted(
        stats.items(),
        key=lambda item: item[1]["gap"],
        reverse=True,
    )[:15]:
        print(
            f"{number:02d}: gap={data['gap']} "
            f"total={data['total']}"
        )

    print("\n===== RECENT WINDOW FREQUENCIES =====")
    print("Number | " + " | ".join(str(w) for w in WINDOWS))

    for number in range(1, 46):
        values = [
            str(stats[number]["window_counts"][w])
            for w in WINDOWS
        ]
        print(f"{number:02d}     | " + " | ".join(values))

    print("\n===== INTERVAL ANALYSIS =====")
    for number, data in sorted(
        stats.items(),
        key=lambda item: (
            item[1]["average_interval"]
            if item[1]["average_interval"] is not None
            else 9999
        ),
    )[:15]:
        if data["average_interval"] is not None:
            print(
                f"{number:02d}: avg_interval="
                f"{data['average_interval']:.2f}, "
                f"median={data['median_interval']:.2f}, "
                f"std={data['interval_std']:.2f}"
            )

    print("\n===== RECENT TREND =====")
    for number, data in sorted(
        stats.items(),
        key=lambda item: (
            item[1]["trend_30"]
            if item[1]["trend_30"] is not None
            else -999
        ),
        reverse=True,
    )[:15]:
        if data["trend_30"] is not None:
            print(
                f"{number:02d}: recent30="
                f"{data['recent_30']:.3f}, "
                f"previous30={data['previous_30']:.3f}, "
                f"change={data['trend_30']:+.3f}"
            )

    print("\n===== PREVIOUS-DRAW TRANSITION =====")
    for number, data in stats.items():
        if data["next_rate_if_present"] is not None:
            print(
                f"{number:02d}: "
                f"after-present={data['next_rate_if_present']:.3f}, "
                f"after-absent={data['next_rate_if_absent']:.3f}"
            )


# --------------------------------------------------
# 2. TIME-BLOCK STABILITY
# --------------------------------------------------

def stability_report(matrix):
    print("\n===== V8 FREQUENCY STABILITY =====")

    blocks = np.array_split(np.arange(len(matrix)), 4)
    top_sets = []

    for block_number, indices in enumerate(blocks, 1):
        block = matrix[indices]
        frequencies = block.sum(axis=0)

        top = np.argsort(frequencies)[::-1][:10] + 1
        top_set = set(int(x) for x in top)
        top_sets.append(top_set)

        print(
            f"\nBLOCK {block_number} "
            f"({len(indices)} draws)"
        )
        print("Top 10:", " ".join(f"{x:02d}" for x in top))

    print("\nTOP-10 OVERLAP")

    for i in range(4):
        for j in range(i + 1, 4):
            overlap = len(top_sets[i] & top_sets[j])
            print(f"Block {i+1} vs {j+1}: {overlap}/10")


# --------------------------------------------------
# 3. V8 NUMBER RANKING
# --------------------------------------------------

def v8_scores(history):
    """
    Multi-window frequency ranking with shrinkage.

    Gap and interval are reported as diagnostics, not assumed
    to predict the next draw.
    """

    matrix = number_matrix(history[-TRAIN_MAX_DRAWS:])
    scores = np.zeros(45, dtype=float)

    for window, weight in zip(SCORE_WINDOWS, SCORE_WEIGHTS):
        actual_window = min(window, len(matrix))
        sample = matrix[-actual_window:]

        counts = sample.sum(axis=0).astype(float)
        observed_rate = counts / actual_window

        # Shrink noisy short-window estimates toward the
        # theoretical 6/45 rate.
        shrinkage_strength = 60.0
        reliability = actual_window / (
            actual_window + shrinkage_strength
        )

        adjusted_rate = (
            reliability * observed_rate
            + (1 - reliability) * EXPECTED_RATE
        )

        # Standardized deviation from the theoretical rate.
        variance = (
            EXPECTED_RATE * (1 - EXPECTED_RATE)
            / actual_window
        )

        z_scores = (
            adjusted_rate - EXPECTED_RATE
        ) / np.sqrt(variance)

        scores += weight * z_scores

    return scores


def v8_strategy(history):
    scores = v8_scores(history)
    return set(np.argsort(scores)[-6:] + 1)


# --------------------------------------------------
# 4. STRICT WALK-FORWARD BACKTEST
# --------------------------------------------------

def backtest_v8(draws):
    start = len(draws) - BACKTEST_DRAWS

    strategies = {
        "V5 STATISTICS": lambda history: set(
            rank_scores(
                statistical_scores(
                    history[-TRAIN_MAX_DRAWS:]
                )
            )[:6]
        ),
        "V8 INDIVIDUAL NUMBERS": v8_strategy,
    }

    results = {
        name: [] for name in strategies
    }

    actual_sets = []

    print("\n===== V8 STRICT WALK-FORWARD =====")
    print("Test draws:", BACKTEST_DRAWS)
    print("Training cap:", TRAIN_MAX_DRAWS)
    print("Prediction uses only previous draws.")

    for i in range(start, len(draws)):
        history = draws[:i]
        actual = set(draws[i]["result"])

        actual_sets.append(actual)

        for name, strategy in strategies.items():
            prediction = strategy(history)
            results[name].append(
                len(prediction & actual)
            )

    random_totals = monte_carlo_random(actual_sets)

    print("\nTheoretical random expectation: 0.8000")
    print(
        "Monte Carlo random mean:",
        round(
            float(random_totals.mean() / BACKTEST_DRAWS),
            4,
        ),
    )

    for name, hits in results.items():
        total = sum(hits)
        average = mean(hits)
        low, high = confidence_interval(hits)

        p_value = (
            1 + int(np.sum(random_totals >= total))
        ) / (RANDOM_SIMULATIONS + 1)

        print(f"\n===== {name} =====")
        print("Total hits:", total)
        print("Average hits:", round(average, 4))
        print(
            "95% CI:",
            round(low, 4),
            round(high, 4),
        )
        print(
            "Difference vs random:",
            round(average - 0.8, 4),
        )
        print(
            "Random upper-tail p:",
            round(float(p_value), 4),
        )

        for block in range(3):
            segment = hits[
                block * 100:(block + 1) * 100
            ]

            print(
                f"Period {block+1}: "
                f"mean={mean(segment):.3f}, "
                f"zero={segment.count(0)}, "
                f"2+={sum(x >= 2 for x in segment)}"
            )

    return results


# --------------------------------------------------
# 5. LATEST EXPERIMENTAL CANDIDATE
# --------------------------------------------------

def latest_candidate(draws):
    history = draws[-TRAIN_MAX_DRAWS:]
    matrix = number_matrix(history)
    stats = calculate_number_statistics(matrix)

    scores = v8_scores(history)
    selected = np.argsort(scores)[-6:] + 1

    print("\n===== V8 EXPERIMENTAL CANDIDATE =====")
    print(" - ".join(
        f"{int(x):02d}" for x in sorted(selected)
    ))

    print("\n===== V8 SELECTED NUMBER DETAILS =====")

    for number in sorted(int(x) for x in selected):
        data = stats[number]

        print(
            f"{number:02d}: "
            f"total={data['total']}, "
            f"gap={data['gap']}, "
            f"trend30={data['trend_30']}, "
            f"score={scores[number-1]:.4f}"
        )

    print("\nExperimental ranking only.")
    print("Not a claim of higher probability.")


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    draws = download_data()

    print("\n===== DATA =====")
    print("Valid draws:", len(draws))

    latest = draws[-1]
    print("Latest draw:", latest.get("draw_id", "unknown"))
    print("Latest date:", latest.get("date", "unknown"))

    matrix = number_matrix(draws)

    print_number_report(matrix)
    stability_report(matrix)
    backtest_v8(draws)
    latest_candidate(draws)

    print("\n===== AI MEGA 6/45 V8 COMPLETE =====")
    print("Research only.")
    print("Historical patterns do not establish predictive advantage.")


if __name__ == "__main__":
    main()
