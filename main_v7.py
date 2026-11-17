
"""AI Vietlott Mega 6/45 V7
Pair significance screening + temporal stability + strict walk-forward.
Research only. No demonstrated predictive advantage.
"""

from collections import Counter
from itertools import combinations
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
PAIR_PROB = 15 / 990
PAIR_SHRINKAGE = 30.0
NULL_SIMULATIONS = 1000


# --------------------------------------------------
# 1. PAIR COUNTING
# --------------------------------------------------

ALL_PAIRS = list(combinations(range(1, 46), 2))
PAIR_INDEX = {pair: i for i, pair in enumerate(ALL_PAIRS)}

PAIR_LOOKUP = np.full((46, 46), -1, dtype=np.int16)

for pair, index in PAIR_INDEX.items():
    a, b = pair
    PAIR_LOOKUP[a, b] = index
    PAIR_LOOKUP[b, a] = index


def count_pairs(draws):
    counts = np.zeros(990, dtype=np.int32)

    for draw in draws:
        nums = sorted(draw["result"])
        for pair in combinations(nums, 2):
            counts[PAIR_INDEX[pair]] += 1

    return counts


def pair_null_test(draws):
    """Compare the largest observed pair count with random 6/45 draws."""

    n = len(draws)
    counts = count_pairs(draws)

    expected = n * PAIR_PROB
    observed_max = int(counts.max())

    rng = np.random.default_rng(SEED + 707)
    simulated_max = np.zeros(NULL_SIMULATIONS, dtype=np.int32)

    # Simulate complete random 6-number draws.
    for s in range(NULL_SIMULATIONS):
        scores = rng.random((n, 45))
        tickets = np.argpartition(scores, 6, axis=1)[:, :6] + 1

        simulated_counts = np.zeros(990, dtype=np.int32)

        for i, j in combinations(range(6), 2):
            ids = PAIR_LOOKUP[tickets[:, i], tickets[:, j]]
            simulated_counts += np.bincount(
                ids, minlength=990
            ).astype(np.int32)

        simulated_max[s] = simulated_counts.max()

    p_value = (
        1 + np.sum(simulated_max >= observed_max)
    ) / (NULL_SIMULATIONS + 1)

    print("\n===== V7 PAIR SIGNIFICANCE TEST =====")
    print("Historical draws:", n)
    print("Expected count per fixed pair:", round(expected, 2))
    print("Largest observed pair count:", observed_max)
    print("Random maximum-pair mean:", round(float(simulated_max.mean()), 2))
    print("Random maximum-pair 95% range:",
          np.percentile(simulated_max, [2.5, 97.5]).round(2).tolist())
    print("Monte Carlo p-value:", round(float(p_value), 4))

    print("\nTOP 15 HISTORICAL PAIRS")
    top_indices = np.argsort(counts)[::-1][:15]

    for idx in top_indices:
        a, b = ALL_PAIRS[idx]
        print(
            f"{a:02d}-{b:02d}: {counts[idx]} "
            f"| expected={expected:.2f}"
        )

    return counts


# --------------------------------------------------
# 2. PAIR STABILITY OVER TIME
# --------------------------------------------------

def pair_stability_report(draws):
    print("\n===== V7 PAIR STABILITY BY TIME BLOCK =====")

    blocks = np.array_split(np.arange(len(draws)), 4)
    top_sets = []

    for block_number, indices in enumerate(blocks, 1):
        block_draws = [draws[int(i)] for i in indices]
        counts = count_pairs(block_draws)

        top_indices = np.argsort(counts)[::-1][:10]
        top_pairs = [ALL_PAIRS[i] for i in top_indices]

        top_sets.append(set(top_pairs))

        print(f"\nBLOCK {block_number} | {len(indices)} draws")
        print(", ".join(
            f"{a:02d}-{b:02d}" for a, b in top_pairs[:10]
        ))

    print("\nTOP-10 OVERLAP")
    for i in range(4):
        for j in range(i + 1, 4):
            overlap = len(top_sets[i] & top_sets[j])
            print(f"Block {i+1} vs {j+1}: {overlap}/10")


# --------------------------------------------------
# 3. V6 PAIR STRATEGY
# --------------------------------------------------

def v6_strategy(history):
    hist = history[-TRAIN_MAX_DRAWS:]
    n_draws = len(hist)

    singles = Counter()
    pairs = Counter()

    for draw in hist:
        nums = sorted(draw["result"])
        singles.update(nums)
        pairs.update(combinations(nums, 2))

    expected = n_draws * PAIR_PROB
    pair_scores = {}

    for a, b in ALL_PAIRS:
        observed = pairs[(a, b)]
        raw = np.log((observed + 0.5) / (expected + 0.5))

        pair_scores[(a, b)] = (
            raw * observed / (observed + 20.0)
        )

    freq_scores = {
        n: singles[n] / max(n_draws, 1) - 6 / 45
        for n in range(1, 46)
    }

    chosen = []
    remaining = set(range(1, 46))

    for _ in range(6):
        best_number = None
        best_score = -1e9

        for candidate in remaining:
            if not chosen:
                synergy = 0.0
            else:
                synergy = mean(
                    pair_scores[tuple(sorted((candidate, old)))]
                    for old in chosen
                )

            score = 0.25 * freq_scores[candidate] + 0.75 * synergy

            if score > best_score:
                best_score = score
                best_number = candidate

        chosen.append(best_number)
        remaining.remove(best_number)

    return set(chosen)


# --------------------------------------------------
# 4. V7 STABLE PAIR STRATEGY
# --------------------------------------------------

def v7_strategy(history):
    hist = history[-TRAIN_MAX_DRAWS:]

    windows = [
        hist[-100:],
        hist[-300:],
        hist[-500:],
    ]

    singles = Counter()
    for draw in hist:
        singles.update(draw["result"])

    pair_scores = {}

    for a, b in ALL_PAIRS:
        window_scores = []

        for window in windows:
            counts = count_pairs(window)
            observed = int(counts[PAIR_INDEX[(a, b)]])
            expected = len(window) * PAIR_PROB

            raw = np.log(
                (observed + 0.5) / (expected + 0.5)
            )

            shrunk = raw * observed / (
                observed + PAIR_SHRINKAGE
            )

            window_scores.append(shrunk)

        # Reward a pair only when its signal is reasonably stable.
        average_signal = mean(window_scores)
        instability = float(np.std(window_scores))

        pair_scores[(a, b)] = (
            average_signal - 0.5 * instability
        )

    freq_scores = {
        n: singles[n] / max(len(hist), 1) - 6 / 45
        for n in range(1, 46)
    }

    chosen = []
    remaining = set(range(1, 46))

    for _ in range(6):
        best_number = None
        best_score = -1e9

        for candidate in remaining:
            if not chosen:
                synergy = 0.0
            else:
                synergy = mean(
                    pair_scores[tuple(sorted((candidate, old)))]
                    for old in chosen
                )

            score = 0.20 * freq_scores[candidate] + 0.80 * synergy

            if score > best_score:
                best_score = score
                best_number = candidate

        chosen.append(best_number)
        remaining.remove(best_number)

    return set(chosen)


# --------------------------------------------------
# 5. SAME-PERIOD STRICT WALK-FORWARD COMPARISON
# --------------------------------------------------

def backtest_v7(draws):
    start = len(draws) - BACKTEST_DRAWS

    strategies = {
        "V5 STATISTICS": lambda h: set(
            rank_scores(statistical_scores(h[-TRAIN_MAX_DRAWS:]))[:6]
        ),
        "V6 PAIR": v6_strategy,
        "V7 STABLE PAIR": v7_strategy,
    }

    results = {name: [] for name in strategies}
    actual_sets = []

    print("\n===== V7 STRICT WALK-FORWARD BACKTEST =====")
    print("Test draws:", BACKTEST_DRAWS)
    print("Training cap:", TRAIN_MAX_DRAWS)
    print("Leakage protection: past draws only")

    for i in range(start, len(draws)):
        history = draws[:i]
        actual = set(draws[i]["result"])
        actual_sets.append(actual)

        for name, strategy in strategies.items():
            prediction = strategy(history)
            results[name].append(len(prediction & actual))

    random_totals = monte_carlo_random(actual_sets)

    print("\nTheoretical random expectation: 0.8000")
    print(
        "Monte Carlo random mean:",
        round(float(random_totals.mean() / BACKTEST_DRAWS), 4)
    )

    for name, hits in results.items():
        total = sum(hits)
        average = mean(hits)
        low, high = confidence_interval(hits)

        p_value = (
            1 + int(np.sum(random_totals >= total))
        ) / (RANDOM_SIMULATIONS + 1)

        adjusted_p = min(1.0, p_value * len(strategies))

        print(f"\n===== {name} =====")
        print("Total hits:", total)
        print("Average hits:", round(average, 4))
        print("95% CI:", round(low, 4), round(high, 4))
        print("Difference vs random:", round(average - 0.8, 4))
        print("Random upper-tail p:", round(float(p_value), 4))
        print("Adjusted p (3 strategies):", round(float(adjusted_p), 4))

        for block in range(3):
            segment = hits[block * 100:(block + 1) * 100]
            print(
                f"Period {block+1}: "
                f"mean={mean(segment):.3f}, "
                f"zero={segment.count(0)}, "
                f"2+={sum(x >= 2 for x in segment)}"
            )

    return results


# --------------------------------------------------
# 6. LATEST EXPERIMENTAL RANKINGS
# --------------------------------------------------

def latest_candidates(draws):
    history = draws[-TRAIN_MAX_DRAWS:]

    candidates = {
        "V5 STATISTICS": set(
            rank_scores(statistical_scores(history))[:6]
        ),
        "V6 PAIR": v6_strategy(history),
        "V7 STABLE PAIR": v7_strategy(history),
    }

    print("\n===== V7 LATEST EXPERIMENTAL CANDIDATES =====")

    for name, numbers in candidates.items():
        print(
            name + ":",
            " - ".join(f"{n:02d}" for n in sorted(numbers))
        )

    votes = Counter()
    for numbers in candidates.values():
        votes.update(numbers)

    print("\n===== MODEL AGREEMENT =====")
    for number, count in sorted(
        votes.items(), key=lambda item: (-item[1], item[0])
    ):
        print(f"{number:02d}: {count}/3 models")


# --------------------------------------------------
# 7. MAIN
# --------------------------------------------------

def main():
    draws = download_data()

    pair_null_test(draws)
    pair_stability_report(draws)
    backtest_v7(draws)
    latest_candidates(draws)

    print("\n===== AI MEGA 6/45 V7 COMPLETE =====")
    print("Research only.")
    print("Historical patterns do not establish predictive advantage.")


if __name__ == "__main__":
    main()
