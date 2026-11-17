"""AI Vietlott Mega 6/45 V6 - pattern diagnostics + walk-forward research.
Uses main_v5.py only for shared data loading and random benchmark utilities.
Research only; lottery draws are random and this does not predict winning numbers.
"""
from collections import Counter
from itertools import combinations
from statistics import mean, median, stdev
from math import sqrt
import numpy as np

from main_v5 import (
    download_data,
    monte_carlo_random,
    confidence_interval,
    BACKTEST_DRAWS,
    RANDOM_SIMULATIONS,
    SEED,
)

LOW_MAX = 22
TRAIN_MAX_DRAWS = 500
MIN_HISTORY = 200
PAIR_SHRINKAGE = 20.0


def draw_shape(numbers):
    nums = sorted(numbers)
    return {
        "odd": sum(n % 2 for n in nums),
        "low": sum(n <= LOW_MAX for n in nums),
        "total": sum(nums),
        "span": nums[-1] - nums[0],
        "consecutive_pairs": sum(nums[i + 1] - nums[i] == 1 for i in range(5)),
    }


def pattern_report(draws):
    print("\n===== V6 HISTORICAL PATTERN ANALYSIS =====")
    shapes = [draw_shape(d["result"]) for d in draws]
    n = len(draws)
    print("Draws analyzed:", n)
    print("\nODD / EVEN SPLIT (odd count -> draws)")
    c = Counter(x["odd"] for x in shapes)
    for k in range(7):
        print(f"{k}/{6-k}: {c.get(k, 0)} ({c.get(k, 0)/n:.1%})")
    print("\nLOW / HIGH SPLIT (low=01-22, high=23-45)")
    c = Counter(x["low"] for x in shapes)
    for k in range(7):
        print(f"{k}/{6-k}: {c.get(k, 0)} ({c.get(k, 0)/n:.1%})")

    for key, label in [("total", "SUM OF SIX NUMBERS"), ("span", "NUMBER SPAN"), ("consecutive_pairs", "ADJACENT CONSECUTIVE PAIRS")]:
        vals = [x[key] for x in shapes]
        print(f"\n{label}")
        print("Mean:", round(mean(vals), 3), "Median:", median(vals), "Min:", min(vals), "Max:", max(vals))
        if key == "total":
            print("10th/25th/75th/90th percentiles:", [round(float(v), 1) for v in np.percentile(vals, [10, 25, 75, 90])])
        else:
            cc = Counter(vals)
            print("Distribution:", " | ".join(f"{k}:{cc[k]}" for k in sorted(cc)))

    repeats = []
    for i in range(1, n):
        repeats.append(len(set(draws[i]["result"]) & set(draws[i-1]["result"])))
    rc = Counter(repeats)
    print("\nREPEATED NUMBERS FROM PREVIOUS DRAW")
    for k in range(7):
        print(f"{k} repeats: {rc.get(k, 0)} ({rc.get(k, 0)/max(len(repeats),1):.1%})")

    pair_counts = Counter()
    triple_counts = Counter()
    for d in draws:
        nums = sorted(d["result"])
        pair_counts.update(combinations(nums, 2))
        triple_counts.update(combinations(nums, 3))

    # Under a uniform 6-of-45 draw, each fixed pair has probability 15/C(45,2).
    pair_p = 15 / 990
    expected_pair = n * pair_p
    print("\nMOST FREQUENT PAIRS (descriptive only; compare with expected random count)")
    print("Expected count for any fixed pair under uniform random draws:", round(expected_pair, 2))
    for pair, count in pair_counts.most_common(15):
        print(f"{pair[0]:02d}-{pair[1]:02d}: {count} | lift={(count+0.5)/(expected_pair+0.5):.3f}")
    print("\nMOST FREQUENT TRIPLES (very sparse; descriptive only)")
    for tri, count in triple_counts.most_common(10):
        print(f"{tri[0]:02d}-{tri[1]:02d}-{tri[2]:02d}: {count}")


def pair_strategy(history):
    """Greedy exploratory strategy using only draws before the target draw."""
    hist = history[-TRAIN_MAX_DRAWS:]
    n_draws = len(hist)
    singles = Counter()
    pairs = Counter()
    for d in hist:
        nums = sorted(d["result"])
        singles.update(nums)
        pairs.update(combinations(nums, 2))

    # Mildly shrink pair co-occurrence toward its random expectation.
    expected = n_draws * (15 / 990)
    pair_score = {}
    for a in range(1, 46):
        for b in range(a + 1, 46):
            observed = pairs[(a, b)]
            raw = np.log((observed + 0.5) / (expected + 0.5))
            pair_score[(a, b)] = raw * (observed / (observed + PAIR_SHRINKAGE))

    # Individual frequency is centered on the theoretical 6/45 rate.
    freq_score = {x: (singles[x] / max(n_draws, 1)) - (6 / 45) for x in range(1, 46)}
    chosen = []
    remaining = set(range(1, 46))
    for _ in range(6):
        best, best_score = None, -1e9
        for candidate in remaining:
            if not chosen:
                synergy = 0.0
            else:
                synergy = mean(pair_score[tuple(sorted((candidate, old)))] for old in chosen)
            score = 0.25 * freq_score[candidate] + 0.75 * synergy
            if score > best_score:
                best, best_score = candidate, score
        chosen.append(best)
        remaining.remove(best)
    return set(chosen)


def backtest_pair_strategy(draws):
    start = len(draws) - BACKTEST_DRAWS
    hits = []
    actual_sets = []
    print("\n===== V6 PAIR STRATEGY STRICT WALK-FORWARD =====")
    print("Test draws:", BACKTEST_DRAWS)
    print("Training history cap:", TRAIN_MAX_DRAWS)
    print("Leakage rule: each prediction uses draws strictly before its target draw")
    for i in range(start, len(draws)):
        history = draws[:i]
        pred = pair_strategy(history)
        actual = set(draws[i]["result"])
        actual_sets.append(actual)
        hits.append(len(pred & actual))
    random_totals = monte_carlo_random(actual_sets)
    total = sum(hits)
    avg = mean(hits)
    lo, hi = confidence_interval(hits)
    p = (1 + int(np.sum(random_totals >= total))) / (RANDOM_SIMULATIONS + 1)
    print("Total hits:", total)
    print("Average hits:", round(avg, 4))
    print("95% CI:", round(lo, 4), round(hi, 4))
    print("Difference vs theoretical random 0.8:", round(avg - 0.8, 4))
    print("Random upper-tail p:", round(p, 4))
    print("Hits distribution:", {k: hits.count(k) for k in range(7)})
    for block in range(3):
        segment = hits[block*100:(block+1)*100]
        print(f"Period {block+1}: mean={mean(segment):.3f}, zero={segment.count(0)}, 2+={sum(x>=2 for x in segment)}")
    return hits


def next_pair_candidate(draws):
    pred = sorted(pair_strategy(draws))
    print("\n===== V6 EXPLORATORY PAIR-BASED CANDIDATE =====")
    print(" - ".join(f"{n:02d}" for n in pred))
    print("Note: this is an experimental ranking, not a higher-probability ticket.")


def main():
    draws = download_data()
    pattern_report(draws)
    backtest_pair_strategy(draws)
    next_pair_candidate(draws)
    print("\n===== AI MEGA 6/45 V6 COMPLETE =====")
    print("Research only. Historical patterns do not establish predictive advantage.")


if __name__ == "__main__":
    main()
