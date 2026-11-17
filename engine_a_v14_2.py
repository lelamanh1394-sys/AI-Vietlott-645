
import json
import math
import random
import urllib.request
from collections import Counter
from itertools import combinations

# ============================================================
# AUTONOMOUS AI - MEGA 6/45
# VERSION 14.2
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "vietvudanh/vietlott-data/main/data/power645.jsonl"
)

TEST_DRAWS = 300
MIN_HISTORY = 200
PRIOR_STRENGTH = 60
MIN_SELECTION_HISTORY = 30
RANDOM_SEED = 645142

STRATEGIES = [
    "freq60",
    "freq180",
    "freq500",
    "freq_all",
    "overdue",
    "hybrid",
    "pair",
    "random",
]


# ============================================================
# 1. DOWNLOAD AND VALIDATE DATA
# ============================================================

def download_data():
    print("DOWNLOADING MEGA 6/45 DATA")

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "AI-Vietlott-645/14.2"}
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read().decode("utf-8")

    draws = []
    seen = set()

    for line in content.splitlines():
        try:
            row = json.loads(line)
            numbers = row.get("result", [])

            if len(numbers) < 6:
                continue

            numbers = [int(x) for x in numbers[:6]]

            if (
                len(set(numbers)) != 6
                or any(x < 1 or x > 45 for x in numbers)
            ):
                continue

            draw_id = str(row.get("id", ""))

            if draw_id and draw_id in seen:
                continue

            if draw_id:
                seen.add(draw_id)

            draws.append({
                "id": draw_id,
                "date": str(row.get("date", "")),
                "numbers": sorted(numbers),
            })

        except (ValueError, TypeError, KeyError):
            continue

    draws.sort(key=lambda x: (x["date"], x["id"]))

    if len(draws) < MIN_HISTORY + 50:
        raise ValueError(
            f"Not enough valid draws: {len(draws)}"
        )

    return draws


# ============================================================
# 2. STATISTICAL UTILITIES
# ============================================================

def z_scores(values):
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    sd = math.sqrt(variance)

    if sd == 0:
        return [0.0] * len(values)

    return [(x - mean) / sd for x in values]


def frequency_scores(history, window):
    sample = history[-window:] if window else history
    counts = Counter()

    for draw in sample:
        counts.update(draw)

    values = [counts[n] for n in range(1, 46)]
    return z_scores(values)


def overdue_scores(history):
    gaps = []

    for number in range(1, 46):
        gap = 0

        for draw in reversed(history):
            if number in draw:
                break
            gap += 1

        gaps.append(math.log1p(gap))

    return z_scores(gaps)


def pair_statistics(history):
    counts = Counter()

    for draw in history:
        for pair in combinations(sorted(draw), 2):
            counts[pair] += 1

    return counts


def pair_strength(pair_counts, history_length, a, b):
    # Expected probability that a particular pair appears
    expected = (6 / 45) * (5 / 44)

    # Bayesian smoothing prevents rare pairs from dominating
    prior = 20
    observed = pair_counts[tuple(sorted((a, b)))]

    estimated = (
        observed + prior * expected
    ) / (
        history_length + prior
    )

    return max(-1.0, min(3.0, estimated / expected - 1.0))


# ============================================================
# 3. BUILD INDEPENDENT STRATEGIES
# ============================================================

def build_score_maps(history):
    f60 = frequency_scores(history, 60)
    f180 = frequency_scores(history, 180)
    f500 = frequency_scores(history, 500)
    fall = frequency_scores(history, 0)
    overdue = overdue_scores(history)

    hybrid = [
        0.35 * f60[i]
        + 0.30 * f180[i]
        + 0.20 * f500[i]
        + 0.15 * fall[i]
        for i in range(45)
    ]

    pair_base = [
        0.70 * hybrid[i] + 0.30 * overdue[i]
        for i in range(45)
    ]

    return {
        "freq60": f60,
        "freq180": f180,
        "freq500": f500,
        "freq_all": fall,
        "overdue": overdue,
        "hybrid": hybrid,
        "pair": pair_base,
    }


# ============================================================
# 4. AUTONOMOUS NUMBER SELECTION
# ============================================================

def greedy_selection(
    scores,
    pair_counts,
    history_length,
    pair_weight,
    seed
):
    rng = random.Random(seed)
    selected = []

    while len(selected) < 6:
        candidates = []

        for number in range(1, 46):
            if number in selected:
                continue

            value = scores[number - 1]

            if selected:
                synergy = sum(
                    pair_strength(
                        pair_counts,
                        history_length,
                        number,
                        previous
                    )
                    for previous in selected
                )

                value += pair_weight * synergy

            # Tiny seeded tie-breaker, not a prediction signal
            value += rng.random() * 0.000001

            candidates.append((value, number))

        candidates.sort(reverse=True)
        selected.append(candidates[0][1])

    return sorted(selected)


def create_candidates(history, draw_index):
    maps = build_score_maps(history)
    pairs = pair_statistics(history)

    candidates = {}

    for name in STRATEGIES:
        if name == "random":
            rng = random.Random(
                RANDOM_SEED + draw_index * 1009
            )
            candidates[name] = sorted(
                rng.sample(range(1, 46), 6)
            )
            continue

        weight = 0.25 if name == "pair" else 0.08

        candidates[name] = greedy_selection(
            maps[name],
            pairs,
            len(history),
            weight,
            RANDOM_SEED + draw_index * 101 + len(name)
        )

    return candidates, maps


# ============================================================
# 5. AUTONOMOUS STRATEGY ARBITER
# ============================================================

def adjusted_performance(results):
    if not results:
        return 0.8

    total = sum(results)
    count = len(results)

    # Shrink small samples towards the theoretical baseline
    return (
        total + PRIOR_STRENGTH * 0.8
    ) / (
        count + PRIOR_STRENGTH
    )


def choose_strategy(performance_history):
    available = [
        name for name in STRATEGIES
        if len(performance_history[name]) >= MIN_SELECTION_HISTORY
    ]

    if not available:
        return "hybrid"

    ranked = sorted(
        available,
        key=lambda name: (
            adjusted_performance(performance_history[name]),
            sum(performance_history[name]) / len(
                performance_history[name]
            ),
            name
        ),
        reverse=True
    )

    return ranked[0]


def count_hits(ticket, actual):
    return len(set(ticket) & set(actual))


# ============================================================
# 6. WALK-FORWARD AUTONOMOUS BACKTEST
# ============================================================

def run_backtest(draws):
    start = max(
        MIN_HISTORY,
        len(draws) - TEST_DRAWS
    )

    performance_history = {
        name: [] for name in STRATEGIES
    }

    model_hits = []
    random_hits = []
    selected_strategies = []

    print("\n===== V14.2 WALK-FORWARD TEST =====")
    print("Test draws:", len(draws) - start)

    for index in range(start, len(draws)):
        history = [
            item["numbers"] for item in draws[:index]
        ]

        actual = draws[index]["numbers"]

        candidates, _ = create_candidates(
            history,
            index
        )

        # Arbiter decides using only previous test results
        chosen = choose_strategy(performance_history)

        model_ticket = candidates[chosen]

        rng = random.Random(
            RANDOM_SEED + index * 7919
        )
        random_ticket = sorted(
            rng.sample(range(1, 46), 6)
        )

        current_hits = {}

        for name in STRATEGIES:
            hits = count_hits(candidates[name], actual)
            current_hits[name] = hits
            performance_history[name].append(hits)

        model_result = count_hits(model_ticket, actual)
        random_result = count_hits(random_ticket, actual)

        model_hits.append(model_result)
        random_hits.append(random_result)
        selected_strategies.append(chosen)

    return (
        performance_history,
        model_hits,
        random_hits,
        selected_strategies,
        start
    )


# ============================================================
# 7. FINAL AUTONOMOUS DECISION
# ============================================================

def final_decision(draws, performance_history):
    latest_history = [
        item["numbers"] for item in draws
    ]

    latest_index = len(draws)

    candidates, maps = create_candidates(
        latest_history,
        latest_index
    )

    chosen = choose_strategy(performance_history)
    ticket = candidates[chosen]

    ranked = sorted(
        range(1, 46),
        key=lambda n: maps["hybrid"][n - 1],
        reverse=True
    )

    return chosen, ticket, ranked, candidates


# ============================================================
# 8. REPORT
# ============================================================

def main():
    draws = download_data()

    print("\n===== V14.2 DATA =====")
    print("Total draws:", len(draws))
    print("Latest draw:", draws[-1]["id"])
    print("Latest date:", draws[-1]["date"])

    (
        performance,
        model_hits,
        random_hits,
        selected,
        start
    ) = run_backtest(draws)

    total = len(model_hits)

    model_average = sum(model_hits) / total
    random_average = sum(random_hits) / total

    wins = sum(
        m > r for m, r in zip(model_hits, random_hits)
    )
    ties = sum(
        m == r for m, r in zip(model_hits, random_hits)
    )
    losses = sum(
        m < r for m, r in zip(model_hits, random_hits)
    )

    print("\n===== AUTONOMOUS BACKTEST =====")
    print("Test draws:", total)
    print(f"Model average: {model_average:.5f}")
    print(f"Random average per ticket: {random_average:.5f}")
    print("Theoretical random baseline: 0.80000")

    print("\n===== FAIR COMPARISON =====")
    print(f"Model wins: {wins}")
    print(f"Model ties: {ties}")
    print(f"Model losses: {losses}")
    print(f"Difference: {model_average - random_average:+.5f}")

    print("\n===== STRATEGY PERFORMANCE =====")

    ranking = sorted(
        STRATEGIES,
        key=lambda name: adjusted_performance(
            performance[name]
        ),
        reverse=True
    )

    for name in ranking:
        values = performance[name]
        average = sum(values) / len(values)

        print(
            f"{name:10s} | "
            f"Average: {average:.5f} | "
            f"Samples: {len(values)}"
        )

    chosen, ticket, top_numbers, candidates = final_decision(
        draws,
        performance
    )

    print("\n===== AUTONOMOUS FINAL DECISION =====")
    print("Selected strategy:", chosen)
    print("Final numbers:", "-".join(
        f"{n:02d}" for n in ticket
    ))

    print("\n===== OTHER STRATEGY CANDIDATES =====")

    for name in ranking:
        print(
            f"{name:10s}: "
            + "-".join(
                f"{n:02d}" for n in candidates[name]
            )
        )

    print("\n===== TOP 15 HYBRID NUMBERS =====")

    print(", ".join(
        f"{n:02d}" for n in top_numbers[:15]
    ))

    print("\n===== DECISION STATUS =====")

    if model_average > random_average:
        print("Historical result: above sampled random baseline")
    else:
        print("Historical result: not above sampled random baseline")

    print(
        "Important: This does not prove future lottery advantage."
    )

    print("\nENGINE A V14.2 COMPLETE")


if __name__ == "__main__":
    main()
