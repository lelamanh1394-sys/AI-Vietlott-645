
import json
import math
import random
import urllib.request
from collections import Counter
from itertools import combinations

# ============================================================
# AUTONOMOUS META-LEARNING ENGINE
# MEGA 6/45 - VERSION 14.3
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "vietvudanh/vietlott-data/main/data/power645.jsonl"
)

TEST_DRAWS = 300
MIN_HISTORY = 200
PRIOR_STRENGTH = 80
MIN_LEARNING_SAMPLES = 30
RECENT_WINDOW = 100
MONTE_CARLO_TICKETS = 100
RANDOM_SEED = 645143

STRATEGIES = [
    "freq60",
    "freq180",
    "freq500",
    "freq_all",
    "overdue",
    "hybrid",
    "pair",
    "balanced",
    "trend",
    "random",
]


# ============================================================
# 1. DATA
# ============================================================

def download_data():
    print("DOWNLOADING MEGA 6/45 DATA")

    req = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "AI-Vietlott-645/14.3"}
    )

    with urllib.request.urlopen(req, timeout=30) as response:
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
                or any(n < 1 or n > 45 for n in numbers)
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
                "numbers": sorted(numbers)
            })

        except (ValueError, TypeError, KeyError):
            continue

    draws.sort(key=lambda x: (x["date"], x["id"]))

    if len(draws) < MIN_HISTORY + 50:
        raise ValueError(
            f"Insufficient valid data: {len(draws)}"
        )

    return draws


# ============================================================
# 2. STATISTICS
# ============================================================

def z_scores(values):
    mean = sum(values) / len(values)

    variance = sum(
        (x - mean) ** 2 for x in values
    ) / len(values)

    sd = math.sqrt(variance)

    if sd == 0:
        return [0.0] * len(values)

    return [
        (x - mean) / sd for x in values
    ]


def frequency_scores(history, window):
    sample = history[-window:] if window else history
    counts = Counter()

    for draw in sample:
        counts.update(draw)

    return z_scores([
        counts[n] for n in range(1, 46)
    ])


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
        for pair in combinations(draw, 2):
            counts[tuple(sorted(pair))] += 1

    return counts


def pair_signal(pair_counts, history_length, a, b):
    expected = (6 / 45) * (5 / 44)
    prior = 20

    observed = pair_counts[
        tuple(sorted((a, b)))
    ]

    probability = (
        observed + prior * expected
    ) / (
        history_length + prior
    )

    return max(
        -1.0,
        min(3.0, probability / expected - 1.0)
    )


# ============================================================
# 3. STRATEGY SCORE MAPS
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

    balanced = [
        0.40 * f180[i]
        + 0.25 * f500[i]
        + 0.20 * f60[i]
        + 0.15 * overdue[i]
        for i in range(45)
    ]

    trend = [
        f60[i] - f500[i]
        for i in range(45)
    ]

    return {
        "freq60": f60,
        "freq180": f180,
        "freq500": f500,
        "freq_all": fall,
        "overdue": overdue,
        "hybrid": hybrid,
        "balanced": balanced,
        "trend": trend,
    }


# ============================================================
# 4. COMBINATION GENERATOR
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
        best_number = None
        best_score = float("-inf")

        for number in range(1, 46):
            if number in selected:
                continue

            value = scores[number - 1]

            if selected:
                synergy = sum(
                    pair_signal(
                        pair_counts,
                        history_length,
                        number,
                        previous
                    )
                    for previous in selected
                )

                value += pair_weight * synergy

            value += rng.random() * 0.000001

            if value > best_score:
                best_score = value
                best_number = number

        selected.append(best_number)

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
# 5. META-LEARNING
# ============================================================

def posterior_average(results):
    count = len(results)
    total = sum(results)

    return (
        total + PRIOR_STRENGTH * 0.8
    ) / (
        count + PRIOR_STRENGTH
    )


def strategy_utility(results):
    if len(results) < MIN_LEARNING_SAMPLES:
        return float("-inf")

    all_score = posterior_average(results)

    recent = results[-RECENT_WINDOW:]
    recent_score = posterior_average(recent)

    mean = sum(recent) / len(recent)

    variance = sum(
        (x - mean) ** 2 for x in recent
    ) / len(recent)

    standard_error = math.sqrt(
        variance / len(recent)
    )

    # Recent performance + long-term stability
    # Penalize uncertainty instead of rewarding lucky spikes
    utility = (
        0.60 * recent_score
        + 0.40 * all_score
        - 0.25 * standard_error
    )

    return utility


def choose_strategy(performance):
    ranked = []

    for name in STRATEGIES:
        utility = strategy_utility(performance[name])

        if utility != float("-inf"):
            ranked.append((utility, name))

    if not ranked:
        return "hybrid"

    ranked.sort(reverse=True)

    return ranked[0][1]


# ============================================================
# 6. EVALUATION
# ============================================================

def count_hits(ticket, actual):
    return len(set(ticket) & set(actual))


def update_performance(
    candidates,
    actual,
    performance
):
    for name in STRATEGIES:
        hits = count_hits(
            candidates[name],
            actual
        )

        performance[name].append(hits)


def make_random_ticket(seed):
    rng = random.Random(seed)

    return sorted(
        rng.sample(range(1, 46), 6)
    )


def monte_carlo_average(actual, seed):
    total = 0

    for i in range(MONTE_CARLO_TICKETS):
        ticket = make_random_ticket(
            seed + i * 7919
        )

        total += count_hits(ticket, actual)

    return total / MONTE_CARLO_TICKETS


# ============================================================
# 7. WALK-FORWARD TRAINING AND TESTING
# ============================================================

def run_backtest(draws):
    start = max(
        MIN_HISTORY,
        len(draws) - TEST_DRAWS
    )

    performance = {
        name: [] for name in STRATEGIES
    }

    # Warm-up: learn only from draws before test period
    print("\n===== META-LEARNING WARM-UP =====")

    for index in range(MIN_HISTORY, start):
        history = [
            item["numbers"]
            for item in draws[:index]
        ]

        actual = draws[index]["numbers"]

        candidates, _ = create_candidates(
            history,
            index
        )

        update_performance(
            candidates,
            actual,
            performance
        )

    print("Warm-up draws:", start - MIN_HISTORY)

    # Online walk-forward test
    model_hits = []
    paired_random_hits = []
    monte_carlo_hits = []
    chosen_strategies = []

    print("\n===== V14.3 WALK-FORWARD TEST =====")

    for index in range(start, len(draws)):
        history = [
            item["numbers"]
            for item in draws[:index]
        ]

        actual = draws[index]["numbers"]

        # AI chooses BEFORE seeing current result
        chosen = choose_strategy(performance)

        candidates, _ = create_candidates(
            history,
            index
        )

        model_ticket = candidates[chosen]

        paired_random = make_random_ticket(
            RANDOM_SEED + index * 3571
        )

        model_result = count_hits(
            model_ticket,
            actual
        )

        random_result = count_hits(
            paired_random,
            actual
        )

        mc_result = monte_carlo_average(
            actual,
            RANDOM_SEED + index * 12347
        )

        model_hits.append(model_result)
        paired_random_hits.append(random_result)
        monte_carlo_hits.append(mc_result)
        chosen_strategies.append(chosen)

        # Current outcome is revealed only AFTER decision
        update_performance(
            candidates,
            actual,
            performance
        )

    return (
        performance,
        model_hits,
        paired_random_hits,
        monte_carlo_hits,
        chosen_strategies,
        start
    )


# ============================================================
# 8. FINAL AUTONOMOUS DECISION
# ============================================================

def final_decision(draws, performance):
    history = [
        item["numbers"] for item in draws
    ]

    index = len(draws)

    candidates, maps = create_candidates(
        history,
        index
    )

    selected_strategy = choose_strategy(
        performance
    )

    selected_ticket = candidates[
        selected_strategy
    ]

    top_numbers = sorted(
        range(1, 46),
        key=lambda n: maps["hybrid"][n - 1],
        reverse=True
    )

    return (
        selected_strategy,
        selected_ticket,
        top_numbers,
        candidates
    )


# ============================================================
# 9. MAIN REPORT
# ============================================================

def main():
    draws = download_data()

    print("\n===== V14.3 DATA =====")
    print("Total draws:", len(draws))
    print("Latest draw:", draws[-1]["id"])
    print("Latest date:", draws[-1]["date"])

    (
        performance,
        model_hits,
        random_hits,
        mc_hits,
        chosen,
        start
    ) = run_backtest(draws)

    total = len(model_hits)

    model_average = sum(model_hits) / total
    random_average = sum(random_hits) / total
    mc_average = sum(mc_hits) / total

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
    print(f"Paired random average: {random_average:.5f}")
    print(f"Monte Carlo random average: {mc_average:.5f}")
    print("Theoretical baseline: 0.80000")

    print("\n===== PAIRED COMPARISON =====")
    print("Model wins:", wins)
    print("Model ties:", ties)
    print("Model losses:", losses)
    print(
        f"Model vs paired random: "
        f"{model_average - random_average:+.5f}"
    )

    print("\n===== STRATEGY PERFORMANCE =====")

    ranking = sorted(
        STRATEGIES,
        key=lambda name: strategy_utility(
            performance[name]
        ),
        reverse=True
    )

    for name in ranking:
        values = performance[name]

        average = sum(values) / len(values)

        utility = strategy_utility(values)

        print(
            f"{name:10s} | "
            f"Average: {average:.5f} | "
            f"Utility: {utility:.5f} | "
            f"Samples: {len(values)}"
        )

    print("\n===== STRATEGY SELECTION HISTORY =====")

    strategy_counts = Counter(chosen)

    for name, count in strategy_counts.most_common():
        print(
            f"{name:10s}: {count} times "
            f"({count / total * 100:.1f}%)"
        )

    (
        selected_strategy,
        ticket,
        top_numbers,
        candidates
    ) = final_decision(draws, performance)

    print("\n===== AUTONOMOUS FINAL DECISION =====")
    print("Selected strategy:", selected_strategy)

    print(
        "Final numbers:",
        "-".join(f"{n:02d}" for n in ticket)
    )

    print("\n===== ALL STRATEGY CANDIDATES =====")

    for name in ranking:
        print(
            f"{name:10s}: "
            + "-".join(
                f"{n:02d}" for n in candidates[name]
            )
        )

    print("\n===== TOP 15 HYBRID NUMBERS =====")

    print(
        ", ".join(
            f"{n:02d}" for n in top_numbers[:15]
        )
    )

    print("\n===== DECISION STATUS =====")

    if model_average > mc_average:
        print("Above Monte Carlo random average.")
    else:
        print("Not above Monte Carlo random average.")

    print(
        "Historical performance is not proof "
        "of future prediction advantage."
    )

    print("\nENGINE A V14.3 COMPLETE")


if __name__ == "__main__":
    main()
