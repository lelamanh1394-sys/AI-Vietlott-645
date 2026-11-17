
import json
import math
import random
import urllib.request
from collections import Counter, deque
from itertools import combinations

# ==========================================
# ENGINE A V14.5
# AUTONOMOUS JACKPOT RESEARCH ENGINE
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "vietvudanh/vietlott-data/main/data/power645.jsonl"
)

TEST_DRAWS = 300
MIN_HISTORY = 200
RANDOM_TICKETS = 1000

PRIOR_AVERAGE = 0.8
PRIOR_STRENGTH = 80

RECENT_WINDOW = 100
RANDOM_SEED = 645145

JACKPOT_ODDS = 8145060

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


# ==========================================
# 1. DOWNLOAD AND VALIDATE DATA
# ==========================================

def download_data():
    print("DOWNLOADING MEGA 6/45 DATA")

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8")

    draws = []
    seen = set()

    for line in raw.splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            item = json.loads(line)

            draw_id = str(item.get("id", ""))
            date = str(item.get("date", ""))
            result = item.get("result", [])

            if isinstance(result, str):
                result = [
                    int(x)
                    for x in result.replace(",", "-").split("-")
                    if x.strip()
                ]

            result = [int(x) for x in result]

            if len(result) != 6:
                continue

            if len(set(result)) != 6:
                continue

            if any(n < 1 or n > 45 for n in result):
                continue

            if draw_id in seen:
                continue

            seen.add(draw_id)

            draws.append({
                "id": draw_id,
                "date": date,
                "numbers": sorted(result)
            })

        except (ValueError, TypeError, json.JSONDecodeError):
            continue

    draws.sort(key=lambda x: (x["date"], x["id"]))

    if len(draws) <= MIN_HISTORY + 20:
        raise RuntimeError(
            "Not enough valid draw data: {}".format(len(draws))
        )

    return draws


# ==========================================
# 2. STATISTICAL FUNCTIONS
# ==========================================

def z_scores(values):
    mean = sum(values) / len(values)

    variance = sum(
        (x - mean) ** 2 for x in values
    ) / len(values)

    std = math.sqrt(variance)

    if std == 0:
        return [0.0] * len(values)

    return [(x - mean) / std for x in values]


def frequency_scores(history, window=None):
    selected = history if window is None else history[-window:]

    counts = Counter()

    for draw in selected:
        counts.update(draw["numbers"])

    values = [counts[n] for n in range(1, 46)]

    return z_scores(values)


def overdue_scores(history):
    last_seen = [None] * 46

    for index, draw in enumerate(history):
        for number in draw["numbers"]:
            last_seen[number] = index

    gaps = []

    for number in range(1, 46):
        if last_seen[number] is None:
            gaps.append(len(history))
        else:
            gaps.append(
                len(history) - 1 - last_seen[number]
            )

    return z_scores(gaps)


def trend_scores(history):
    if len(history) < 80:
        return frequency_scores(history)

    recent = history[-40:]
    previous = history[-160:-40]

    recent_counts = Counter()
    previous_counts = Counter()

    for draw in recent:
        recent_counts.update(draw["numbers"])

    for draw in previous:
        previous_counts.update(draw["numbers"])

    recent_values = [
        recent_counts[n] / 40
        for n in range(1, 46)
    ]

    previous_values = [
        previous_counts[n] / max(1, len(previous))
        for n in range(1, 46)
    ]

    differences = [
        recent_values[i] - previous_values[i]
        for i in range(45)
    ]

    return z_scores(differences)


def pair_scores(history):
    selected = history[-300:]

    pair_counts = Counter()
    number_counts = Counter()

    for draw in selected:
        numbers = draw["numbers"]
        number_counts.update(numbers)

        for a, b in combinations(numbers, 2):
            pair_counts[(a, b)] += 1

    scores = [0.0] * 45

    for number in range(1, 46):
        total = 0

        for other in range(1, 46):
            if other == number:
                continue

            pair = tuple(sorted((number, other)))
            total += pair_counts[pair]

        scores[number - 1] = (
            total / max(1, number_counts[number])
        )

    return z_scores(scores)


# ==========================================
# 3. BUILD STRATEGY MAPS
# ==========================================

def build_score_maps(history):
    freq60 = frequency_scores(history, 60)
    freq180 = frequency_scores(history, 180)
    freq500 = frequency_scores(history, 500)
    freq_all = frequency_scores(history)

    overdue = overdue_scores(history)
    trend = trend_scores(history)
    pair = pair_scores(history)

    hybrid = [
        0.35 * freq60[i]
        + 0.30 * freq180[i]
        + 0.20 * overdue[i]
        + 0.15 * trend[i]
        for i in range(45)
    ]

    balanced = [
        0.45 * freq180[i]
        + 0.30 * overdue[i]
        + 0.25 * trend[i]
        for i in range(45)
    ]

    pair_map = [
        0.70 * hybrid[i] + 0.30 * pair[i]
        for i in range(45)
    ]

    return {
        "freq60": freq60,
        "freq180": freq180,
        "freq500": freq500,
        "freq_all": freq_all,
        "overdue": overdue,
        "hybrid": hybrid,
        "pair": pair_map,
        "balanced": balanced,
        "trend": trend,
    }


# ==========================================
# 4. SELECT SIX NUMBERS
# ==========================================

def greedy_selection(scores):
    ranked = sorted(
        range(1, 46),
        key=lambda n: (scores[n - 1], -n),
        reverse=True
    )

    selected = []

    buckets = [0, 0, 0]
    odd = 0
    even = 0

    for number in ranked:
        if len(selected) == 6:
            break

        bucket = (
            0 if number <= 15
            else 1 if number <= 30
            else 2
        )

        is_odd = number % 2 == 1

        if buckets[bucket] >= 3:
            continue

        if is_odd and odd >= 4:
            continue

        if not is_odd and even >= 4:
            continue

        selected.append(number)
        buckets[bucket] += 1

        if is_odd:
            odd += 1
        else:
            even += 1

    if len(selected) < 6:
        for number in ranked:
            if number not in selected:
                selected.append(number)

            if len(selected) == 6:
                break

    return sorted(selected)


def create_candidates(history, draw_index):
    maps = build_score_maps(history)

    candidates = {}

    for name in STRATEGIES:
        if name == "random":
            rng = random.Random(
                RANDOM_SEED + draw_index * 1009
            )

            candidates[name] = sorted(
                rng.sample(range(1, 46), 6)
            )

        else:
            candidates[name] = greedy_selection(
                maps[name]
            )

    return candidates, maps


# ==========================================
# 5. HIT AND JACKPOT EVALUATION
# ==========================================

def count_hits(ticket, actual):
    return len(set(ticket) & set(actual))


def is_jackpot(ticket, actual):
    return count_hits(ticket, actual) == 6


def new_performance():
    return {
        name: {
            "hits": 0,
            "samples": 0,
            "recent": deque(maxlen=RECENT_WINDOW)
        }
        for name in STRATEGIES
    }


def update_performance(performance, candidates, actual):
    for name in STRATEGIES:
        hits = count_hits(candidates[name], actual)

        stats = performance[name]

        stats["hits"] += hits
        stats["samples"] += 1
        stats["recent"].append(hits)


# ==========================================
# 6. AUTONOMOUS META-LEARNING
# ==========================================

def posterior_average(total_hits, samples):
    return (
        total_hits + PRIOR_AVERAGE * PRIOR_STRENGTH
    ) / (
        samples + PRIOR_STRENGTH
    )


def strategy_utility(stats):
    samples = stats["samples"]

    long_score = posterior_average(
        stats["hits"],
        samples
    )

    recent = list(stats["recent"])

    recent_score = posterior_average(
        sum(recent),
        len(recent)
    )

    # Balance long-term and recent performance.
    combined = (
        0.50 * long_score
        + 0.50 * recent_score
    )

    uncertainty = 0.10 / math.sqrt(
        samples + PRIOR_STRENGTH
    )

    return combined - uncertainty


def choose_strategy(performance):
    utilities = {
        name: strategy_utility(performance[name])
        for name in STRATEGIES
    }

    return max(
        STRATEGIES,
        key=lambda name: utilities[name]
    )


# ==========================================
# 7. RANDOM BENCHMARK
# ==========================================

def monte_carlo_random(actual, seed):
    rng = random.Random(seed)

    total_hits = 0
    jackpot_count = 0
    distribution = Counter()

    for _ in range(RANDOM_TICKETS):
        ticket = rng.sample(range(1, 46), 6)

        hits = count_hits(ticket, actual)

        total_hits += hits
        distribution[hits] += 1

        if hits == 6:
            jackpot_count += 1

    return (
        total_hits / RANDOM_TICKETS,
        jackpot_count,
        distribution
    )


# ==========================================
# 8. CONFIDENCE INTERVAL
# ==========================================

def mean_and_ci(values):
    if not values:
        return 0.0, 0.0, 0.0

    mean = sum(values) / len(values)

    if len(values) < 2:
        return mean, mean, mean

    variance = sum(
        (x - mean) ** 2 for x in values
    ) / (len(values) - 1)

    standard_error = math.sqrt(
        variance / len(values)
    )

    margin = 1.96 * standard_error

    return (
        mean,
        mean - margin,
        mean + margin
    )


# ==========================================
# 9. WALK-FORWARD BACKTEST
# ==========================================

def run_backtest(draws):
    test_count = min(
        TEST_DRAWS,
        len(draws) - MIN_HISTORY
    )

    test_start = len(draws) - test_count

    performance = new_performance()

    print("\n===== V14.5 META-LEARNING WARM-UP =====")

    warmup_count = 0

    for index in range(MIN_HISTORY, test_start):
        history = draws[:index]
        actual = draws[index]["numbers"]

        candidates, _ = create_candidates(
            history,
            index
        )

        update_performance(
            performance,
            candidates,
            actual
        )

        warmup_count += 1

    print("Warm-up draws:", warmup_count)

    # AI test results
    model_hits = []
    model_distribution = Counter()

    # Random benchmark
    random_averages = []
    random_jackpots = 0
    random_distribution = Counter()

    # Per-draw paired difference
    paired_differences = []

    # Jackpot results
    ai_jackpots = 0
    ai_non_jackpots = 0

    # Strategy selection
    strategy_choices = Counter()

    # Additional diagnostic comparison
    ai_better_draws = 0
    ai_equal_draws = 0
    ai_worse_draws = 0

    print("\n===== V14.5 WALK-FORWARD TEST =====")
    print("Test draws:", test_count)
    print("Random tickets per draw:", RANDOM_TICKETS)

    for index in range(test_start, len(draws)):
        history = draws[:index]
        actual = draws[index]["numbers"]

        candidates, _ = create_candidates(
            history,
            index
        )

        selected_strategy = choose_strategy(
            performance
        )

        selected_ticket = candidates[selected_strategy]

        ai_hit = count_hits(
            selected_ticket,
            actual
        )

        ai_is_jackpot = is_jackpot(
            selected_ticket,
            actual
        )

        if ai_is_jackpot:
            ai_jackpots += 1
        else:
            ai_non_jackpots += 1

        model_hits.append(ai_hit)
        model_distribution[ai_hit] += 1

        strategy_choices[selected_strategy] += 1

        (
            random_average,
            random_jackpot_count,
            random_hit_distribution
        ) = monte_carlo_random(
            actual,
            RANDOM_SEED + 900000 + index
        )

        random_averages.append(random_average)
        random_jackpots += random_jackpot_count
        random_distribution.update(
            random_hit_distribution
        )

        difference = ai_hit - random_average

        paired_differences.append(difference)

        if difference > 0:
            ai_better_draws += 1
        elif difference < 0:
            ai_worse_draws += 1
        else:
            ai_equal_draws += 1

        # Update only after the real result is known.
        update_performance(
            performance,
            candidates,
            actual
        )

    # ======================================
    # RESULTS
    # ======================================

    model_average = (
        sum(model_hits) / len(model_hits)
    )

    random_average = (
        sum(random_averages) / len(random_averages)
    )

    diff_average, ci_low, ci_high = mean_and_ci(
        paired_differences
    )

    print("\n===== JACKPOT RESULT =====")

    print("AI Jackpot 6/6:", ai_jackpots)
    print("AI non-Jackpot 0-5:", ai_non_jackpots)

    print(
        "AI Jackpot rate:",
        "{:.6f}%".format(
            100 * ai_jackpots / test_count
        )
    )

    print(
        "Theoretical one-ticket Jackpot probability:",
        "1 / {:,}".format(JACKPOT_ODDS)
    )

    print("\n===== AI HIT DISTRIBUTION =====")

    for hits in range(7):
        print(
            "{} hits: {}".format(
                hits,
                model_distribution[hits]
            )
        )

    print("\n===== RANDOM HIT DISTRIBUTION =====")

    total_random_tickets = (
        test_count * RANDOM_TICKETS
    )

    for hits in range(7):
        count = random_distribution[hits]

        percentage = (
            100 * count / total_random_tickets
        )

        print(
            "{} hits: {} ({:.4f}%)".format(
                hits,
                count,
                percentage
            )
        )

    print("\n===== RANDOM JACKPOT CHECK =====")

    print("Random Jackpot tickets:", random_jackpots)

    print(
        "Expected random Jackpot tickets:",
        round(
            total_random_tickets / JACKPOT_ODDS,
            6
        )
    )

    print("\n===== STATISTICAL COMPARISON =====")

    print("AI average hits:", round(model_average, 5))
    print("Random average hits:", round(random_average, 5))
    print("Theoretical average: 0.80000")
    print("AI minus random:", round(diff_average, 5))

    print("\n===== AI VS RANDOM DRAW COMPARISON =====")

    print("AI better:", ai_better_draws)
    print("AI equal:", ai_equal_draws)
    print("AI worse:", ai_worse_draws)

    print("\n===== 95% CONFIDENCE INTERVAL =====")

    print("Difference lower:", round(ci_low, 5))
    print("Difference upper:", round(ci_high, 5))

    if ci_low > 0:
        print("Interpretation: AI above random in this test.")
    elif ci_high < 0:
        print("Interpretation: AI below random in this test.")
    else:
        print("Interpretation: No clear difference from random.")

    print("\n===== STRATEGY PERFORMANCE =====")

    ranking = []

    for name in STRATEGIES:
        stats = performance[name]

        average = (
            stats["hits"] / max(1, stats["samples"])
        )

        recent = list(stats["recent"])

        recent_average = (
            sum(recent) / len(recent)
            if recent else 0
        )

        utility = strategy_utility(stats)

        ranking.append((
            name,
            average,
            recent_average,
            utility,
            stats["samples"]
        ))

    ranking.sort(
        key=lambda x: x[3],
        reverse=True
    )

    for name, average, recent, utility, samples in ranking:
        print(
            "{:<10} | Long: {:.5f} | Recent: {:.5f} "
            "| Utility: {:.5f} | Samples: {}".format(
                name,
                average,
                recent,
                utility,
                samples
            )
        )

    print("\n===== STRATEGY SELECTION HISTORY =====")

    for name, count in strategy_choices.most_common():
        percentage = (
            100 * count / max(1, test_count)
        )

        print(
            "{:<10}: {} times ({:.1f}%)".format(
                name,
                count,
                percentage
            )
        )

    return performance


# ==========================================
# 10. FINAL AUTONOMOUS DECISION
# ==========================================

def final_decision(draws, performance):
    candidates, maps = create_candidates(
        draws,
        len(draws) + 100000
    )

    selected_strategy = choose_strategy(
        performance
    )

    final_ticket = candidates[selected_strategy]

    print("\n===== V14.5 AUTONOMOUS FINAL DECISION =====")

    print("Selected strategy:", selected_strategy)

    print(
        "Final numbers:",
        "-".join(
            "{:02d}".format(n)
            for n in final_ticket
        )
    )

    print("\n===== ALL STRATEGY CANDIDATES =====")

    for name in STRATEGIES:
        ticket = candidates[name]

        print(
            "{:<10}: {}".format(
                name,
                "-".join(
                    "{:02d}".format(n)
                    for n in ticket
                )
            )
        )

    print("\n===== TOP 15 HYBRID NUMBERS =====")

    top_numbers = sorted(
        range(1, 46),
        key=lambda n: (
            maps["hybrid"][n - 1],
            -n
        ),
        reverse=True
    )[:15]

    print(
        ", ".join(
            "{:02d}".format(n)
            for n in top_numbers
        )
    )

    print("\n===== DECISION STATUS =====")

    print("One autonomous research ticket selected.")

    print(
        "No historical result guarantees a future Jackpot."
    )


# ==========================================
# 11. MAIN
# ==========================================

def main():
    draws = download_data()

    print("\n===== V14.5 DATA =====")

    print("Total draws:", len(draws))
    print("Latest draw:", draws[-1]["id"])
    print("Latest date:", draws[-1]["date"])

    performance = run_backtest(draws)

    final_decision(
        draws,
        performance
    )

    print("\nENGINE A V14.5 COMPLETE")


if __name__ == "__main__":
    main()
