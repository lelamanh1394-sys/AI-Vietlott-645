
import json
import math
import random
import re
import urllib.request
from collections import Counter
from itertools import combinations
from datetime import datetime, timezone

# ============================================================
# AI VIETLOTT MEGA 6/45 - V100
# Unified engine: V2 + V14 + V14.6 + V15
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "vietvudanh/vietlott-data/main/data/power645.jsonl"
)

MIN_HISTORY = 200
HOLDOUT_DRAWS = 300
SEED = 100645
NUMBERS = list(range(1, 46))

STRATEGIES = [
    "freq30", "freq60", "freq100", "freq180",
    "freq500", "freq_all", "overdue", "trend",
    "transition", "pair", "hybrid", "balanced",
    "v100_ensemble"
]


# ============================================================
# 1. DATA DOWNLOAD AND PARSING
# ============================================================

def valid_numbers(values):
    try:
        values = [int(x) for x in values]
    except (ValueError, TypeError):
        return None

    if len(values) != 6:
        return None

    if len(set(values)) != 6:
        return None

    if any(n < 1 or n > 45 for n in values):
        return None

    return sorted(values)


def find_numbers(obj):
    """Read common Mega 6/45 JSON structures."""

    preferred = [
        "numbers", "winning_numbers", "winningNumbers",
        "result", "balls", "drawn_numbers",
        "combination", "red"
    ]

    if isinstance(obj, dict):

        # Try explicitly named number fields first.
        for key in preferred:
            if key in obj:
                found = find_numbers(obj[key])
                if found:
                    return found

        # Try n1...n6, ball1...ball6, number1...number6.
        for prefix in ["n", "ball", "number"]:
            keys = [prefix + str(i) for i in range(1, 7)]

            if all(k in obj for k in keys):
                found = valid_numbers([obj[k] for k in keys])
                if found:
                    return found

        # Search nested structures.
        for value in obj.values():
            found = find_numbers(value)
            if found:
                return found

    elif isinstance(obj, list):

        # Accept an exact list of six numbers.
        found = valid_numbers(obj)
        if found:
            return found

        for value in obj:
            found = find_numbers(value)
            if found:
                return found

    return None


def get_metadata(obj):
    result = {
        "id": "",
        "date": ""
    }

    if not isinstance(obj, dict):
        return result

    id_keys = [
        "draw_id", "drawId", "issue",
        "period", "draw_number", "drawNumber", "id"
    ]

    date_keys = [
        "date", "draw_date", "drawDate",
        "drawing_date", "ngay_quay"
    ]

    for key in id_keys:
        if key in obj:
            result["id"] = str(obj[key])
            break

    for key in date_keys:
        if key in obj:
            result["date"] = str(obj[key])
            break

    return result


def download_draws():
    print("\nDOWNLOADING MEGA 6/45 DATA")

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read().decode("utf-8")

    draws = []
    seen_ids = set()

    for line in raw.splitlines():
        try:
            obj = json.loads(line)
            numbers = find_numbers(obj)

            if not numbers:
                continue

            metadata = get_metadata(obj)
            draw_id = metadata["id"]

            # Only remove duplicate draw IDs.
            # Never remove a repeated six-number combination.
            if draw_id:
                if draw_id in seen_ids:
                    continue
                seen_ids.add(draw_id)

            draws.append({
                "id": draw_id,
                "date": metadata["date"],
                "numbers": numbers
            })

        except (ValueError, TypeError, json.JSONDecodeError):
            continue

    # Sort by date only when every record has an ISO-like date.
    if draws and all(
        re.match(r"^\d{4}-\d{2}-\d{2}", d["date"])
        for d in draws
    ):
        draws.sort(key=lambda d: (d["date"], d["id"]))

    if len(draws) < MIN_HISTORY + 50:
        raise RuntimeError(
            "Insufficient valid data: {}".format(len(draws))
        )

    return draws


# ============================================================
# 2. STATISTICAL FEATURES
# ============================================================

def zscore(values):
    avg = sum(values) / len(values)

    variance = sum(
        (x - avg) ** 2 for x in values
    ) / len(values)

    sd = math.sqrt(variance)

    if sd == 0:
        return [0.0] * len(values)

    return [(x - avg) / sd for x in values]


def score_map(values):
    values = zscore(values)
    return {
        n: values[n - 1]
        for n in NUMBERS
    }


def frequency(history, window=None):
    data = history if window is None else history[-window:]

    counts = Counter(
        n for draw in data for n in draw
    )

    return [
        counts[n] / max(1, len(data))
        for n in NUMBERS
    ]


def overdue(history):
    last_seen = [None] * 46

    for index, draw in enumerate(history):
        for n in draw:
            last_seen[n] = index

    return [
        len(history) if last_seen[n] is None
        else len(history) - 1 - last_seen[n]
        for n in NUMBERS
    ]


def trend(history):
    recent = history[-40:]
    previous = history[-160:-40]

    recent_count = Counter(
        n for draw in recent for n in draw
    )

    previous_count = Counter(
        n for draw in previous for n in draw
    )

    return [
        recent_count[n] / max(1, len(recent))
        - previous_count[n] / max(1, len(previous))
        for n in NUMBERS
    ]


def transition(history):
    """Measure historical next-draw co-occurrence with latest draw."""

    latest = set(history[-1])
    counts = Counter()
    exposure = 0

    for previous, current in zip(history[:-1], history[1:]):
        overlap = len(latest.intersection(previous))

        if overlap:
            exposure += overlap

            for n in current:
                counts[n] += overlap

    return [
        counts[n] / max(1, exposure)
        for n in NUMBERS
    ]


def pair_association(history):
    """Smoothed pair association with numbers in latest draw."""

    selected = history[-500:]
    latest = set(history[-1])

    pair_counts = Counter()
    number_counts = Counter()

    for draw in selected:
        number_counts.update(draw)

        for a, b in combinations(draw, 2):
            pair_counts[(a, b)] += 1

    total = max(1, len(selected))
    result = []

    for n in NUMBERS:
        values = []

        for anchor in latest:
            if n == anchor:
                continue

            pair = tuple(sorted((n, anchor)))

            observed = pair_counts[pair]

            expected = (
                number_counts[n] * number_counts[anchor]
                / total
            )

            values.append(
                math.log(
                    (observed + 1.0) /
                    (expected + 1.0)
                )
            )

        result.append(
            sum(values) / max(1, len(values))
        )

    return result


# ============================================================
# 3. UNIFIED STRATEGIES
# ============================================================

def build_features(history):

    raw = {
        "freq30": frequency(history, 30),
        "freq60": frequency(history, 60),
        "freq100": frequency(history, 100),
        "freq180": frequency(history, 180),
        "freq500": frequency(history, 500),
        "freq_all": frequency(history),
        "overdue": overdue(history),
        "trend": trend(history),
        "transition": transition(history),
        "pair": pair_association(history)
    }

    f = {
        name: score_map(values)
        for name, values in raw.items()
    }

    # Hybrid strategy.
    f["hybrid"] = {
        n: (
            0.30 * f["freq180"][n]
            + 0.20 * f["freq60"][n]
            + 0.15 * f["freq500"][n]
            + 0.15 * f["overdue"][n]
            + 0.10 * f["trend"][n]
            + 0.10 * f["transition"][n]
        )
        for n in NUMBERS
    }

    # Balanced strategy.
    f["balanced"] = {
        n: sum(
            f[k][n] for k in [
                "freq60", "freq180",
                "overdue", "trend",
                "pair", "transition"
            ]
        ) / 6
        for n in NUMBERS
    }

    # V100 unified ensemble.
    f["v100_ensemble"] = {
        n: (
            0.18 * f["freq30"][n]
            + 0.12 * f["freq60"][n]
            + 0.12 * f["freq100"][n]
            + 0.16 * f["freq180"][n]
            + 0.16 * f["freq500"][n]
            + 0.08 * f["overdue"][n]
            + 0.06 * f["trend"][n]
            + 0.06 * f["transition"][n]
            + 0.06 * f["pair"][n]
        )
        for n in NUMBERS
    }

    return f


# ============================================================
# 4. TICKET GENERATION
# ============================================================

def generate_ticket(scores):
    ranked = sorted(
        NUMBERS,
        key=lambda n: (-scores[n], n)
    )

    return sorted(ranked[:6])


def generate_all_tickets(history):
    features = build_features(history)

    return {
        name: generate_ticket(features[name])
        for name in STRATEGIES
    }


def hit_count(ticket, actual):
    return len(set(ticket) & set(actual))


# ============================================================
# 5. WALK-FORWARD TRAINING
# ============================================================

def train_strategies(draws, test_start):

    performance = {
        name: [] for name in STRATEGIES
    }

    print("\n===== V100 TRAINING =====")

    for index in range(MIN_HISTORY, test_start):

        history = [
            d["numbers"] for d in draws[:index]
        ]

        actual = draws[index]["numbers"]

        tickets = generate_all_tickets(history)

        for name in STRATEGIES:
            performance[name].append(
                hit_count(tickets[name], actual)
            )

        completed = index - MIN_HISTORY + 1

        if completed % 100 == 0:
            print("Training draws:", completed)

    ranking = []

    # Shrink estimates towards theoretical random mean 0.8.
    prior_mean = 0.8
    prior_strength = 80

    for name in STRATEGIES:

        values = performance[name]

        average = sum(values) / max(1, len(values))

        recent = values[-100:]

        recent_average = (
            sum(recent) / max(1, len(recent))
        )

        posterior = (
            sum(values) + prior_mean * prior_strength
        ) / (
            len(values) + prior_strength
        )

        utility = (
            0.75 * posterior
            + 0.25 * recent_average
        )

        ranking.append({
            "name": name,
            "average": average,
            "recent": recent_average,
            "posterior": posterior,
            "utility": utility,
            "samples": len(values)
        })

    ranking.sort(
        key=lambda x: x["utility"],
        reverse=True
    )

    print("\n===== TRAINING RANKING =====")

    for item in ranking:
        print(
            "{:<15} average={:.4f} recent={:.4f} "
            "score={:.4f}".format(
                item["name"],
                item["average"],
                item["recent"],
                item["utility"]
            )
        )

    return ranking


# ============================================================
# 6. FROZEN HOLDOUT TEST
# ============================================================

def confidence_interval(values):

    if len(values) < 2:
        return 0, 0, 0

    average = sum(values) / len(values)

    variance = sum(
        (x - average) ** 2 for x in values
    ) / (len(values) - 1)

    standard_error = math.sqrt(
        variance / len(values)
    )

    margin = 1.96 * standard_error

    return (
        average,
        average - margin,
        average + margin
    )


def run_holdout(draws, test_start, selected_strategies):

    print("\n===== V100 FROZEN HOLDOUT =====")

    results = {
        name: [] for name in STRATEGIES
    }

    portfolio_results = []
    random_portfolio = []
    paired_difference = []

    for index in range(test_start, len(draws)):

        history = [
            d["numbers"] for d in draws[:index]
        ]

        actual = draws[index]["numbers"]

        tickets = generate_all_tickets(history)

        for name in STRATEGIES:
            results[name].append(
                hit_count(tickets[name], actual)
            )

        # Two selected V100 strategies.
        v100_hits = sum(
            hit_count(tickets[name], actual)
            for name in selected_strategies
        )

        portfolio_results.append(v100_hits)

        # Two distinct random tickets.
        rng = random.Random(SEED + index)

        first = rng.sample(NUMBERS, 6)

        remaining = [
            n for n in NUMBERS if n not in first
        ]

        second = rng.sample(remaining, 6)

        random_hits = (
            hit_count(first, actual)
            + hit_count(second, actual)
        )

        random_portfolio.append(random_hits)

        paired_difference.append(
            v100_hits - random_hits
        )

    print("\n===== STRATEGY RESULTS =====")

    for name in STRATEGIES:

        values = results[name]

        average = sum(values) / len(values)

        three_plus = sum(x >= 3 for x in values)

        four_plus = sum(x >= 4 for x in values)

        print(
            "{:<15} average={:.4f} | 3+={} | 4+={}".format(
                name,
                average,
                three_plus,
                four_plus
            )
        )

    v100_average = (
        sum(portfolio_results)
        / len(portfolio_results)
    )

    random_average = (
        sum(random_portfolio)
        / len(random_portfolio)
    )

    diff, low, high = confidence_interval(
        paired_difference
    )

    print("\n===== TWO-TICKET COMPARISON =====")

    print("V100 average total hits:", round(v100_average, 4))

    print("Random average total hits:", round(random_average, 4))

    print("Theoretical random average: 1.6000")

    print("Paired difference:", round(diff, 4))

    print(
        "95% confidence interval:",
        round(low, 4),
        "to",
        round(high, 4)
    )

    if low > 0:
        print("RESULT: Positive difference in this sample.")
    elif high < 0:
        print("RESULT: Below random in this sample.")
    else:
        print("RESULT: No clear advantage over random.")

    print(
        "This is a historical test, not a guarantee."
    )


# ============================================================
# 7. FINAL TWO PREDICTIONS
# ============================================================

def final_prediction(draws, ranking):

    history = [
        d["numbers"] for d in draws
    ]

    features = build_features(history)

    selected = [
        ranking[0]["name"],
        ranking[1]["name"]
    ]

    first = generate_ticket(
        features[selected[0]]
    )

    second = generate_ticket(
        features[selected[1]]
    )

    # If both strategies select the same combination,
    # find the next-ranked strategy with a different ticket.
    if first == second:

        for item in ranking[2:]:

            candidate = generate_ticket(
                features[item["name"]]
            )

            if candidate != first:
                selected[1] = item["name"]
                second = candidate
                break

    print("\n================================")
    print("          V100 FINAL")
    print("================================")

    print(
        "Strategy ranking:",
        " > ".join(x["name"] for x in ranking)
    )

    print(
        "\nDAY SO 1:",
        " - ".join("{:02d}".format(n) for n in first)
    )

    print("Strategy:", selected[0])

    print(
        "\nDAY SO 2:",
        " - ".join("{:02d}".format(n) for n in second)
    )

    print("Strategy:", selected[1])

    print(
        "Overlapping numbers:",
        len(set(first) & set(second))
    )

    print("\nLatest draw:", draws[-1]["id"])
    print("Latest date:", draws[-1]["date"])
    print("Latest result:", draws[-1]["numbers"])

    print("Total valid draws:", len(draws))

    print("\nOne-ticket jackpot odds: 1 / 8,145,060")
    print("Two-ticket jackpot odds: 2 / 8,145,060")

    record = {
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "latest_draw_id": draws[-1]["id"],
        "latest_draw_date": draws[-1]["date"],
        "latest_result": draws[-1]["numbers"],
        "ticket1": first,
        "ticket1_strategy": selected[0],
        "ticket2": second,
        "ticket2_strategy": selected[1],
        "ranking": ranking
    }

    with open(
        "v100_prediction.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            record,
            file,
            ensure_ascii=False,
            indent=2
        )

    print("\nPrediction saved: v100_prediction.json")


# ============================================================
# 8. MAIN
# ============================================================

def main():

    print("================================")
    print("      AI VIETLOTT MEGA 6/45")
    print("               V100")
    print("================================")

    draws = download_draws()

    print("\nTotal valid draws:", len(draws))
    print("Latest draw ID:", draws[-1]["id"])
    print("Latest date:", draws[-1]["date"])
    print("Latest numbers:", draws[-1]["numbers"])

    holdout = min(
        HOLDOUT_DRAWS,
        len(draws) - MIN_HISTORY - 1
    )

    test_start = len(draws) - holdout

    if test_start <= MIN_HISTORY:
        raise RuntimeError("Not enough training data.")

    # Training uses only data before the frozen holdout.
    ranking = train_strategies(
        draws,
        test_start
    )

    selected_strategies = [
        ranking[0]["name"],
        ranking[1]["name"]
    ]

    # Test without changing the selected strategies.
    run_holdout(
        draws,
        test_start,
        selected_strategies
    )

    # Final two combinations use all currently available draws.
    final_prediction(
        draws,
        ranking
    )

    print("\n========== V100 COMPLETE ==========")


if __name__ == "__main__":
    main()
