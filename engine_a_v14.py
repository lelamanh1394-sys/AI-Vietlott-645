
import json
import math
import random
import requests
from collections import Counter
from itertools import combinations

URL = (
    "https://raw.githubusercontent.com/"
    "vietvudanh/vietlott-data/main/data/power645.jsonl"
)

# V14.1 configuration
TEST_DRAWS = 300
MIN_HISTORY = 200
BACKTEST_CANDIDATES = 3000
FINAL_CANDIDATES = 12000
RANDOM_TICKETS_PER_DRAW = 500
SEED = 645141

P = 6 / 45
PAIR_P = (6 / 45) * (5 / 44)


def download_data():
    print("DOWNLOADING MEGA 6/45 DATA")

    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    draws = []

    for line in response.text.splitlines():
        if not line.strip():
            continue

        try:
            item = json.loads(line)
            numbers = sorted(int(n) for n in item["result"])

            if (
                len(numbers) != 6
                or len(set(numbers)) != 6
                or any(n < 1 or n > 45 for n in numbers)
            ):
                continue

            draws.append({
                "id": str(item["id"]),
                "date": str(item["date"]),
                "result": numbers
            })

        except (ValueError, KeyError, TypeError):
            continue

    draws.sort(key=lambda x: x["date"])

    if len(draws) <= MIN_HISTORY:
        raise ValueError("Not enough valid historical draws.")

    ids = [d["id"] for d in draws]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate draw IDs detected.")

    return draws


def clamp(value, low=-2.5, high=2.5):
    return max(low, min(high, value))


def frequency_signal(history, window):
    data = history[-window:]
    total = len(data)

    if total == 0:
        return {n: 0.0 for n in range(1, 46)}

    counts = Counter(
        n for draw in data for n in draw["result"]
    )

    result = {}

    for n in range(1, 46):
        observed = counts[n]
        expected = total * P
        variance = total * P * (1 - P)

        z = (observed - expected) / math.sqrt(variance)
        shrink = total / (total + 100)

        result[n] = clamp(z) * shrink

    return result


def transition_signal(history):
    result = {n: 0.0 for n in range(1, 46)}

    if len(history) < 30:
        return result

    present = Counter()
    present_next = Counter()

    for i in range(len(history) - 1):
        current = set(history[i]["result"])
        following = set(history[i + 1]["result"])

        for n in range(1, 46):
            if n in current:
                present[n] += 1
                if n in following:
                    present_next[n] += 1

    total_pairs = len(history) - 1

    for n in range(1, 46):
        count_yes = present[n]
        count_no = total_pairs - count_yes

        if count_yes < 10 or count_no < 10:
            continue

        rate_yes = present_next[n] / count_yes
        rate_no = (6 * total_pairs / 45 - present_next[n]) / count_no

        # Compare conditional rates with uncertainty adjustment.
        pooled = P
        se = math.sqrt(
            pooled * (1 - pooled)
            * (1 / count_yes + 1 / count_no)
        )

        if se > 0:
            z = (rate_yes - rate_no) / se
            shrink = total_pairs / (total_pairs + 100)
            result[n] = clamp(z) * shrink

    return result


def pair_signal(history):
    result = {}

    total = len(history)

    if total < 30:
        return result

    counts = Counter()

    for draw in history:
        for a, b in combinations(draw["result"], 2):
            counts[(a, b)] += 1

    for a in range(1, 46):
        for b in range(a + 1, 46):
            observed = counts[(a, b)]
            expected = total * PAIR_P
            variance = total * PAIR_P * (1 - PAIR_P)

            z = (observed - expected) / math.sqrt(variance)
            shrink = total / (total + 200)

            result[(a, b)] = clamp(z) * shrink

    return result


def build_scores(history):
    f60 = frequency_signal(history, 60)
    f180 = frequency_signal(history, 180)
    f500 = frequency_signal(history, 500)
    f_all = frequency_signal(history, len(history))

    transition = transition_signal(history)
    pairs = pair_signal(history)

    scores = {}

    for n in range(1, 46):
        scores[n] = (
            0.20 * f60[n]
            + 0.20 * f180[n]
            + 0.20 * f500[n]
            + 0.15 * f_all[n]
            + 0.25 * transition[n]
        )

    return scores, pairs


def score_ticket(ticket, scores, pairs):
    individual = sum(scores[n] for n in ticket)

    pair_total = 0.0

    for a, b in combinations(ticket, 2):
        pair_total += pairs.get((a, b), 0.0)

    # Pair contribution is deliberately limited.
    return individual + 0.15 * pair_total


def generate_candidates(history, count, seed):
    scores, pairs = build_scores(history)

    rng = random.Random(seed)
    best_ticket = None
    best_score = float("-inf")

    seen = set()

    for _ in range(count):
        ticket = tuple(sorted(rng.sample(range(1, 46), 6)))

        if ticket in seen:
            continue

        seen.add(ticket)

        value = score_ticket(ticket, scores, pairs)

        if value > best_score:
            best_score = value
            best_ticket = ticket

    return list(best_ticket), best_score, scores


def backtest(draws):
    test_start = max(MIN_HISTORY, len(draws) - TEST_DRAWS)

    model_hits = []
    random_hits = []

    rng = random.Random(SEED)

    print("\n===== V14.1 WALK-FORWARD TEST =====")
    print("Test draws:", len(draws) - test_start)
    print("Candidate combinations per draw:", BACKTEST_CANDIDATES)

    for i in range(test_start, len(draws)):
        history = draws[:i]
        actual = set(draws[i]["result"])

        prediction, _, _ = generate_candidates(
            history,
            BACKTEST_CANDIDATES,
            SEED + i
        )

        hits = len(set(prediction) & actual)
        model_hits.append(hits)

        for _ in range(RANDOM_TICKETS_PER_DRAW):
            ticket = set(rng.sample(range(1, 46), 6))
            random_hits.append(len(ticket & actual))

    model_average = sum(model_hits) / len(model_hits)
    random_average = sum(random_hits) / len(random_hits)

    print("\nV14.1 Engine A")
    print("Total hits:", sum(model_hits))
    print("Average hits per draw:", round(model_average, 4))

    print("\nRandom control")
    print("Average hits per ticket:", round(random_average, 4))
    print("Theoretical random baseline: 0.8000")

    print("\n===== HIT DISTRIBUTION =====")

    for k in range(7):
        print(f"{k} hits: {model_hits.count(k)} draws")

    print("\nDifference vs theoretical baseline:",
          round(model_average - 0.8, 4))

    print("IMPORTANT: This is exploratory testing,")
    print("not proof of predictive advantage.")


def main():
    draws = download_data()

    print("\n===== V14.1 DATA =====")
    print("Total draws:", len(draws))
    print("Latest draw:", draws[-1]["id"])
    print("Latest date:", draws[-1]["date"])

    backtest(draws)

    print("\n===== CURRENT ENGINE A CANDIDATE =====")

    ticket, score, scores = generate_candidates(
        draws,
        FINAL_CANDIDATES,
        SEED
    )

    print("V14.1 candidate:",
          "-".join(f"{n:02d}" for n in ticket))

    print("Candidate score:", round(score, 5))

    ranked = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0])
    )

    print("\nTop 15 individual scores:")

    for number, value in ranked[:15]:
        print(f"{number:02d}: {value:.5f}")

    print("\nV14.1 ENGINE A COMPLETE")
    print("Research only. No guaranteed prediction.")


if __name__ == "__main__":
    main()
