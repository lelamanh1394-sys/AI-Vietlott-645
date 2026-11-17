
import json
import math
import random
import requests
from collections import Counter
from itertools import combinations

# ==================================================
# ENGINE A V14.1.1
# Original V14.1 remains untouched
# ==================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "vietvudanh/vietlott-data/main/data/power645.jsonl"
)

TEST_DRAWS = 300
MIN_HISTORY = 200
CANDIDATES = 3000
FINAL_CANDIDATES = 12000
RANDOM_TICKETS = 500

SEED = 645141

FREQUENCY_WEIGHTS = {
    60: 0.20,
    180: 0.20,
    500: 0.20,
    0: 0.15
}

TRANSITION_WEIGHT = 0.25
PAIR_WEIGHT = 0.15

P_NUMBER = 6 / 45
P_PAIR = (6 / 45) * (5 / 44)


# ==================================================
# 1. DOWNLOAD DATA
# ==================================================

def download_data():

    print("DOWNLOADING MEGA 6/45 DATA")

    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    draws = []

    for line in response.text.splitlines():

        if not line.strip():
            continue

        try:
            item = json.loads(line)

            numbers = sorted(
                int(n) for n in item["result"]
            )

            if len(numbers) != 6:
                continue

            if len(set(numbers)) != 6:
                continue

            if min(numbers) < 1 or max(numbers) > 45:
                continue

            draws.append({
                "id": str(item["id"]),
                "date": str(item["date"]),
                "result": numbers
            })

        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue

    draws.sort(
        key=lambda x: (x["date"], x["id"])
    )

    if len(draws) <= MIN_HISTORY:
        raise ValueError("Insufficient valid draw data")

    return draws


# ==================================================
# 2. FREQUENCY ANALYSIS
# ==================================================

def frequency_signal(history, window):

    data = history[-window:]
    total = len(data)

    counts = Counter(
        n
        for draw in data
        for n in draw["result"]
    )

    expected = total * P_NUMBER
    variance = total * P_NUMBER * (1 - P_NUMBER)

    scores = {}

    for n in range(1, 46):

        if variance == 0:
            scores[n] = 0
            continue

        z = (
            counts[n] - expected
        ) / math.sqrt(variance)

        shrink = total / (total + 100)

        scores[n] = max(-2.5, min(2.5, z)) * shrink

    return scores


# ==================================================
# 3. TRANSITION ANALYSIS - FIXED
# ==================================================

def transition_signal(history):

    scores = {}

    if len(history) < 30:
        return {n: 0 for n in range(1, 46)}

    yes_count = Counter()
    yes_to_yes = Counter()
    no_to_yes = Counter()

    total = len(history) - 1

    for i in range(total):

        current = set(history[i]["result"])
        following = set(history[i + 1]["result"])

        for n in range(1, 46):

            if n in current:

                yes_count[n] += 1

                if n in following:
                    yes_to_yes[n] += 1

            else:

                # Actual absent -> present transitions
                if n in following:
                    no_to_yes[n] += 1

    for n in range(1, 46):

        count_yes = yes_count[n]
        count_no = total - count_yes

        if count_yes < 10 or count_no < 10:
            scores[n] = 0
            continue

        rate_yes = yes_to_yes[n] / count_yes

        # Correct denominator and actual transition count
        rate_no = no_to_yes[n] / count_no

        pooled = (
            yes_to_yes[n] + no_to_yes[n]
        ) / total

        se = math.sqrt(
            max(
                0,
                pooled * (1 - pooled)
                * (1 / count_yes + 1 / count_no)
            )
        )

        if se == 0:
            scores[n] = 0
            continue

        z = (rate_yes - rate_no) / se

        shrink = total / (total + 100)

        scores[n] = (
            max(-2.5, min(2.5, z)) * shrink
        )

    return scores


# ==================================================
# 4. PAIR ANALYSIS
# ==================================================

def pair_signal(history):

    total = len(history)
    counts = Counter()

    for draw in history:

        for pair in combinations(
            draw["result"], 2
        ):
            counts[pair] += 1

    expected = total * P_PAIR
    variance = total * P_PAIR * (1 - P_PAIR)

    result = {}

    if variance <= 0:
        return result

    for a in range(1, 46):

        for b in range(a + 1, 46):

            observed = counts[(a, b)]

            z = (
                observed - expected
            ) / math.sqrt(variance)

            shrink = total / (total + 200)

            result[(a, b)] = (
                max(-2.5, min(2.5, z)) * shrink
            )

    return result


# ==================================================
# 5. ENGINE A SCORING
# ==================================================

def build_scores(history):

    scores = {
        n: 0.0 for n in range(1, 46)
    }

    for window, weight in FREQUENCY_WEIGHTS.items():

        actual_window = (
            len(history) if window == 0
            else window
        )

        signal = frequency_signal(
            history, actual_window
        )

        for n in range(1, 46):
            scores[n] += signal[n] * weight

    transition = transition_signal(history)
    pairs = pair_signal(history)

    for n in range(1, 46):
        scores[n] += (
            transition[n] * TRANSITION_WEIGHT
        )

    return scores, pairs


# ==================================================
# 6. TICKET SCORING
# ==================================================

def score_ticket(ticket, scores, pairs):

    individual = sum(
        scores[n] for n in ticket
    )

    pair_total = sum(
        pairs.get(pair, 0)
        for pair in combinations(ticket, 2)
    )

    return individual + PAIR_WEIGHT * pair_total


# ==================================================
# 7. GENERATE CANDIDATES
# ==================================================

def generate_candidates(
    history,
    candidate_count,
    seed
):

    scores, pairs = build_scores(history)

    rng = random.Random(seed)

    best_ticket = None
    best_score = float("-inf")

    seen = set()

    for _ in range(candidate_count):

        ticket = tuple(sorted(
            rng.sample(range(1, 46), 6)
        ))

        if ticket in seen:
            continue

        seen.add(ticket)

        value = score_ticket(
            ticket, scores, pairs
        )

        if value > best_score:

            best_score = value
            best_ticket = ticket

    if best_ticket is None:
        raise RuntimeError("No candidate generated")

    return list(best_ticket), best_score, scores


# ==================================================
# 8. WALK-FORWARD BACKTEST
# ==================================================

def walk_forward_test(draws):

    start = max(
        MIN_HISTORY,
        len(draws) - TEST_DRAWS
    )

    model_results = []
    random_results = []

    rng = random.Random(SEED)

    print("\n===== V14.1.1 BACKTEST =====")

    for index in range(start, len(draws)):

        history = draws[:index]

        actual = set(
            draws[index]["result"]
        )

        prediction, _, _ = generate_candidates(
            history,
            CANDIDATES,
            SEED + index
        )

        model_hits = len(
            set(prediction) & actual
        )

        model_results.append(model_hits)

        # 500 random tickets, same target draw
        for _ in range(RANDOM_TICKETS):

            ticket = rng.sample(
                range(1, 46), 6
            )

            hits = len(
                set(ticket) & actual
            )

            random_results.append(hits)

    model_average = (
        sum(model_results) / len(model_results)
    )

    random_average = (
        sum(random_results) / len(random_results)
    )

    print("Test draws:", len(model_results))

    print(
        "Model average:",
        round(model_average, 5)
    )

    print(
        "Random average per ticket:",
        round(random_average, 5)
    )

    print(
        "Theoretical random baseline: 0.80000"
    )

    print("\n===== MODEL HIT DISTRIBUTION =====")

    for hits in range(7):

        print(
            f"{hits} hits:",
            model_results.count(hits)
        )

    print("\n===== FAIR COMPARISON =====")

    print(
        "Difference:",
        round(model_average - random_average, 5)
    )

    print(
        "Model better than random baseline:",
        sum(
            x > random_average
            for x in model_results
        )
    )

    print(
        "Model equal:",
        sum(
            x == random_average
            for x in model_results
        )
    )

    print(
        "Model below random baseline:",
        sum(
            x < random_average
            for x in model_results
        )
    )


# ==================================================
# 9. MAIN
# ==================================================

def main():

    random.seed(SEED)

    draws = download_data()

    print("\n===== V14.1.1 DATA =====")

    print("Total draws:", len(draws))
    print("Latest draw:", draws[-1]["id"])
    print("Latest date:", draws[-1]["date"])

    walk_forward_test(draws)

    print("\n===== FINAL CANDIDATE =====")

    ticket, score, scores = generate_candidates(
        draws,
        FINAL_CANDIDATES,
        SEED
    )

    print(
        "Engine A V14.1.1:",
        "-".join(f"{n:02d}" for n in ticket)
    )

    print("Score:", round(score, 5))

    print("\n===== TOP 15 NUMBERS =====")

    ranked = sorted(
        scores.items(),
        key=lambda x: (-x[1], x[0])
    )

    for number, value in ranked[:15]:

        print(
            f"{number:02d}: {value:.5f}"
        )

    print("\nENGINE A V14.1.1 COMPLETE")


if __name__ == "__main__":
    main()
