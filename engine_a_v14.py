
import requests
import numpy as np
import pandas as pd
import random
from collections import Counter

# ==================================================
# 1. CAU HINH MO HINH
# ==================================================

DATA_URL = "https://raw.githubusercontent.com/lelamanh1394-sys/AI-Vietlott-645/main/data.csv"

NUMBER_MIN = 1
NUMBER_MAX = 45
NUMBERS_PER_DRAW = 6

TEST_DRAWS = 300
CANDIDATES = 3000
RANDOM_SEED = 42

WEIGHT_FREQUENCY = 0.30
WEIGHT_RECENCY = 0.25
WEIGHT_GAP = 0.20
WEIGHT_TRANSITION = 0.25


# ==================================================
# 2. TAI DU LIEU
# ==================================================

def download_data():

    print("DOWNLOADING MEGA 6/45 DATA")

    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()

    from io import StringIO

    df = pd.read_csv(StringIO(response.text))

    print("Total rows:", len(df))

    return df


# ==================================================
# 3. CHUAN HOA DU LIEU
# ==================================================

def extract_draws(df):

    draws = []

    # Dieu chinh ten cot theo file CSV thuc te neu can
    number_columns = [
        col for col in df.columns
        if col.lower().startswith("n")
    ]

    if len(number_columns) < 6:
        raise ValueError("Khong tim thay 6 cot ket qua")

    for _, row in df.iterrows():

        numbers = []

        for col in number_columns[:6]:

            value = int(row[col])

            if NUMBER_MIN <= value <= NUMBER_MAX:
                numbers.append(value)

        if len(numbers) == 6 and len(set(numbers)) == 6:
            draws.append(sorted(numbers))

    return draws


# ==================================================
# 4. PHAN TICH TAN SUAT
# ==================================================

def calculate_frequency(history):

    counter = Counter()

    for draw in history:
        counter.update(draw)

    total = max(len(history), 1)

    scores = {}

    for number in range(1, 46):
        scores[number] = counter[number] / total

    return scores


# ==================================================
# 5. PHAN TICH TAN SUAT GAN DAY
# ==================================================

def calculate_recency(history, window=100):

    recent = history[-window:]

    counter = Counter()

    for draw in recent:
        counter.update(draw)

    scores = {}

    for number in range(1, 46):
        scores[number] = counter[number] / max(len(recent), 1)

    return scores


# ==================================================
# 6. PHAN TICH KHOANG CACH XUAT HIEN
# ==================================================

def calculate_gap(history):

    scores = {}

    for number in range(1, 46):

        gap = 0

        for draw in reversed(history):

            if number in draw:
                break

            gap += 1

        scores[number] = gap

    return scores


# ==================================================
# 7. PHAN TICH CHUYEN TIEP
# ==================================================

def calculate_transition(history):

    counter = Counter()

    for previous, current in zip(history[:-1], history[1:]):

        for number in current:
            counter[number] += 1

    total = max(len(history) - 1, 1)

    return {
        number: counter[number] / total
        for number in range(1, 46)
    }


# ==================================================
# 8. CHUAN HOA DIEM
# ==================================================

def normalize(scores):

    values = np.array(list(scores.values()), dtype=float)

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        return {key: 0.5 for key in scores}

    return {
        key: (value - minimum) / (maximum - minimum)
        for key, value in scores.items()
    }


# ==================================================
# 9. ENGINE A - CHAM DIEM 45 SO
# ==================================================

def engine_a(history):

    frequency = calculate_frequency(history)
    recency = calculate_recency(history)
    gap = calculate_gap(history)
    transition = calculate_transition(history)

    frequency = normalize(frequency)
    recency = normalize(recency)
    gap = normalize(gap)
    transition = normalize(transition)

    final_scores = {}

    for number in range(1, 46):

        score = (
            frequency[number] * WEIGHT_FREQUENCY
            + recency[number] * WEIGHT_RECENCY
            + gap[number] * WEIGHT_GAP
            + transition[number] * WEIGHT_TRANSITION
        )

        final_scores[number] = score

    return final_scores


# ==================================================
# 10. TAO BO SO UNG VIEN
# ==================================================

def generate_candidates(scores, count=CANDIDATES):

    numbers = list(scores.keys())

    probabilities = np.array([
        scores[number] + 0.0001
        for number in numbers
    ])

    probabilities = probabilities / probabilities.sum()

    candidates = set()

    while len(candidates) < count:

        selected = np.random.choice(
            numbers,
            size=6,
            replace=False,
            p=probabilities
        )

        combination = tuple(sorted(selected))

        candidates.add(combination)

    return list(candidates)


# ==================================================
# 11. CHON BO SO TOT NHAT
# ==================================================

def select_best(scores):

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return sorted([
        number for number, score in ranked[:6]
    ])


# ==================================================
# 12. TINH SO TRUNG
# ==================================================

def count_matches(prediction, actual):

    return len(set(prediction) & set(actual))


# ==================================================
# 13. BACKTEST WALK-FORWARD
# ==================================================

def walk_forward_test(draws):

    print("\n===== ENGINE A BACKTEST =====")

    start = max(100, len(draws) - TEST_DRAWS)

    total_matches = 0
    total_random_matches = 0

    wins = 0
    ties = 0
    losses = 0

    random_generator = random.Random(RANDOM_SEED)

    for index in range(start, len(draws)):

        history = draws[:index]
        actual = draws[index]

        scores = engine_a(history)

        prediction = select_best(scores)

        random_prediction = sorted(
            random_generator.sample(range(1, 46), 6)
        )

        model_hits = count_matches(prediction, actual)

        random_hits = count_matches(
            random_prediction, actual
        )

        total_matches += model_hits
        total_random_matches += random_hits

        if model_hits > random_hits:
            wins += 1
        elif model_hits == random_hits:
            ties += 1
        else:
            losses += 1

    tested = len(draws) - start

    print("Test draws:", tested)
    print("Model average:", total_matches / tested)
    print("Random average:", total_random_matches / tested)

    print("Wins:", wins)
    print("Ties:", ties)
    print("Losses:", losses)


# ==================================================
# 14. CHAY MO HINH
# ==================================================

def main():

    np.random.seed(RANDOM_SEED)

    df = download_data()

    draws = extract_draws(df)

    print("\n===== DATA =====")
    print("Total draws:", len(draws))

    if not draws:
        raise ValueError("Du lieu khong hop le")

    print("Latest draw:", draws[-1])

    walk_forward_test(draws)

    scores = engine_a(draws)

    print("\n===== TOP 10 NUMBERS =====")

    ranked = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    for number, score in ranked[:10]:
        print(f"{number:02d}: {score:.5f}")

    print("\n===== BEST 6 NUMBERS =====")

    print(select_best(scores))


if __name__ == "__main__":
    main()
