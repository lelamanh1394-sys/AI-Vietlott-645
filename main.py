
import json
import os
import random
import requests
from collections import Counter

URL = (
    "https://raw.githubusercontent.com/"
    "vietvudanh/vietlott-data/main/data/power645.jsonl"
)

START_TEST = 200
RECENT_WINDOW = 100
RANDOM_TRIALS = 20


def download_data():
    print("DANG TAI DU LIEU MEGA 6/45...")

    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    draws = []

    for line in response.text.splitlines():
        if not line.strip():
            continue

        try:
            item = json.loads(line)
            numbers = [int(n) for n in item["result"]]

            if (
                len(numbers) != 6
                or len(set(numbers)) != 6
                or any(n < 1 or n > 45 for n in numbers)
            ):
                continue

            draws.append({
                "id": item["id"],
                "date": item["date"],
                "result": sorted(numbers)
            })

        except (ValueError, KeyError, TypeError):
            continue

    draws.sort(key=lambda x: x["date"])

    if len(draws) <= START_TEST:
        raise ValueError("Khong du du lieu de kiem nghiem.")

    os.makedirs("data", exist_ok=True)

    with open("data/mega645.jsonl", "w", encoding="utf-8") as f:
        for draw in draws:
            f.write(json.dumps(draw, ensure_ascii=False) + "\n")

    return draws


def get_frequency(history):
    frequency = Counter()

    for draw in history:
        frequency.update(draw["result"])

    return frequency


def predict_full(history):
    frequency = get_frequency(history)
    ranked = sorted(
        range(1, 46),
        key=lambda n: (-frequency[n], n)
    )
    return set(ranked[:6])


def predict_recent(history):
    recent = history[-RECENT_WINDOW:]
    frequency = get_frequency(recent)

    ranked = sorted(
        range(1, 46),
        key=lambda n: (-frequency[n], n)
    )
    return set(ranked[:6])


def predict_combined(history):
    long_frequency = get_frequency(history)
    recent_frequency = get_frequency(history[-RECENT_WINDOW:])

    scores = {}

    for number in range(1, 46):
        long_rate = long_frequency[number] / len(history)
        recent_rate = recent_frequency[number] / min(
            RECENT_WINDOW, len(history)
        )

        scores[number] = 0.5 * long_rate + 0.5 * recent_rate

    ranked = sorted(
        range(1, 46),
        key=lambda n: (-scores[n], n)
    )

    return set(ranked[:6])


def backtest(draws):
    methods = {
        "Tan suat toan lich su": predict_full,
        "100 ky gan nhat": predict_recent,
        "Ket hop dai han va ngan han": predict_combined
    }

    results = {name: [] for name in methods}
    random_results = []

    rng = random.Random(6452026)

    for i in range(START_TEST, len(draws)):
        history = draws[:i]
        actual = set(draws[i]["result"])

        for name, predictor in methods.items():
            prediction = predictor(history)
            hits = len(prediction & actual)
            results[name].append(hits)

        # 20 bo so ngau nhien cho cung mot ky kiem nghiem
        random_hits = []

        for _ in range(RANDOM_TRIALS):
            ticket = set(rng.sample(range(1, 46), 6))
            random_hits.append(len(ticket & actual))

        random_results.extend(random_hits)

    print("\n===== SO SANH 4 PHUONG PHAP =====")
    print("So ky kiem nghiem:", len(results["100 ky gan nhat"]))

    for name, hits_list in results.items():
        average = sum(hits_list) / len(hits_list)
        print(f"\n{name}")
        print(f"So trung binh moi ky: {average:.4f}")

    random_average = sum(random_results) / len(random_results)

    print("\nDoi chung chon ngau nhien")
    print(f"So trung binh moi bo so: {random_average:.4f}")
    print("Moc ly thuyet ngau nhien: 0.8000")

    print("\n===== PHAN BO KET QUA =====")

    for name, hits_list in results.items():
        print(f"\n{name}")

        for count in range(7):
            print(f"Trung {count} so: {hits_list.count(count)} ky")

    print("\n===== 6 SO THAM KHAO HIEN TAI =====")

    for name, predictor in methods.items():
        prediction = sorted(predictor(draws))
        print(name + ": " + " - ".join(f"{n:02d}" for n in prediction))


def main():
    draws = download_data()

    print("\n===== DU LIEU MEGA 6/45 =====")
    print("Tong so ky:", len(draws))
    print("Ky moi nhat:", draws[-1]["id"])
    print("Ngay quay:", draws[-1]["date"])

    backtest(draws)

    print("\n===== HOAN TAT PHAN TICH =====")
    print("Ket qua chi phuc vu nghien cuu, khong bao dam trung thuong.")


if __name__ == "__main__":
    main()
