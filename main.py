
import json
import os
import random
import requests
from collections import Counter

URL = (
    "https://raw.githubusercontent.com/"
    "vietvudanh/vietlott-data/main/data/power645.jsonl"
)

def download_data():
    print("DOWNLOADING MEGA 6/45 DATA")

    response = requests.get(URL, timeout=60)
    response.raise_for_status()

    draws = []

    for line in response.text.splitlines():
        if not line.strip():
            continue

        item = json.loads(line)
        numbers = item["result"]

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

    draws.sort(key=lambda x: x["date"])

    os.makedirs("data", exist_ok=True)

    with open("data/mega645.jsonl", "w", encoding="utf-8") as f:
        for draw in draws:
            f.write(json.dumps(draw, ensure_ascii=False) + "\n")

    return draws


def score_numbers(history):
    long_term = Counter()
    recent = Counter()

    for draw in history:
        long_term.update(draw["result"])

    for draw in history[-100:]:
        recent.update(draw["result"])

    scores = {}

    for number in range(1, 46):
        scores[number] = (
            0.5 * long_term[number] / len(history)
            + 0.5 * recent[number] / min(100, len(history))
        )

    return scores


def predict(history):
    scores = score_numbers(history)

    ranked = sorted(
        scores,
        key=lambda n: (-scores[n], n)
    )

    return sorted(ranked[:6])


def backtest(draws, start=200, step=1):
    results = []

    for i in range(start, len(draws), step):
        prediction = predict(draws[:i])
        actual = set(draws[i]["result"])

        hits = len(set(prediction) & actual)
        results.append(hits)

    print("\n===== BACKTEST =====")
    print("So ky kiem nghiem:", len(results))

    if results:
        average = sum(results) / len(results)
        print(f"So trung binh moi ky: {average:.4f}")

        for hits in range(7):
            count = results.count(hits)
            print(f"Trung {hits} so: {count} ky")

        print("Moc chon ngau nhien: 0.8 so/ky")


def main():
    draws = download_data()

    print("\n===== DU LIEU MEGA 6/45 =====")
    print("Tong so ky:", len(draws))
    print("Ky moi nhat:", draws[-1]["id"])
    print("Ngay quay:", draws[-1]["date"])

    print("\n===== TOP 10 TAN SUAT =====")

    frequency = Counter()

    for draw in draws:
        frequency.update(draw["result"])

    for number, count in frequency.most_common(10):
        print(f"So {number:02d}: {count} lan")

    backtest(draws)

    prediction = predict(draws)

    print("\n===== 6 SO THAM KHAO =====")
    print(" - ".join(f"{n:02d}" for n in prediction))
    print("Day so chi mang tinh nghien cuu, khong bao dam trung thuong.")


if __name__ == "__main__":
    main()
