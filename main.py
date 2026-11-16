
import json
import os
import requests
from collections import Counter

URL = (
    "https://raw.githubusercontent.com/"
    "vietvudanh/vietlott-data/main/data/power645.jsonl"
)

def download_data():
    print("Dang tai du lieu Mega 6/45...")

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
            raise ValueError("Du lieu khong hop le")

        draws.append(item)

    if not draws:
        raise ValueError("Khong co du lieu")

    os.makedirs("data", exist_ok=True)

    with open("data/mega645.jsonl", "w", encoding="utf-8") as f:
        for draw in draws:
            f.write(json.dumps(draw, ensure_ascii=False) + "\n")

    return draws


def analyze(draws):
    frequency = Counter()

    for draw in draws:
        frequency.update(draw["result"])

    print("\n===== AI VIETLOTT 6/45 =====")
    print("Tong so ky:", len(draws))
    print("Ky moi nhat:", draws[-1]["id"])
    print("Ngay quay:", draws[-1]["date"])

    print("\nTOP 10 SO XUAT HIEN NHIEU NHAT:")

    for number, count in frequency.most_common(10):
        print(f"So {number:02d}: {count} lan")


if __name__ == "__main__":
    history = download_data()
    analyze(history)
