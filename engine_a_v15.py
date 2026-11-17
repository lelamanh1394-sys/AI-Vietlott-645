
import json
import math
import random
import re
import urllib.request
from collections import Counter
from itertools import combinations
from datetime import datetime

# ============================================================
# AI VIETLOTT MEGA 6/45 - ENGINE A V15
# Tu phan tich - Walk-forward test - Danh gia lai - 2 day so
# ============================================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "vietvudanh/vietlott-data/main/data/power645.jsonl"
)

TEST_DRAWS = 300
MIN_HISTORY = 200
CANDIDATES = 70
FINAL_CANDIDATES = 250
SEED = 1545

NUMBERS = range(1, 46)

STRATEGIES = [
    "freq60",
    "freq180",
    "freq500",
    "freq_all",
    "overdue",
    "trend",
    "hybrid",
    "balanced",
    "pair_hybrid",
]


# ------------------------------------------------------------
# 1. DOWNLOAD AND READ DATA
# ------------------------------------------------------------

def download_data():
    print("DOWNLOADING MEGA 6/45 DATA...")

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "Mozilla/5.0"}
    )

    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8")

    return raw


def extract_numbers(obj):
    """Tim 6 so trong cac dang JSON pho bien."""

    if isinstance(obj, list):
        values = [
            int(x) for x in obj
            if isinstance(x, (int, float))
            and not isinstance(x, bool)
            and float(x).is_integer()
        ]

        if len(values) == 6:
            result = sorted(values)
            if len(set(result)) == 6 and all(1 <= x <= 45 for x in result):
                return result

        for item in obj:
            found = extract_numbers(item)
            if found:
                return found

    if isinstance(obj, dict):
        keys = [
            "numbers", "result", "winning_numbers",
            "winningNumbers", "balls", "drawn_numbers",
            "combination", "red"
        ]

        for key in keys:
            if key in obj:
                found = extract_numbers(obj[key])
                if found:
                    return found

        # Thu tim danh sach so nam trong cac truong con
        for value in obj.values():
            found = extract_numbers(value)
            if found:
                return found

        # Truong hop du lieu co n1, n2... n6
        for key_group in [
            [f"n{i}" for i in range(1, 7)],
            [f"ball{i}" for i in range(1, 7)],
            [f"number{i}" for i in range(1, 7)]
        ]:
            if all(k in obj for k in key_group):
                try:
                    result = sorted(int(obj[k]) for k in key_group)
                    if len(set(result)) == 6 and all(
                        1 <= x <= 45 for x in result
                    ):
                        return result
                except (ValueError, TypeError):
                    pass

    return None


def extract_date(obj):
    if isinstance(obj, dict):
        for key in ["date", "draw_date", "drawDate", "ngay_quay"]:
            if key in obj:
                return str(obj[key])

        for value in obj.values():
            if isinstance(value, dict):
                found = extract_date(value)
                if found:
                    return found

    return ""


def load_draws():
    raw = download_data()
    draws = []

    for line in raw.splitlines():
        line = line.strip()

        if not line:
            continue

        try:
            obj = json.loads(line)
            numbers = extract_numbers(obj)

            if numbers:
                draws.append({
                    "date": extract_date(obj),
                    "numbers": numbers
                })

        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    # Loai cac ky trung lap
    unique = {}
    for draw in draws:
        key = tuple(draw["numbers"])
        unique[key] = draw

    draws = list(unique.values())

    if len(draws) < MIN_HISTORY + 10:
        raise RuntimeError(
            "Khong doc du du lieu. So ky hop le: "
            + str(len(draws))
            + ". Can kiem tra lai cau truc file JSONL."
        )

    # Sap xep theo ngay neu ngay co dang YYYY-MM-DD
    if all(
        re.match(r"^\d{4}-\d{2}-\d{2}", d["date"])
        for d in draws
    ):
        draws.sort(key=lambda x: x["date"])

    return draws


# ------------------------------------------------------------
# 2. STATISTICAL ANALYSIS
# ------------------------------------------------------------

def zscore(values):
    avg = sum(values) / len(values)
    variance = sum((x - avg) ** 2 for x in values) / len(values)
    sd = math.sqrt(variance)

    if sd == 0:
        return [0.0] * len(values)

    return [(x - avg) / sd for x in values]


def build_statistics(history):
    total = len(history)

    windows = {
        60: history[-60:],
        180: history[-180:],
        500: history[-500:],
        999999: history
    }

    frequency = {}

    for window, data in windows.items():
        counter = Counter(
            number
            for draw in data
            for number in draw
        )

        frequency[window] = {
            n: counter[n] for n in NUMBERS
        }

    # So ky ke tu lan xuat hien gan nhat
    overdue = {}

    for n in NUMBERS:
        gap = total

        for i in range(total - 1, -1, -1):
            if n in history[i]:
                gap = total - 1 - i
                break

        overdue[n] = gap

    # Xu huong 30 ky gan day so voi 120 ky truoc
    recent = history[-30:]
    previous = history[-150:-30]

    recent_count = Counter(
        n for draw in recent for n in draw
    )
    previous_count = Counter(
        n for draw in previous for n in draw
    )

    trend = {}

    for n in NUMBERS:
        recent_rate = recent_count[n] / max(1, len(recent))
        previous_rate = previous_count[n] / max(
            1, len(previous)
        )

        trend[n] = recent_rate - previous_rate

    # Tan suat cap so
    pair_count = Counter()

    for draw in history:
        for pair in combinations(sorted(draw), 2):
            pair_count[pair] += 1

    all_pairs = list(combinations(NUMBERS, 2))
    pair_values = [pair_count[p] for p in all_pairs]
    pair_z = zscore(pair_values)

    pair_score = {
        pair: pair_z[i]
        for i, pair in enumerate(all_pairs)
    }

    return {
        "frequency": frequency,
        "overdue": overdue,
        "trend": trend,
        "pair_score": pair_score
    }


# ------------------------------------------------------------
# 3. BUILD NINE STRATEGIES
# ------------------------------------------------------------

def build_strategies(stats):
    freq = stats["frequency"]
    overdue = stats["overdue"]
    trend = stats["trend"]

    z60 = zscore([freq[60][n] for n in NUMBERS])
    z180 = zscore([freq[180][n] for n in NUMBERS])
    z500 = zscore([freq[500][n] for n in NUMBERS])
    zall = zscore([freq[999999][n] for n in NUMBERS])
    zlate = zscore([overdue[n] for n in NUMBERS])
    ztrend = zscore([trend[n] for n in NUMBERS])

    result = {}

    for i, n in enumerate(NUMBERS):
        pass

    result["freq60"] = {
        n: z60[n - 1] for n in NUMBERS
    }

    result["freq180"] = {
        n: z180[n - 1] for n in NUMBERS
    }

    result["freq500"] = {
        n: z500[n - 1] for n in NUMBERS
    }

    result["freq_all"] = {
        n: zall[n - 1] for n in NUMBERS
    }

    result["overdue"] = {
        n: zlate[n - 1] for n in NUMBERS
    }

    result["trend"] = {
        n: ztrend[n - 1] for n in NUMBERS
    }

    result["hybrid"] = {
        n: (
            0.45 * z180[n - 1]
            + 0.25 * z60[n - 1]
            + 0.20 * zlate[n - 1]
            + 0.10 * ztrend[n - 1]
        )
        for n in NUMBERS
    }

    result["balanced"] = {
        n: (
            0.25 * z60[n - 1]
            + 0.25 * z180[n - 1]
            + 0.25 * zlate[n - 1]
            + 0.25 * ztrend[n - 1]
        )
        for n in NUMBERS
    }

    result["pair_hybrid"] = {
        n: (
            0.40 * z180[n - 1]
            + 0.20 * zlate[n - 1]
            + 0.20 * ztrend[n - 1]
            + 0.20 * z60[n - 1]
        )
        for n in NUMBERS
    }

    return result


# ------------------------------------------------------------
# 4. GENERATE AND SCORE COMBINATIONS
# ------------------------------------------------------------

def pair_bonus(ticket, pair_score):
    total = 0.0
    count = 0

    for a, b in combinations(ticket, 2):
        total += pair_score.get((a, b), 0.0)
        count += 1

    return total / max(1, count)


def ticket_quality(ticket, scores, pair_score):
    base = sum(scores[n] for n in ticket)

    odd_count = sum(n % 2 for n in ticket)
    total_sum = sum(ticket)

    # Chi la dieu chinh nhe, khong ep ket qua
    balance_penalty = 0.08 * abs(odd_count - 3)
    sum_penalty = 0.003 * abs(total_sum - 138)

    pair_value = pair_bonus(ticket, pair_score)

    return (
        base
        + 0.18 * pair_value
        - balance_penalty
        - sum_penalty
    )


def weighted_sample(scores, rng):
    remaining = list(NUMBERS)
    chosen = []

    for _ in range(6):
        minimum = min(scores[n] for n in remaining)

        weights = [
            math.exp(max(-5, min(5, scores[n] - minimum)))
            for n in remaining
        ]

        selected = rng.choices(
            remaining,
            weights=weights,
            k=1
        )[0]

        chosen.append(selected)
        remaining.remove(selected)

    return sorted(chosen)


def generate_ticket(scores, pair_score, seed, attempts=CANDIDATES):
    rng = random.Random(seed)

    best_ticket = None
    best_quality = float("-inf")

    for _ in range(attempts):
        ticket = weighted_sample(scores, rng)

        quality = ticket_quality(
            ticket, scores, pair_score
        )

        if quality > best_quality:
            best_quality = quality
            best_ticket = ticket

    return best_ticket


# ------------------------------------------------------------
# 5. WALK-FORWARD BACKTEST
# ------------------------------------------------------------

def run_backtest(draws):
    print("\n===== V15 WALK-FORWARD TEST =====")

    start = max(MIN_HISTORY, len(draws) - TEST_DRAWS)

    results = {
        name: [] for name in STRATEGIES
    }

    random_results = []

    for index in range(start, len(draws)):
        history = [
            d["numbers"] for d in draws[:index]
        ]

        actual = set(draws[index]["numbers"])

        stats = build_statistics(history)
        strategies = build_strategies(stats)

        for s_index, name in enumerate(STRATEGIES):
            ticket = generate_ticket(
                strategies[name],
                stats["pair_score"],
                SEED + index * 100 + s_index,
                attempts=35
            )

            hits = len(set(ticket) & actual)
            results[name].append(hits)

        rng = random.Random(SEED + index)

        random_ticket = sorted(
            rng.sample(list(NUMBERS), 6)
        )

        random_results.append(
            len(set(random_ticket) & actual)
        )

        if (index - start + 1) % 50 == 0:
            print(
                "Tested draws:",
                index - start + 1
            )

    print("\n===== BACKTEST RESULTS =====")

    summary = {}

    for name in STRATEGIES:
        values = results[name]
        average = sum(values) / max(1, len(values))
        three_plus = sum(x >= 3 for x in values)

        summary[name] = average

        print(
            f"{name:15s} | "
            f"Average hits: {average:.4f} | "
            f"3+ hits: {three_plus}"
        )

    random_average = (
        sum(random_results) / max(1, len(random_results))
    )

    print(
        f"\nRandom baseline average hits: "
        f"{random_average:.4f}"
    )

    best_name = max(summary, key=summary.get)

    print("Best historical strategy:", best_name)

    print(
        "Theoretical random average: 0.8000"
    )

    return best_name, summary, random_average


# ------------------------------------------------------------
# 6. GENERATE TWO FINAL COMBINATIONS
# ------------------------------------------------------------

def generate_final(draws, best_name, summary):
    history = [d["numbers"] for d in draws]

    stats = build_statistics(history)
    strategies = build_strategies(stats)

    # Xep hang chien luoc theo ket qua backtest
    ranked = sorted(
        summary,
        key=summary.get,
        reverse=True
    )

    pool = []

    for s_index, name in enumerate(ranked):
        for attempt in range(30):
            ticket = generate_ticket(
                strategies[name],
                stats["pair_score"],
                SEED + 900000 + s_index * 1000 + attempt,
                attempts=FINAL_CANDIDATES // 5
            )

            pool.append({
                "ticket": ticket,
                "strategy": name,
                "quality": ticket_quality(
                    ticket,
                    strategies[name],
                    stats["pair_score"]
                )
            })

    # Loai bo cac day trung nhau
    unique = {}

    for item in pool:
        key = tuple(item["ticket"])

        if (
            key not in unique
            or item["quality"] > unique[key]["quality"]
        ):
            unique[key] = item

    candidates = list(unique.values())
    candidates.sort(
        key=lambda x: x["quality"],
        reverse=True
    )

    first = candidates[0]

    second = None

    # Uu tien hai day khong trung qua 3 so
    for item in candidates[1:]:
        overlap = len(
            set(first["ticket"]) & set(item["ticket"])
        )

        if overlap <= 3:
            second = item
            break

    if second is None:
        second = next(
            item for item in candidates[1:]
            if item["ticket"] != first["ticket"]
        )

    print("\n================================")
    print("       V15 FINAL PREDICTION")
    print("================================")

    print(
        "Strategy ranking:",
        ", ".join(ranked)
    )

    print(
        "\nDAY SO 1:",
        " - ".join(f"{n:02d}" for n in first["ticket"])
    )

    print("Strategy:", first["strategy"])

    print(
        "\nDAY SO 2:",
        " - ".join(f"{n:02d}" for n in second["ticket"])
    )

    print("Strategy:", second["strategy"])

    overlap = len(
        set(first["ticket"]) & set(second["ticket"])
    )

    print("\nNumbers overlapping:", overlap)
    print("Total draws analyzed:", len(draws))
    print("Latest draw:", draws[-1]["numbers"])
    print("Latest date:", draws[-1]["date"])

    print("\nNOTE:")
    print(
        "Backtest is historical evaluation, "
        "not a guarantee of future winning."
    )

    print(
        "One ticket jackpot odds: 1 / 8,145,060"
    )

    print(
        "Two distinct tickets jackpot odds: "
        "2 / 8,145,060"
    )


# ------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------

def main():
    print("================================")
    print("       AI VIETLOTT V15")
    print("================================")

    draws = load_draws()

    print("Total valid draws:", len(draws))
    print("Latest date:", draws[-1]["date"])
    print("Latest numbers:", draws[-1]["numbers"])

    best_name, summary, random_average = run_backtest(draws)

    generate_final(
        draws,
        best_name,
        summary
    )


if __name__ == "__main__":
    main()
