
import requests
import pandas as pd
from collections import Counter

def analyze_results(results):
    """Phân tích tần suất xuất hiện của 45 số."""
    numbers = []

    for draw in results:
        numbers.extend(draw)

    frequency = Counter(numbers)

    print("THỐNG KÊ MEGA 6/45")
    print("-" * 30)

    for number, count in frequency.most_common():
        print(f"Số {number:02d}: {count} lần")

    return frequency


def main():
    print("AI VIETLOTT 6/45")
    print("Hệ thống thống kê đã khởi động.")

    # Dữ liệu mẫu để kiểm tra chương trình
    sample_results = [
        [1, 8, 15, 22, 31, 44],
        [3, 8, 17, 22, 35, 42],
        [1, 12, 17, 25, 31, 40],
    ]

    analyze_results(sample_results)


if __name__ == "__main__":
    main()
