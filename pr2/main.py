# Словарь с метриками и функциями сравнения
import math
import numpy as np

metrics = {
    "Цена (100 тыс руб.)": {"weight": 5, "compare": lambda x, y: x < y},  # min лучше
    "Расход топлива (л/1000 км)": {"weight": 4, "compare": lambda x, y: x < y},  # min лучше
    "Мощность (л.с.)": {"weight": 3, "compare": lambda x, y: x > y},  # max лучше
    "Объем багажника (л)": {"weight": 2, "compare": lambda x, y: x > y},  # max лучше
}

cars = [
    {"name": "Toyota Camry", "metrics": {"Цена (100 тыс руб.)": 25, "Мощность (л.с.)": 200,
                                         "Расход топлива (л/1000 км)": 85, "Объем багажника (л)": 500}},
    {"name": "BMW 3 Series", "metrics": {"Цена (100 тыс руб.)": 42, "Мощность (л.с.)": 258,
                                         "Расход топлива (л/1000 км)": 70, "Объем багажника (л)": 480}},
    {"name": "Mercedes C-Class", "metrics": {"Цена (100 тыс руб.)": 45, "Мощность (л.с.)": 255,
                                             "Расход топлива (л/1000 км)": 72, "Объем багажника (л)": 460}},
    {"name": "Audi A4", "metrics": {"Цена (100 тыс руб.)": 38, "Мощность (л.с.)": 249,
                                    "Расход топлива (л/1000 км)": 74, "Объем багажника (л)": 450}},
    {"name": "Kia K5", "metrics": {"Цена (100 тыс руб.)": 23, "Мощность (л.с.)": 194,
                                   "Расход топлива (л/1000 км)": 80, "Объем багажника (л)": 510}},
    {"name": "Hyundai Sonata", "metrics": {"Цена (100 тыс руб.)": 24, "Мощность (л.с.)": 292,
                                           "Расход топлива (л/1000 км)": 78, "Объем багажника (л)": 500}},
    {"name": "Volkswagen Passat", "metrics": {"Цена (100 тыс руб.)": 31, "Мощность (л.с.)": 220,
                                              "Расход топлива (л/1000 км)": 76, "Объем багажника (л)": 520}},
    {"name": "Skoda Superb", "metrics": {"Цена (100 тыс руб.)": 34, "Мощность (л.с.)": 220,
                                         "Расход топлива (л/1000 км)": 75, "Объем багажника (л)": 550}},
    {"name": "Lexus ES", "metrics": {"Цена (100 тыс руб.)": 50, "Мощность (л.с.)": 302,
                                     "Расход топлива (л/1000 км)": 78, "Объем багажника (л)": 450}},
    {"name": "Volvo S60", "metrics": {"Цена (100 тыс руб.)": 40, "Мощность (л.с.)": 250,
                                      "Расход топлива (л/1000 км)": 73, "Объем багажника (л)": 430}},
]


def getD(a, b, st):
    p, n = 0, 0
    mp, mn = ["0" for _ in range(len(metrics))], ["0" for _ in range(len(metrics))]
    i = 0
    for k, v in metrics.items():
        if v["compare"](a['metrics'][k], b['metrics'][k]):
            mp[i] = str(v["weight"])
            p += v["weight"]
        else:
            n += v["weight"]
            mn[i] = str(v["weight"])
        i += 1
    print(f"P{st} = {" + ".join(mp)} = {p};\nN{st} = {" + ".join(mn)} = {n};")
    print(
        f"D{st} = P{st}/N{st} = {p}/{n} = {"∞" if n == 0 else round(p / n, 3)} {("< 1 — отбрасываем" if n != 0 and p / n < 1 else "= 1 — отбрасываем") if n != 0 and p / n <= 1 else "> 1 — принимаем"};")
    return "inf" if n == 0 else p / n


matr = [[-1 for _ in range(len(cars))] for _ in range(len(cars))]

for i in range(len(cars)):
    for j in range(i + 1, len(cars)):
        print(f"Рассмотрим альтернативы (i = {i + 1}, j = {j + 1}):")
        d = getD(cars[i], cars[j], f"{i + 1}{j + 1}")
        print(f"Рассмотрим альтернативы (i = {j + 1}, j = {i + 1}):")
        getD(cars[j], cars[i], f"{j + 1}{i + 1}")
        if d != "inf" and d < 1:
            matr[j][i] = 1 / d if d != 0 else "inf"
        elif d == 1:
            continue
        else:
            matr[i][j] = d

print(*map(lambda x: '\t'.join(map(lambda y: str(round(y, 1)) if "inf" != y else y, x)), matr), sep="\n")
bibl = []
for x, var in enumerate(matr):
    bibl.append((x + 1, sum([v == "inf" or v >= 2.0 for v in var])))

print(*sorted(bibl, key=lambda x:x[1], reverse=True), sep="\n")
