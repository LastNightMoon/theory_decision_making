cars = [
    {"№": 1, "Модель": "Toyota Camry", "Цена (100 тыс руб.)": 25, "Мощность (л.с.)": 200,
     "Расход топлива (л/1000 км)": 85, "Объем багажника (л)": 500},
    {"№": 2, "Модель": "BMW 3 Series", "Цена (100 тыс руб.)": 42, "Мощность (л.с.)": 258,
     "Расход топлива (л/1000 км)": 70, "Объем багажника (л)": 480},
    {"№": 3, "Модель": "Mercedes C-Class", "Цена (100 тыс руб.)": 45, "Мощность (л.с.)": 255,
     "Расход топлива (л/1000 км)": 72, "Объем багажника (л)": 460},
    {"№": 4, "Модель": "Audi A4", "Цена (100 тыс руб.)": 38, "Мощность (л.с.)": 249, "Расход топлива (л/1000 км)": 74,
     "Объем багажника (л)": 450},
    {"№": 5, "Модель": "Kia K5", "Цена (100 тыс руб.)": 23, "Мощность (л.с.)": 194, "Расход топлива (л/1000 км)": 80,
     "Объем багажника (л)": 510},
    {"№": 6, "Модель": "Hyundai Sonata", "Цена (100 тыс руб.)": 24, "Мощность (л.с.)": 292,
     "Расход топлива (л/1000 км)": 78, "Объем багажника (л)": 500},
    {"№": 7, "Модель": "Volkswagen Passat", "Цена (100 тыс руб.)": 31, "Мощность (л.с.)": 220,
     "Расход топлива (л/1000 км)": 76, "Объем багажника (л)": 520},
    {"№": 8, "Модель": "Skoda Superb", "Цена (100 тыс руб.)": 34, "Мощность (л.с.)": 220,
     "Расход топлива (л/1000 км)": 75, "Объем багажника (л)": 550},
    {"№": 9, "Модель": "Lexus ES", "Цена (100 тыс руб.)": 50, "Мощность (л.с.)": 302, "Расход топлива (л/1000 км)": 78,
     "Объем багажника (л)": 450},
    {"№": 10, "Модель": "Volvo S60", "Цена (100 тыс руб.)": 40, "Мощность (л.с.)": 250,
     "Расход топлива (л/1000 км)": 73, "Объем багажника (л)": 430},
]
rm = set()
metrics = {
    "Цена (100 тыс руб.)": (lambda x, y: x <= y, [*range(20, 330)], 33),  # чем меньше, тем лучше
    "Мощность (л.с.)": (lambda x, y: x >= y, [*range(0, 300)], 200),  # чем больше, тем лучше
    "Расход топлива (л/1000 км)": (lambda x, y: x <= y, [*range(75, 850)], 85),  # чем меньше, тем лучше
    "Объем багажника (л)": (lambda x, y: x >= y, [*range(0, 600)], 500),  # чем больше, тем лучше
}

for i in range(0, len(cars)):
    for j in range(i + 1, len(cars)):
        a, b = cars[i], cars[j]
        res = ([func[0](a[metric], b[metric]) for metric, func in metrics.items()])
        if all(res):
            rm.add(j)
            # print(a, b) # a > b
        if not any(map(lambda k: res[k[0]] and  a[k[1]] != b[k[1]], enumerate(metrics))):
            rm.add(i)
            # print(b, a) # a < b

for r in sorted(rm, reverse=True):
    cars.pop(r)

print("Парёто-оптимально множество:", "; ".join([f"{car['Модель']},{car['№']}" for car in cars]))
res = []
for car in cars:
    if all([car[metric] in func[1] for metric, func in metrics.items()]):
        res.append(car)

print("Оптимальное множество, удовлетворяющее ограничениям верхних и нижних границ:",
      "; ".join([f"{car["Модель"]},{car["№"]}" for car in res]))

# главный критерий цена
res = []
for car in cars:
    if all([func[0](car[metric], func[2]) for metric, func in metrics.items() if metric != "Цена (100 тыс руб.)"]):
        res.append(car)
car_r = sorted(res, key=lambda x: [x[i] for i in metrics])
print("По субоптимизации ",  "".join([f"{car["Модель"]},{car["№"]}" for car in car_r][0]))

a = list(sorted(cars, key=lambda item: tuple(item[key] for key in metrics.keys())))
print("Лексико-графическая оптимизация:", ([f"{car["Модель"]},{car["№"]}" for car in a])[0])
