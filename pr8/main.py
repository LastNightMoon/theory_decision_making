import pandas as pd

# Параметры задачи
budget = 400
step = 20
n = 6

# Данные f_k(x), включая f_k(0)
доходы = [
    [0, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 27, 30, 32, 35, 38, 40, 42, 45, 48, 50],
    [0, 2, 4, 6, 9, 11, 13, 15, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42],
    [0, 1, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, 31, 33, 35, 37, 39, 41],
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 31, 34, 36, 39, 41, 43, 45],
    [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 24, 26, 28, 31, 33, 35, 38, 40, 42],
    [0, 3, 5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 30, 32, 35, 38, 41, 44, 46, 48],
]

# Возможные состояния
states = [i for i in range(0, budget + step, step)]
num_states = len(states)

# Инициализация таблиц
Z = [[0] * num_states for _ in range(n + 1)]
X = [[0] * num_states for _ in range(n)]

# Обратный ход — условная оптимизация
for k in reversed(range(n)):
    for i, ξ in enumerate(states):
        max_profit = -1
        best_x = 0
        for x in range(0, ξ + step, step):
            x_index = x // step
            next_ξ = ξ - x
            next_index = next_ξ // step
            profit = доходы[k][x_index] + Z[k + 1][next_index]
            if profit > max_profit:
                max_profit = profit
                best_x = x
        Z[k][i] = max_profit
        X[k][i] = best_x

# Таблица 2 — итоговая
summary_data = []
for i, ξ in list(enumerate(states))[1:]:
    row = {"ξ": ξ}
    for k in range(n - 1, -1, -1):
        row[f"Z*{k + 1}"] = Z[k][i]
        row[f"x*{k + 1}"] = X[k][i]
    summary_data.append(row)
df_summary = pd.DataFrame(summary_data)

# Финальный путь — безусловная оптимизация
ξ = budget
solution = []
for k in range(n):
    x = X[k][ξ // step]
    solution.append(x)
    ξ -= x

total_profit = Z[0][budget // step]
final_rows = [{"Предприятие": f"Предприятие {i + 1}", "Выделено (млн руб.)": x} for i, x in enumerate(solution)]
final_rows.append({"Предприятие": "Суммарная прибыль", "Выделено (млн руб.)": total_profit})
df_final = pd.DataFrame(final_rows)

# Сохранение в Excel
writer = pd.ExcelWriter("распределение_таблицы.xlsx", engine='openpyxl')
df_summary.to_excel(writer, sheet_name="Таблица 2 (итоговая)", index=False)
df_final.to_excel(writer, sheet_name="Финальный ответ", index=False)
writer.close()

# Консольный вывод
print(f"Максимальный доход: {total_profit} млн. руб.")
print("Оптимальное распределение средств:")
for i, x in enumerate(solution):
    print(f"  Предприятию {i + 1}: {x} млн. руб.")

print("\nТаблицы сохранены в файл: распределение_таблицы.xlsx")
