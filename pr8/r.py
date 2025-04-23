import numpy as np
import pandas as pd

budget = 400
step = 20
n = 6

# Исходные доходы (f_k(x)), x от 0 до 400 включительно с шагом 20 (21 значение)
доходы = [
    [0, 3, 5, 8, 10, 12, 15, 18, 20, 22, 25, 27, 30, 32, 35, 38, 40, 42, 45, 48, 50],
    [0, 2, 4, 6, 9, 11, 13, 15, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42],
    [0, 1, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 29, 31, 33, 35, 37, 39, 41],
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28, 31, 34, 36, 39, 41, 43, 45],
    [0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 24, 26, 28, 31, 33, 35, 38, 40, 42],
    [0, 3, 5, 7, 9, 11, 13, 16, 18, 20, 22, 24, 27, 30, 32, 35, 38, 41, 44, 46, 48],
]
# budget = 200
# step = 40
# n = 4
#
#
# # Добавляем нули для f_k(0)
# доходы = [
#     [0, 8, 10, 11, 12, 18],  # f1(x)
#     [0, 6, 9, 11, 13, 15],   # f2(x)
#     [0, 3, 4, 7, 11, 18],    # f3(x)
#     [0, 4, 6, 8, 13, 16],    # f4(x)
# ]


num_enterprises = n


def profit_from_table(enterprise, x, profit_table):
    """Возвращает доход f_k(x) из таблицы для предприятия enterprise и суммы x."""
    if x == 0:
        return 0
    x_values = profit_table[enterprise][:, 0]
    idx = np.where(x_values == x)[0]
    if len(idx) == 0:
        return 0
    return profit_table[enterprise][idx[0], 1]


def print_allocation_table(initial_funds, profit_table):
    global step, num_enterprises
    steps = [i for i in range(0, initial_funds + step, step)]

    # Массивы для динамического программирования (максимальная прибыль и оптимальные выделения)
    optimal_profits = np.zeros((len(steps), num_enterprises + 1))
    allocations = np.zeros((len(steps), num_enterprises), dtype=int)

    # Инициализация для последнего предприятия: прибыль = доход из таблицы при выделении всех средств
    for i, remaining_funds in enumerate(steps):
        optimal_profits[i, num_enterprises] = profit_from_table(num_enterprises - 1, remaining_funds, profit_table)
        allocations[i, num_enterprises - 1] = remaining_funds

    # Построение оптимальных решений для предприятий с конца к началу
    for k in range(num_enterprises - 1, 0, -1):
        for i, remaining_funds in enumerate(steps):
            max_profit = -1
            best_alloc = 0
            for alloc in steps:
                if alloc <= remaining_funds:
                    idx_next = steps.index(remaining_funds - alloc)
                    current_profit = profit_from_table(k - 1, alloc, profit_table) + optimal_profits[idx_next, k + 1]
                    if current_profit > max_profit:
                        max_profit = current_profit
                        best_alloc = alloc
            optimal_profits[i, k] = max_profit
            allocations[i, k - 1] = best_alloc

    # Печать первой подробной таблицы (Таблица 1) без столбцов для 𝑓6
    print(
        f"{'𝜉𝑘−1':<10}{'𝑥𝑘':<10}{'𝜉𝑘':<10}"
        f"{'𝑓5(𝑥5)':<10}{'Z*(𝜉5)':<10}{'Z5(𝜉4,𝑥5)':<15}"
        f"{'𝑓4(𝑥4)':<10}{'Z*(𝜉4)':<10}{'Z4(𝜉3,𝑥4)':<15}"
        f"{'𝑓3(𝑥3)':<10}{'Z*(𝜉3)':<10}{'Z3(𝜉2,𝑥3)':<15}"
        f"{'𝑓2(𝑥2)':<10}{'Z*(𝜉2)':<10}{'Z2(𝜉1,𝑥2)':<15}"
        f"{'𝑓1(𝑥1)':<10}{'Z*(𝜉1)':<10}{'Z1(𝜉0,𝑥1)':<15}"
    )

    detailed_rows = []

    for xi_k_minus_1 in range(step, initial_funds + 1, step):
        for alloc in steps:
            if alloc <= xi_k_minus_1:
                xi_k = xi_k_minus_1 - alloc
                profits = []
                z_stars = []
                z_ks = []
                # Для каждого предприятия считаем f_k(x_k), Z*(𝜉_k) и Z_k(𝜉_{k-1}, x_k)
                # Но выводим только с 5-го по 1-е предприятие
                for i in range(num_enterprises):
                    f_val = profit_from_table(i, alloc, profit_table)
                    profits.append(f_val)
                    if i < num_enterprises - 1:
                        idx_xi_k = steps.index(xi_k)
                        z_star = optimal_profits[idx_xi_k, i + 2]  # Берем Z* из optimal_profits
                    else:
                        z_star = 0
                    z_stars.append(z_star)
                    z_ks.append(f'{int(f_val)} + {int(z_star)} = {int(f_val + z_star)}')

                # Вывод строки
                if alloc == steps[1]:
                    print(f"{xi_k_minus_1:<10}{alloc:<10}{xi_k:<10}", end="")
                else:
                    print(f"{'':<10}{alloc:<10}{xi_k:<10}", end="")

                # Выводим только с 5-го по 1-е предприятие (индексы 4..0)
                for i in range(num_enterprises - 2, -1, -1):
                    print(f"{profits[i]:<10}{z_stars[i]:<10}{z_ks[i]:<15}", end="")
                print()

                row = {
                    "ξk-1": xi_k_minus_1, "xk": alloc, "ξk": xi_k,
                }
                for i in range(num_enterprises):
                    row[f"f{i + 1}(x{i + 1})"] = profits[i]
                    row[f"Z*(ξ{i + 1})"] = z_stars[i]
                    row[f"Z{i + 1}(ξk-1,x{i + 1})"] = z_ks[i]
                detailed_rows.append(row)

    # Печать итоговой таблицы (Таблица 2) — здесь

    print("\n\nИТОГОВАЯ ТАБЛИЦА (Таблица 2):")
    print(f"{'𝜉':<10}", end="")
    for k in range(num_enterprises, 0, -1):  # с 6 до 1 включительно
        print(f"{'Z*' + str(k):<10}{'x*' + str(k):<10}", end="")
    print()

    summary_data = []

    for i, xi in enumerate(steps[1:]):
        print(f"{xi:<10}", end="")
        row = {"ξ": xi}
        for j in reversed(range(num_enterprises)):
            print(f"{optimal_profits[i + 1, j + 1]:<10.0f}{allocations[i + 1, j]:<10}", end="")
            row[f"Z*{j + 1}"] = optimal_profits[i + 1, j + 1]
            row[f"x*{j + 1}"] = allocations[i + 1, j]
        summary_data.append(row)
        print()

    # Финальный ответ
    print("\n\nФИНАЛЬНЫЙ ОТВЕТ (безусловная оптимизация):")
    current_funds = initial_funds
    allocation_plan = []

    for step in range(num_enterprises):
        idx = steps.index(current_funds)
        x_star = allocations[idx, step]
        allocation_plan.append(x_star)
        current_funds -= x_star

    total_profit = optimal_profits[steps.index(initial_funds), 1]

    final_rows = []
    for i, x in enumerate(allocation_plan):
        print(f"Предприятию {i + 1} выделить: {x} млн. руб.")
        final_rows.append({"Предприятие": f"Предприятие {i + 1}", "Выделено (млн руб.)": x})

    print(f"\nМаксимальный суммарный доход: {total_profit} млн. руб.")
    final_rows.append({"Предприятие": "Суммарная прибыль", "Выделено (млн руб.)": total_profit})

    # Сохраняем в Excel
    writer = pd.ExcelWriter("tables.xlsx", engine='openpyxl')

    # Порядок столбцов для подробной таблицы (Таблица 1) — без 6-го предприятия
    columns_order = ["ξk-1", "xk", "ξk"]
    for i in range(n - 1, 0, -1):
        columns_order.extend([
            f"f{i}(x{i})",
            f"Z*(ξ{i})",
            f"Z{i}(ξk-1,x{i})"
        ])

    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed = df_detailed[columns_order]
    df_detailed.to_excel(writer, sheet_name="Таблица 1 (подробная)", index=False)

    # Для итоговой таблицы (Таблица 2) сохраняем все с 6-го по 1-е предприятие
    df_summary = pd.DataFrame(summary_data)
    # Порядок столбцов для итоговой таблицы — с 6 до 1
    summary_columns = ["ξ"]
    for i in range(num_enterprises, 0, -1):
        summary_columns.extend([f"Z*{i}", f"x*{i}"])
    df_summary = df_summary[summary_columns]
    df_summary.to_excel(writer, sheet_name="Таблица 2 (итоговая)", index=False)

    pd.DataFrame(final_rows).to_excel(writer, sheet_name="Финальный ответ", index=False)

    writer.close()
    print("\nТаблицы сохранены в файл: tables.xlsx")


# Данные из таблицы для 6 предприятий


# Преобразуем в формат profit_table: список np.array([[x, f(x)], ...])
profit_table = []

for firm in доходы:
    firm_table = []
    for i, f in enumerate(firm):
        x = i * step  # соответствующее значение вложения
        firm_table.append([x, f])
    profit_table.append(np.array(firm_table))

# Начальная сумма
initial_funds = budget

# Запуск
print_allocation_table(initial_funds, profit_table)
