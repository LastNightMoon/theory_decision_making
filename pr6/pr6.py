import re
import math
import sympy
import numpy as np

NUM_CRITERIA = 2
PRECISION = 8
SEP = 25
y = None

def print_table(system, basis_coef, non_basis_coef, basis_values, non_basis_values):
    print(' '.ljust(SEP), end='')
    print('Cj'.ljust(SEP), end='')
    for i in range(len(non_basis_coef) + 1):
        if i == len(non_basis_coef):
            print(' '.ljust(SEP))
        else:
            print(str(non_basis_coef[i]).ljust(SEP), end='')
    print('Cv'.ljust(SEP), end='')
    for i in range(len(non_basis_values) + 1):
        if i == 0:
            print(''.ljust(SEP), end='')
        else:
            print(str(non_basis_values[i - 1]).ljust(SEP), end='')
    print('A0'.ljust(SEP))
    system.insert(0, basis_coef + [' '])
    system.insert(1, basis_values + ['f'])
    for column in range(len(system[0])):
        for row in range(len(system)):
            print(str(system[row][column]).ljust(SEP), end='')
        print()
    del system[0]
    del system[0]


def get_coefficients(data):
    '''Function to extract coefficients from the system of constraints'''
    criteria_coefficients, boundaries = [], []
    for exp in data:
        if '<=' in exp:
            parse = exp.split('<=')
        elif '>=' in exp:
            parse = exp.split('>=')
        elif '<' in exp:
            parse = exp.split('<')
        elif '>' in exp:
            parse = exp.split('>')
        elif '=' in exp:
            parse = exp.split('=')

        parse = list(map(str.strip, parse))
        boundaries.append(float(parse[1]))
        criteria_coefficients.append(
            [float(match.group()) for match in re.finditer(r'\b\d+(\.\d+)?\b', parse[0]) if match.group()])

    return criteria_coefficients, boundaries


def count_scalar_product(vec1, vec2):
    res = 0
    for i in range(len(vec1)):
        res += (vec1[i] * vec2[i])
    return res


def create_simplex_table(system, basis_coef, non_basis_coef, basis_values, non_basis_values):
    F_str = [0] * len(non_basis_values)
    for i in range(len(non_basis_values)):
        F_str[i] = count_scalar_product(
            basis_coef, system[i]) - non_basis_coef[i]
    Q = count_scalar_product(basis_coef, system[-1])
    for i in range(len(F_str)):
        system[i].append(F_str[i])
    system[-1].append(Q)
    return F_str, Q


def simplex_iteration(system, basis_coef, non_basis_coef, basis_values, non_basis_values, F_str, Q):
    index_column = F_str.index(min(F_str))
    mini = 1e10
    for i in range(len(system[-1]) - 1):
        tmp = system[-1][i] / system[index_column][i]
        if tmp < mini:
            mini = tmp
            index_row = i
    key_element = system[index_column][index_row]
    basis_values.insert(index_row, non_basis_values[index_column])
    non_basis_values.insert(index_column, basis_values.pop(index_row + 1))
    del non_basis_values[index_row]
    basis_coef[index_row], non_basis_coef[index_column] = non_basis_coef[index_column], basis_coef[index_row]
    new_key_element = round(1 / key_element, PRECISION)
    data = [[0] * (len(basis_coef) + 1)
            for _ in range(len(non_basis_coef) + 1)]
    for i in range(len(system[index_column])):
        data[index_column][i] = - \
            round(system[index_column][i] / key_element, PRECISION)
    for i in range(len(system)):
        data[i][index_row] = round(
            system[i][index_row] / key_element, PRECISION)
    data[index_column][index_row] = new_key_element
    for row in range(len(data[0])):
        for column in range(len(data)):
            if data[column][row] == 0:
                data[column][row] = round(((system[column][row] * key_element) - (
                        system[index_column][row] * system[column][index_row])) / key_element, PRECISION)
    F_str = [data[i][-1] for i in range(len(data) - 1)]
    Q = data[-1][-1]
    return data, basis_coef, non_basis_coef, basis_values, non_basis_values, F_str, Q


def check_inequality(inequality, variables, optimal_basis_indices):
    inequality = inequality.replace('*', '')

    for i in range(len(variables)):
        if variables[i] not in inequality:
            inequality = inequality.replace(
                'x2', '*' + '0')

        inequality = inequality.replace(
            variables[i], '*' + str(optimal_basis_indices[i]))

    inequality += '- 0.1'
    result = str(sympy.sympify(inequality))
    return eval(result)


def dual_task(target_coefficients=None, boundaries=None):
    target_coefficients = np.array(target_coefficients)
    boundaries = np.array(boundaries)
    constraint_matrix, _ = get_coefficients(criteria_function)
    transposed_constraint_matrix = np.transpose(constraint_matrix)
    optimal_basis_indices = np.array(system[-1][:-1])
    y = np.array([])
    y0 = np.array([])
    D = list()
    for i in range(len(basis_values)):
        index = int(basis_values[i][1:]) - 1
        if index < len(transposed_constraint_matrix):
            D.append(transposed_constraint_matrix[index])
        else:
            D.append(
                np.array([1 if i == j else 0 for j in range(NUM_CRITERIA)]))
    D_inversed = np.linalg.inv(np.transpose(D))

    def first_duality_theorem():
        global y
        y = np.dot(np.array(basis_coef), D_inversed)
        G_min = np.dot(boundaries, y)
        print("Матрица D, составленная из компонентов векторов входящих в последний базис," \
              "\nпри котором получен оптимальный план исходной задачи:")
        print(np.linalg.inv(np.transpose(D_inversed)), "\n")
        print("\nПреобразуем матрицу D в обратную (D_inversed):")
        print(D_inversed, "\n")
        print(f"Базисными переменными в симплекс-таблице являются C_b={basis_coef}, тогда ")
        y_ = np.dot(basis_coef, D_inversed)
        print(f'y^* = C_b * D_inv = {y_}')
        print(f"G_min = G(y^*) = (b, y^*) = {G_min} по первой теореме двойственности")
        print(f'f_max(x) = G_min(y) = {G_min}[тыс.ден.ед.].\n')
        assert abs(G_min - Q) < 0.00001

    def second_duality_theorem():
        # Восстановление значений переменных прямой задачи
        variables = {}
        # Получаем имена всех переменных (x1, x2, x3, x4 и т.д.)
        all_vars = sorted(set(basis_values + non_basis_values), key=lambda x: int(x[1:]))
        # Заполняем значения базисных переменных из последнего столбца симплекс-таблицы
        for var in all_vars:
            if var in basis_values:
                idx = basis_values.index(var)
                variables[var] = system[-1][idx]
            else:
                variables[var] = 0.0

        # Формируем уравнения для двойственных переменных
        equations = []
        y_symbols = sympy.symbols(f'y1:{len(boundaries) + 1}')  # Создаем символы y1, y2, ...

        # 1. Уравнения из положительных переменных прямой задачи
        for j in range(len(target_coefficients)):
            var_name = f'x{j + 1}'
            x_val = variables.get(var_name, 0.0)
            if x_val > 1e-6:
                # Получаем коэффициенты для переменной xj из каждого ограничения
                column = [row[j] for row in criteria_coefficients]
                lhs = sum(coeff * y for coeff, y in zip(column, y_symbols))
                equation = sympy.Eq(lhs, target_coefficients[j])
                equations.append(equation)

        # 2. Уравнения из неактивных ограничений прямой задачи (yi = 0)
        for i in range(len(boundaries)):
            # Вычисляем левую часть ограничения
            lhs = 0.0
            for j in range(len(target_coefficients)):
                var_name = f'x{j + 1}'
                lhs += criteria_coefficients[i][j] * variables.get(var_name, 0.0)
            # Проверяем, активно ли ограничение
            if not np.isclose(lhs, boundaries[i], atol=1e-6):
                equations.append(sympy.Eq(y_symbols[i], 0.0))

        # Решаем систему уравнений
        solution = sympy.solve(equations, y_symbols)
        if solution:
            # Если решение найдено, преобразуем в список значений
            y_values = [float(solution.get(y, 0.0)) for y in y_symbols]
            G_min = np.dot(boundaries, y_values)
            print(f"Решение системы: y = {y_values}")
            print(f"G_min = {G_min} ден.ед.")
        else:
            print("Система не имеет решения")

        # Проверка соответствия первой теореме
        assert np.isclose(G_min, Q, atol=1e-5), "Результаты теорем не совпадают!"

    def third_duality_theorem():
        lower_bounds = []
        upper_bounds = []
        # Извлекаем числовые индексы из basis_values (например, ['x3', 'x5', 'x1'] -> [3, 5, 1])
        indices = [int(var[1:]) for var in basis_values]
        # Сортируем индексы по возрастанию (например, [1, 3, 5])
        sorted_indices = sorted(indices)
        resource_column_order = [basis_values.index(f'x{idx}') for idx in sorted_indices]

        for resource_index, col in enumerate(resource_column_order):
            column = [row[col] for row in D_inversed]
            print(
                f"Ресурс #{resource_index + 1} (переменная x{sorted_indices[resource_index]}, столбец {col + 1} обратной матрицы):")

            positive = []
            negative = []
            for j in range(len(D_inversed[0])):
                val = D_inversed[resource_index][j]
                if val > 0:
                    positive.append(boundaries[resource_index] / val)
                elif val < 0:
                    negative.append(boundaries[resource_index] / abs(val))

            lower = min(positive) if positive else -math.inf
            upper = max(negative) if negative else math.inf

            lower_bounds.append(lower)
            upper_bounds.append(upper)

            print(f"b{resource_index + 1} ∈ ({round(boundaries[resource_index] - lower, 3)}; \
                  {boundaries[resource_index] + upper})")

        print("\nВлияние на целевую функцию:")
        total = 0
        global y
        for i in range(len(y)):
            if y[i] != 0:
                delta = y[i] * upper_bounds[i]
                total += delta
                print(f"∆𝐺𝑚𝑎𝑥{i + 1} = y{i + 1} * ∆b{i + 1}^B = {y[i]} * {upper_bounds[i]} = {delta}")
            else:
                print(upper_bounds[i])
        print(
            f'Совокупный эффект от изменения этих ресурсов приводит к изменению максимальной стоимости продукта ∆𝐺𝑚𝑎𝑥 на: {total}')
        print(
            f'Поэтому оптимальное значение целевой функции при максимальном изменении ресурсов: 𝐺𝑚𝑎𝑥 = {round(Q)} + {total} = {Q + total}[тыс.ден.ед./неделю]')

    print(('\x1b[6;30;42m' + f" ПЕРВАЯ ТЕОРЕМА ДВОЙСТВЕННОСТИ " + '\x1b[0m').center(150))
    first_duality_theorem()
    print(('\x1b[6;30;42m' + f" ВТОРАЯ ТЕОРЕМА ДВОЙСТВЕННОСТИ " + '\x1b[0m').center(150))
    second_duality_theorem()
    print(('\x1b[6;30;42m' + f" ТРЕТЬЯ ТЕОРЕМА ДВОЙСТВЕННОСТИ " + '\x1b[0m').center(150))
    third_duality_theorem()


with open('TPR_PRACT6.csv', encoding='utf-8') as file:
    target_function = file.readline().rstrip()
    target_coefficients = list(map(float, [i.group(1) for i in re.finditer(
        r'(\d+(\.\d+)?) {0,}[*]? {0,}\w', target_function)]))
    criteria_function = [file.readline().rstrip() for _ in range(NUM_CRITERIA)]
    criteria_coefficients, boundaries = get_coefficients(criteria_function)
    print('Переход к задаче линейного программирования:',
          target_function, sep='\n')
    for i in criteria_function:
        print("{ " + i)
    system = list(map(list, list(zip(*criteria_coefficients))))
    system.append(boundaries.copy())
    basis_coef = [0] * NUM_CRITERIA
    non_basis_coef = target_coefficients.copy()
    non_basis_values = re.findall(r'[A-Za-z]\d{1,}', target_function)
    basis_values = [f'{non_basis_values[-1][0]}{i}' for i in range(
        int(non_basis_values[-1][1]) + 1, NUM_CRITERIA + int(non_basis_values[-1][1]) + 1)]
    F_str, Q = create_simplex_table(
        system, basis_coef, non_basis_coef, basis_values, non_basis_values)
    num_iteration = 0
    while num_iteration < 50 and min(F_str) < 0:
        print(
            ('\x1b[6;30;42m' + f"Итерация #{num_iteration}" + '\x1b[0m').center(150))
        system, basis_coef, non_basis_coef, basis_values, non_basis_values, F_str, Q = simplex_iteration(
            system, basis_coef, non_basis_coef, basis_values, non_basis_values, F_str, Q)
        print_table(system, basis_coef, non_basis_coef,
                    basis_values, non_basis_values)
        num_iteration += 1
    if num_iteration != 50:
        print(f'Решение найдено! Общая прибыль составляет {round(Q, 3)}\n\n')
    else:
        print('Проблема не имеет решения')
        exit(0)
    dual_task(target_coefficients, boundaries)
