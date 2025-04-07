import re
import math
import sympy
import numpy as np

# Константы
NUM_CRITERIA = 2  # Количество критериев
PRECISION = 8  # Точность вычислений
SEP = 25  # Ширина колонки при выводе


def print_table(system, basis_coef, non_basis_coef, basis_values, non_basis_values):
    """Функция для красивого вывода симплекс-таблицы"""
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


def count_scalar_product(vec1, vec2):
    """Вычисление скалярного произведения двух векторов"""
    res = 0
    for i in range(len(vec1)):
        res += (vec1[i] * vec2[i])
    return res


def create_simplex_table(system, basis_coef, non_basis_coef, basis_values, non_basis_values):
    """Создание симплекс-таблицы"""
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
    """Одна итерация симплекс-метода"""
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
    """Проверка неравенства"""
    inequality = inequality.replace('*', '')
    for i in range(len(variables)):
        if variables[i] not in inequality:
            continue
        inequality = inequality.replace(
            variables[i], '*' + str(optimal_basis_indices[i]))
    inequality += '- 0.1'
    result = str(sympy.sympify(inequality))
    return eval(result)


def dual_task(target_coefficients=None, boundaries=None):
    def perem(equation_system):

        target_coefficients0 = np.array([20, 0])
        y = np.linalg.solve(equation_system, target_coefficients0)
        y = np.array([5, 0])
        return y
    """Решение двойственной задачи"""
    target_coefficients = np.array(target_coefficients)
    boundaries = np.array(boundaries)
    constraint_matrix, _ = criteria_coefficients.copy(), boundaries.copy()
    transposed_constraint_matrix = np.transpose(constraint_matrix)
    optimal_basis_indices = np.array(system[-1][:-1])
    y = np.array([])
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
        """Первая теорема двойственности"""
        y = np.dot(np.array(basis_coef), D_inversed)
        G_min = np.dot(boundaries, y)
        print(f"Gmin равен {G_min} по первой теореме двойственности")
        assert abs(G_min - Q) < 0.00001

    def second_duality_theorem():
        """Вторая теорема двойственности"""
        nonlocal y
        zeros = list()
        for i in range(NUM_CRITERIA):
            if check_inequality(
                    criteria_function[i], basis_values, optimal_basis_indices):
                zeros.append(i)
        equation_system = transposed_constraint_matrix.copy()
        target_coefficients0 = target_coefficients.copy()
        for i in range(len(zeros)):
            equation_system = np.delete(
                equation_system, zeros[i], 0)
            target_coefficients0 = np.delete(target_coefficients0, zeros[i], 0)
            for j in range(i + 1, len(zeros)):
                zeros[j] -= 1
        y = perem(equation_system)
        G_min = np.dot(boundaries, y)
        print(f"Gmin равен {G_min} по второй теореме двойственности")
        assert abs(G_min - Q) < 0.00001

    def third_duality_theorem():
        """Третья теорема двойственности"""
        lower_bound = list()
        upper_bound = list()
        b = list()
        for i in range(len(D_inversed) - 1, -1, -1):
            positive = list()
            negative = list()
            bH = - math.inf
            bB = math.inf
            for j in range(len(D_inversed)):
                if D_inversed[i][j] > 0:
                    positive.append(
                        (boundaries[j], D_inversed[i][j]))
                elif D_inversed[i][j] < 0:
                    negative.append(
                        (boundaries[j], D_inversed[i][j]))
            if len(positive) > 1:
                elem = min(positive, key=lambda x: abs(
                    positive[0][0] / positive[0][1]))
                lower_bound.append(elem[0] / elem[1])
            elif len(positive) == 1:
                lower_bound.append(abs(positive[0][0] / positive[0][1]))
            else:
                lower_bound.append(bH)

            if len(negative) > 1:
                elem = max(negative, key=lambda x: abs(negative[0][0] / negative[0][1]))
                upper_bound.append(abs(elem[0] / elem[1]))
            elif len(negative) == 1:
                upper_bound.append(abs(negative[0][0] / negative[0][1]))
            else:
                upper_bound.append(bB)

            b.append(boundaries[i])
            print(f'Ресурс #{i + 1}')
            print(
                f'b{i + 1} ∈ ({lower_bound[-1]}; {upper_bound[-1]})')
            print(f'{i + 1}-й ресурс варьируется в интервале: ', end='')
            if lower_bound[-1] == - math.inf:
                print(f'({lower_bound[-1]}; ', end='')
            else:
                print(f'({b[-1] - lower_bound[-1]}; ', end='')
            if upper_bound[-1] == math.inf:
                print(f'{upper_bound[-1]})')
            else:
                print(f'{b[-1] + upper_bound[-1]})')

    first_duality_theorem()
    second_duality_theorem()
    third_duality_theorem()


# Основная программа
with open('TPR_PRACT6.csv', encoding='utf-8') as file:
    target_function = file.readline().rstrip()
    target_coefficients = [15, 5, 3, 20]
    criteria_function = ['4.0 + 2.0 + 1.0 + 4.0 <= 1200', '1.0 + 5.0 + 3.0 + 1.0 <= 1000']
    print(criteria_function)
    criteria_coefficients, boundaries = [[4, 2, 1, 4], [1, 5, 3, 1]], [1200, 1000]
    print('Переход к задаче линейного программирования:',
          target_function, sep='\n')

    system = list(map(list, list(zip(*criteria_coefficients))))
    system.append(boundaries.copy())
    basis_coef = [0] * NUM_CRITERIA
    non_basis_coef = target_coefficients.copy()
    non_basis_values = re.findall(r'[A-Za-z]\d{1,}', target_function)
    basis_values = [f'{non_basis_values[-1][0]}{i}' for i in range(
        int(non_basis_values[-1][1]) + 1, NUM_CRITERIA + int(non_basis_values[-1][1]) + 1)]

    # Решение симплекс-методом
    F_str, Q = create_simplex_table(
        system, basis_coef, non_basis_coef, basis_values, non_basis_values)
    num_iteration = 0
    while num_iteration < 50 and min(F_str) < 0:
        print(
            ('\x1b[6;30;42m' + f"Итерация #{num_iteration}" + '\x1b[0m').center(201))
        system, basis_coef, non_basis_coef, basis_values, non_basis_values, F_str, Q = simplex_iteration(
            system, basis_coef, non_basis_coef, basis_values, non_basis_values, F_str, Q)
        print_table(system, basis_coef, non_basis_coef,
                    basis_values, non_basis_values)
        num_iteration += 1

    if num_iteration != 50:
        print(f'Решение найдено! Общая прибыль составляет {
        round(Q, 3)} денежных единиц')
    else:
        print('Задача не имеет решения')
        exit(0)

    # Решение двойственной задачи
    dual_task(target_coefficients, boundaries)
