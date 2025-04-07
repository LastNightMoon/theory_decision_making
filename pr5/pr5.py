import re

NUM_CRITERIA = 2  # Количество ограничений в математической модели
PRECISION = 4  # Количество знаков после запятой при округлении
SEP = 25  # Разделитель для вывода таблицы


def print_table(system, coef_basis, coef_not_basis, basis_values, not_basis_values):
    print(' '.ljust(SEP), end='')
    print('Cj'.ljust(SEP), end='')
    for i in range(len(coef_not_basis) + 1):
        if i == len(coef_not_basis):
            print(' '.ljust(SEP))
        else:
            print(str(coef_not_basis[i]).ljust(SEP), end='')
    print('Cv'.ljust(SEP), end='')
    for i in range(len(not_basis_values) + 1):
        if i == 0:
            print(''.ljust(SEP), end='')
        else:
            print(str(not_basis_values[i - 1]).ljust(SEP), end='')
    print('A0'.ljust(SEP))

    system.insert(0, coef_basis + [' '])
    system.insert(1, basis_values + ['f'])

    for column in range(len(system[0])):
        for row in range(len(system)):
            print(str(system[row][column]).ljust(SEP), end='')
        print()

    del system[0]
    del system[0]


def get_coefficients(data):
    '''Функция для получения списка коэффициентов из системы ограничений'''
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
    '''Функция для расчёта скалярного произведения двух векторов'''
    return sum(v1 * v2 for v1, v2 in zip(vec1, vec2))


def create_simplex_table(system, coef_basis, coef_not_basis, basis_values, not_basis_values):
    F_str = [count_scalar_product(coef_basis, system[i]) - coef_not_basis[i] for i in range(len(not_basis_values))]
    Q = count_scalar_product(coef_basis, system[-1])

    for i in range(len(F_str)):
        system[i].append(F_str[i])
    system[-1].append(Q)

    return F_str, Q


def simplex_iteration(system, coef_basis, coef_not_basis, basis_values, not_basis_values, F_str, Q):
    index_column = F_str.index(min(F_str))
    mini = float('inf')

    for i in range(len(system[-1]) - 1):
        tmp = system[-1][i] / system[index_column][i]
        if tmp < mini:
            mini, index_row = tmp, i

    key_element = system[index_column][index_row]
    basis_values[index_row], not_basis_values[index_column] = not_basis_values[index_column], basis_values[index_row]
    coef_basis[index_row], coef_not_basis[index_column] = coef_not_basis[index_column], coef_basis[index_row]
    print(key_element)
    new_key_element = round(1 / key_element, PRECISION)
    data = [[0] * (len(coef_basis) + 1) for _ in range(len(coef_not_basis) + 1)]

    for i in range(len(system[index_column])):
        data[index_column][i] = - round(system[index_column][i] / key_element, PRECISION)
    for i in range(len(system)):
        data[i][index_row] = round(system[i][index_row] / key_element, PRECISION)
    data[index_column][index_row] = new_key_element

    for row in range(len(data[0])):
        for column in range(len(data)):
            if data[column][row] == 0:
                data[column][row] = round(((system[column][row] * key_element) - (
                        system[index_column][row] * system[column][index_row])) / key_element, PRECISION)

    F_str = [data[i][-1] for i in range(len(data) - 1)]
    Q = data[-1][-1]

    return data, coef_basis, coef_not_basis, basis_values, not_basis_values, F_str, Q


with open('TPR_PRACT5.csv', encoding='utf-8') as file:
    target_function = file.readline().rstrip()
    target_coefficients = [15, 5, 3, 20]

    criteria_function = [file.readline().rstrip() for _ in range(NUM_CRITERIA)]
    criteria_coefficients, boundaries = get_coefficients(criteria_function)

    print('Переходим к задаче линейного программирования:', target_function, sep='\n')
    for i in criteria_function:
        print(f'{{ {i} }}')

    system = list(map(list, zip(*criteria_coefficients)))
    system.append(boundaries)

    coef_basis = [0] * NUM_CRITERIA
    coef_not_basis = target_coefficients.copy()
    not_basis_values = re.findall(r'x\d+', target_function)
    basis_values = [f'{not_basis_values[-1][0]}{i}' for i in range(
        int(not_basis_values[-1][1]) + 1, NUM_CRITERIA + int(not_basis_values[-1][1]) + 1)]

    F_str, Q = create_simplex_table(system, coef_basis, coef_not_basis, basis_values, not_basis_values)
    num_iteration = 0

    while num_iteration < 50 and min(F_str) < 0:
        print(f'Итерация №{num_iteration}'.center(201))
        system, coef_basis, coef_not_basis, basis_values, not_basis_values, F_str, Q = simplex_iteration(
            system, coef_basis, coef_not_basis, basis_values, not_basis_values, F_str, Q)
        print_table(system, coef_basis, coef_not_basis, basis_values, not_basis_values)
        num_iteration += 1

    if num_iteration != 50:
        print(f'Решение найдено! Общая прибыль составила {Q} денежных единиц')
    else:
        print('Поставленная задача решения не имеет')
