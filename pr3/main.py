import numpy as np

criteria_names = ["Цена", "Мощность", "Расход топлива", "Объём багажника"]

comparison_matrices = {
    "Критерии": np.array([
        [1,     4,      3,      7],
        [1 / 4, 1,      3,      3],
        [1 / 3, 1 / 3,  1,      4],
        [1 / 7, 1 / 3,  1 / 4,  1]
    ]),
    "Цена": np.array([
        [1,     5, 5,   3  , 1],
        [1 / 5, 1, 1, 1 / 3, 1 / 5],
        [1 / 5, 1, 1, 1 / 3, 1 / 5],
        [1 / 3, 3, 3,   1  , 1/ 3],
        [1,     5, 5, 3, 1]
    ]),
    "Мощность": np.array([
        [1, 4, 4, 3, 1 / 2],
        [1 / 4, 1, 1, 1 / 2, 1 / 5],
        [1 / 4, 1, 1, 1, 1 / 5],
        [1 / 3, 2, 1, 1, 1 / 4],
        [2, 5, 5, 4, 1]
    ]),
    "Расход топлива": np.array([
        [1,       5 ,   5   ,   4   , 3],
        [1 / 5,   1 , 1 / 2 , 1 / 2 , 2],
        [1 / 5,   2 ,   1   , 1 / 2 , 2],
        [1 / 4,   2 ,   2   ,   1   , 4],
        [1 / 3, 1 / 2, 1 / 2,   1/4   , 1]
    ]),
    "Объём багажника": np.array([
        [1, 2, 3, 4, 1],
        [1 / 2, 1, 2, 3, 1 / 2],
        [1 / 3, 1 / 2, 1, 1, 1 / 3],
        [1 / 4, 1 / 3, 1, 1, 1 / 4],
        [1, 2, 3, 4, 1]
    ])
}


def get_dels(ИС, СИ = 1.12):

    return ИС / СИ
def geometric_mean(row):
    return np.prod(row) ** (1 / len(row))


def check_consistency(matrix, priority_vector, name):
    '''Функция для определения согласованности матрицы'''
    S = np.sum(matrix, axis=0)

    P = S * priority_vector
    lambda_max = np.sum(P)
    n = len(matrix)

    # print(lambda_max, (lambda_max - n) / (n - 1))
    if len(matrix) == 4:
        ОС = get_dels((lambda_max - n) / (n - 1), 0.9)
    else:
        ОС = get_dels((lambda_max - n) / (n - 1))

    print(f'{name} ОС = ИС/СИ = {ОС:.3f}')

    return ОС


VK_values = {}
K = 0
for name, matrix in comparison_matrices.items():
    VK_values[name] = np.array([])
    b = []
    for index, row in enumerate(matrix):
        b.append(geometric_mean(row))
        VK_values[name] = np.append(VK_values[name], b[-1])
    sumb = round(sum(b), 3)
    # print(sumb)
    ws = []
    for index, data in enumerate(b):
        # print(data)
        ws.append(round(data / float(sumb), 3))
    K += 1
    # print(ws)
    check_consistency(matrix, ws, name)

normalized_priorities = {name: VK / np.sum(VK) for name, VK in VK_values.items()}

W3K = np.array([normalized_priorities["Цена"],
                normalized_priorities["Мощность"],
                normalized_priorities["Расход топлива"],
                normalized_priorities["Объём багажника"]])

W2 = np.array(normalized_priorities["Критерии"])
final_priorities = np.dot(W3K.T, W2)
print(*[f'Альтернатива {i + 1} - {data:.3f}' for i, data in enumerate(final_priorities)], sep='\n')