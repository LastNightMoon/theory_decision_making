from matplotlib import pyplot as plt

import numpy as np
from scipy.optimize import minimize

cons = [
    {'type': 'ineq', 'fun': lambda x: -(2 * x[0] + x[1] - 2)},  # 2x₁ + x₂ ≤ 2
    {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},  # x₁ - x₂ ≥ 0
    {'type': 'ineq', 'fun': lambda x: 1 - (x[0] - x[1])},  # x₁ - x₂ ≤ 1
    {'type': 'ineq', 'fun': lambda x: x[0]},  # x₁ ≥ 0
    {'type': 'ineq', 'fun': lambda x: x[1]}  # x₂ ≥ 0
]

# Целевая функция (пример: минимизировать x² + y²)
def objective(x):
    return x[0] * -1 - 3* x[1]
def objective_(x):
    return -(x[0] * -1 - 3* x[1])

# Ограничения системы (ваши неравенства)
def constraint1(x):
    return 2 * x[0] + x[1] - 2  # 2x₁ + x₂ ≤ 2 → преобразовано для метода штрафов


def constraint2(x):
    return x[0] - x[1]  # x₁ - x₂ ≥ 0 → преобразовано для метода штрафов


def constraint3(x):
    return 1 - (x[0] - x[1])  # x₁ - x₂ ≤ 1 → преобразовано для метода штрафов


# Метод штрафных функций
def penalty_method(x, mu=1.0):
    penalty = mu * (
            max(0, constraint1(x)) ** 2 +  # Для неравенства ≤
            max(0, -constraint2(x)) ** 2 +  # Для неравенства ≥
            max(0, constraint3(x)) ** 2 +  # Для неравенства ≤
            max(0, -x[0]) ** 2 +  # x₁ ≥ 0
            max(0, -x[1]) ** 2  # x₂ ≥ 0
    )
    return objective(x) + penalty


# Градиентный спуск
def gradient_descent(f, x0, lr=0.01, max_iter=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    history = [x.copy()]

    for _ in range(max_iter):
        # Численное вычисление градиента
        grad = np.zeros_like(x)
        h = 1e-5
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += h
            x_minus = x.copy()
            x_minus[i] -= h
            grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)

        # Обновление параметров
        x_new = x - lr * grad

        # Проверка на сходимость
        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        history.append(x.copy())

    return x, np.array(history)


# Начальная точка
x0 = [0.5, 0.5]

# Параметры метода
mu = 10.0  # Параметр штрафа
lr = 0.01  # Скорость обучения
max_iter = 1000

# Запуск градиентного спуска
solution, history = gradient_descent(lambda x: penalty_method(x, mu),
                                     x0, lr, max_iter)
result = minimize(objective, x0, method='SLSQP', constraints=cons)
print("Оптимальное решение:", result.x)
print("Миниамальное Значение функции:", result.fun)
result = minimize(objective_, x0, method='SLSQP', constraints=cons)
print("Оптимальное решение:", result.x)
print("Максимальное Значение функции:", result.fun)

x1 = np.linspace(-1, 3, 400)
x2_1 = 2 - 2 * x1
x2_2 = x1
x2_3 = x1 - 1
xgr = 3 * x1
xmax = 1 / 3 * x1
xmin = 1 / 3 * x1 + 3.1 / 7
x_zero = np.zeros_like(x1)
y_zero = np.zeros_like(x1)

plt.figure(figsize=(10, 8))
plt.xlim(-1.5, 2.5)
plt.ylim(-1.5, 2.5)
plt.plot(x1, x2_1, 'r-', label=r'$2x_1+x_2=2$', linewidth=2)
plt.plot(x1, x2_2, 'g-', label=r'$x_1-x_2=0$', linewidth=2)
plt.plot(x1, x2_3, 'b-', label=r'$x_1-x_2=1$', linewidth=2)
plt.plot(x1, xgr, 'y', label=r'$xgr$', linewidth=2)
plt.plot(x1, xmax, 'o', label=r'$xmax$', linewidth=2)
plt.plot(x1, xmin, 'p', label=r'$xmin$', linewidth=2)
plt.plot(x1, x_zero, 'k--', label='$x=0$', linewidth=1.5)
plt.plot(y_zero, x1, 'k--', label='$y=0$', linewidth=1.5)

x1_fill = np.linspace(0, 2, 400)
x2_fill = np.maximum(0, np.maximum(x1_fill - 1, np.minimum(x1_fill, 2 - 2 * x1_fill)))
vertices = np.array([[0, 0], [2 / 3, 2 / 3], [1, 0]])
plt.fill(vertices[:, 0], vertices[:, 1], color='lightblue', alpha=0.7, label="Допустимая область")

plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14)
plt.title('Область решений и целевая функция', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', linewidth=1)

# Вызовы вспомогательных функций

plt.show()