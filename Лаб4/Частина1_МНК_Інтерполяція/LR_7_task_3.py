import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Вектори даних з методички
x = np.array([0.1, 0.3, 0.4, 0.6, 0.7])
y = np.array([3.2, 3, 1, 1.8, 1.9])
n = len(x)  # 5 точок -> поліном степеня m = n-1 = 4

print("Крок 1. Заповнення матриці X (матриця Вандермонда для полінома степеня 4):")
X = np.vstack([x ** k for k in range(n)]).T  # стовпці: 1, x, x^2, x^3, x^4
print(X)

print("\nКрок 2. Отримання коефіцієнтів інтерполяційного полінома (розв'язок XA = Y):")
A = np.linalg.solve(X, y)
print("Коефіцієнти a0..a4:", np.round(A, 6))

print("\nКрок 3. Визначення функції полінома (ступінь 4):")


def P(xv):
    return sum(A[k] * xv ** k for k in range(n))


# Перевірка: поліном повинен точно проходити через задані точки
print("Перевірка у вузлах інтерполяції:")
for xi, yi in zip(x, y):
    print(f"  P({xi}) = {P(xi):.6f}  (задане y = {yi})")

print("\nКрок 4. Побудова графіка функції для інтерполюючого полінома")
x_dense = np.linspace(x.min(), x.max(), 400)
y_dense = P(x_dense)

plt.figure(figsize=(7, 5))
plt.plot(x_dense, y_dense, color='tab:red', lw=2, label='Інтерполяційний поліном P(x), степінь 4')
plt.scatter(x, y, color='tab:blue', s=70, zorder=3, label='Задані точки')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Інтерполяційний поліном Лагранжа 4-го степеня')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("interpolation.png", dpi=150)
print("Графік збережено у interpolation.png")

print("\nКрок 5. Визначення значення функції в проміжних точках 0.2 і 0.5:")
for xv in [0.2, 0.5]:
    print(f"  P({xv}) = {P(xv):.6f}")
