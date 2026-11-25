import numpy as np

# Приклад з методички: точки (1,6), (2,5), (3,7), (4,10)
x = np.array([1, 2, 3, 4], dtype=float)
y = np.array([6, 5, 7, 10], dtype=float)
n = len(x)

# Нормальні рівняння методу найменших квадратів для y = b0 + b1*x
b1 = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
b0 = (np.sum(y) - b1 * np.sum(x)) / n

print(f"b0 = {b0:.4f}")
print(f"b1 = {b1:.4f}")

y_pred = b0 + b1 * x
residuals = y - y_pred
S = np.sum(residuals ** 2)
print("Залишки (y - y_pred):", np.round(residuals, 4))
print(f"Мінімальна сума квадратів похибок S = {S:.4f}")

print("\nПеревірка за допомогою numpy.polyfit (ступінь 1):")
coeffs = np.polyfit(x, y, 1)
print(f"b1 (nahyl) = {coeffs[0]:.4f}, b0 (peretyn) = {coeffs[1]:.4f}")
