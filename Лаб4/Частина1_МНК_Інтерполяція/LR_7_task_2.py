import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Варіант 9 (№ за списком 24)
x = np.array([0.3, 1.0, 1.5, 2.2, 3.6, 4.5])
y = np.array([5, 10, 13, 16, 17, 18])
n = len(x)

b1 = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x ** 2) - np.sum(x) ** 2)
b0 = (np.sum(y) - b1 * np.sum(x)) / n

print(f"Варіант 9: x = {x.tolist()}, y = {y.tolist()}")
print(f"b0 = {b0:.4f}")
print(f"b1 = {b1:.4f}")
print(f"Рівняння апроксимуючої прямої: y = {b0:.4f} + {b1:.4f}*x")

y_pred = b0 + b1 * x
S = np.sum((y - y_pred) ** 2)
print(f"Сума квадратів похибок S = {S:.4f}")

x_line = np.linspace(min(x) - 0.3, max(x) + 0.3, 100)
y_line = b0 + b1 * x_line

plt.figure(figsize=(7, 5))
plt.scatter(x, y, color='tab:blue', s=60, label='Експериментальні точки', zorder=3)
plt.plot(x_line, y_line, color='tab:red', lw=2, label=f'y = {b0:.3f} + {b1:.3f}x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Апроксимація методом найменших квадратів (варіант 9)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("lsm_variant9.png", dpi=150)
print("Графік збережено у lsm_variant9.png")
