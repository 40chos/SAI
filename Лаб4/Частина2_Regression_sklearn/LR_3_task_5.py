import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)

# Варіант 4 (№ за списком 24): y = 0.7*x^2 + x + 3 + гауссів шум
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)

plt.figure()
plt.scatter(X, y)
plt.xlabel('x1')
plt.ylabel('y')
plt.title('Варіант 4: y = 0.7x^2 + x + 3 + шум')
plt.savefig("task5_data.png", dpi=150)

# Лінійна регресія (пряма лінія)
lin_reg = LinearRegression()
lin_reg.fit(X, y)
print("Лінійна регресія: intercept =", lin_reg.intercept_, " coef =", lin_reg.coef_)

x_line = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
y_line_lin = lin_reg.predict(x_line)

plt.figure()
plt.scatter(X, y, label="Дані")
plt.plot(x_line, y_line_lin, color='red', label="Лінійна регресія")
plt.xlabel('x1')
plt.ylabel('y')
plt.legend()
plt.title('Лінійна регресія (варіант 4)')
plt.savefig("task5_linear.png", dpi=150)

# Поліноміальна регресія (ступінь 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print("\nX[0] =", X[0], "  X_poly[0] =", X_poly[0])

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
print("\nПоліноміальна регресія: intercept =", poly_reg.intercept_, " coef =", poly_reg.coef_)

x_line_poly = poly_features.transform(x_line)
y_line_poly = poly_reg.predict(x_line_poly)

plt.figure()
plt.scatter(X, y, label="Дані")
plt.plot(x_line, y_line_poly, color='red', label="Поліноміальна регресія (ст.2)")
plt.xlabel('x1')
plt.ylabel('y')
plt.legend()
plt.title('Поліноміальна регресія (варіант 4)')
plt.savefig("task5_poly.png", dpi=150)

print(f"\nМодель варіанта (математичне рівняння): y = 0.7*x1^2 + 1.0*x1 + 3.0 + гауссів шум")
print(f"Отримана модель регресії з передбаченими коефіцієнтами: "
      f"y = {poly_reg.coef_[0][1]:.2f}*x1^2 + {poly_reg.coef_[0][0]:.2f}*x1 + {poly_reg.intercept_[0]:.2f}")
