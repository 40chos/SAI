import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Варіант 4 (№ за списком 24)
m = 100
X = 6 * np.random.rand(m, 1) - 5
y = 0.7 * X ** 2 + X + 3 + np.random.randn(m, 1)


def plot_learning_curves(model, X, y, filename, title):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m_ in range(1, len(X_train)):
        model.fit(X_train[:m_], y_train[:m_])
        y_train_predict = model.predict(X_train[:m_])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m_], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))
    plt.figure()
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.xlabel("Розмір навчального набору")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title(title)
    plt.savefig(filename, dpi=150)
    plt.close()


# Криві навчання для звичайної лінійної регресії
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, "task6_curve_linear.png",
                      "Криві навчання: лінійна регресія (варіант 4)")
print("Криві навчання для лінійної моделі збережено у task6_curve_linear.png")

# Криві навчання для поліноміальної моделі 10-го ступеня
polynomial_regression_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_10, X, y, "task6_curve_poly10.png",
                      "Криві навчання: поліноміальна регресія ступеня 10 (варіант 4)")
print("Криві навчання для полінома 10-го ступеня збережено у task6_curve_poly10.png")

# Криві навчання для поліноміальної моделі 2-го ступеня
polynomial_regression_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_2, X, y, "task6_curve_poly2.png",
                      "Криві навчання: поліноміальна регресія ступеня 2 (варіант 4)")
print("Криві навчання для полінома 2-го ступеня збережено у task6_curve_poly2.png")
