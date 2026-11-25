import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

print("Коефіцієнти регресії (regr.coef_):\n", regr.coef_)
print("\nВільний член (regr.intercept_):", regr.intercept_)
print("\nR2 score:", round(r2_score(ytest, ypred), 4))
print("Mean absolute error:", round(mean_absolute_error(ytest, ypred), 4))
print("Mean squared error:", round(mean_squared_error(ytest, ypred), 4))

fig, ax = plt.subplots()
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0))
ax.plot([ytest.min(), ytest.max()], [ytest.min(), ytest.max()], 'k--', lw=4)
ax.set_xlabel('Виміряно')
ax.set_ylabel('Передбачено')
plt.title("Лінійна регресія: набір даних diabetes")
plt.savefig("task4_plot.png", dpi=150)
print("\nГрафік збережено у task4_plot.png")
