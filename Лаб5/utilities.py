import numpy as np
import matplotlib.pyplot as plt


def visualize_classifier(classifier, X, y, title=''):
    # Визначення мінімальних і максимальних значень для X та Y,
    # які будуть використані при побудові сітки
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Визначення кроку для побудови сітки
    mesh_step_size = 0.01

    # Визначення сітки значень X та Y
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
                                  np.arange(min_y, max_y, mesh_step_size))

    # Запуск класифікатора на сітці даних
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Переформатування вихідного масиву
    output = output.reshape(x_vals.shape)

    # Створення графіка
    plt.figure()

    # Вибір схеми кольорів для графіка
    plt.title(title)

    # Відображення результатів класифікатора
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray, shading='auto')

    # Накладання тренувальних точок на графік
    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black',
                linewidth=1, cmap=plt.cm.Paired)

    # Визначення меж графіка
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    # Визначення міток на осях X та Y
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))

    safe_title = "".join(c if c.isalnum() else "_" for c in title)
    plt.savefig(f"viz_{safe_title}.png", dpi=150)
    plt.close()
