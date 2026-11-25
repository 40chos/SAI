import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris          # функція завантаження набору даних Iris
from sklearn.cluster import KMeans              # алгоритм кластеризації k-середніх
from sklearn.metrics import pairwise_distances_argmin  # для власної реалізації алгоритму

# Завантаження набору даних Iris
iris = load_iris()
X = iris['data']   # ознаки: довжина/ширина чашолистка та пелюстки (4 ознаки, 150 прикладів)
y = iris['target']  # справжні мітки сортів (не використовуються при кластеризації без учителя)

# Створення об'єкта KMeans з 3 кластерами (у Iris рівно 3 сорти квітів)
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300,
                 tol=0.0001, verbose=0, random_state=0, copy_x=True, algorithm='lloyd')

# Навчання моделі кластеризації на вхідних даних
kmeans.fit(X)

# Отримання передбачених міток кластерів для кожної точки
y_kmeans = kmeans.predict(X)

# Побудова діаграми розсіювання: точки пофарбовані відповідно до знайденого кластера
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Отримання координат центрів кластерів
centers = kmeans.cluster_centers_

# Відображення центрів кластерів великими напівпрозорими чорними точками
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('KMeans (sklearn), n_clusters=3')
plt.savefig("task2_sklearn_kmeans.png", dpi=150)


def find_clusters(X, n_clusters, rseed=2):
    """Власна (спрощена) реалізація алгоритму k-середніх."""
    # 1. Випадковий вибір n_clusters точок як початкових центрів
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_clusters]
    centers = X[i]

    while True:
        # 2. Призначення кожної точки найближчому центру
        labels = pairwise_distances_argmin(X, centers)

        # 3. Обчислення нових центрів як середніх точок кожного кластера
        new_centers = np.array([X[labels == i].mean(0)
                                 for i in range(n_clusters)])

        # 4. Перевірка збіжності: якщо центри не змінились - завершуємо
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels


# Власна реалізація k-середніх з початковим значенням rseed=2
centers, labels = find_clusters(X, 3)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('Власна реалізація find_clusters, rseed=2')
plt.savefig("task2_custom_rseed2.png", dpi=150)

# Власна реалізація k-середніх з іншим початковим значенням rseed=0
centers, labels = find_clusters(X, 3, rseed=0)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('Власна реалізація find_clusters, rseed=0')
plt.savefig("task2_custom_rseed0.png", dpi=150)

# Порівняння зі стандартним KMeans із бібліотеки sklearn (n_clusters=3)
labels = KMeans(3, random_state=0, n_init=10).fit_predict(X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.title('sklearn.cluster.KMeans, random_state=0')
plt.savefig("task2_sklearn_random_state0.png", dpi=150)

print("Усі 4 графіки збережено (task2_*.png)")
