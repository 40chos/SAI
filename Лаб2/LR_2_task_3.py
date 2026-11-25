import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# КРОК 1. Завантаження та вивчення даних
iris_dataset = load_iris()
print("Ключі iris_dataset:\n{}".format(iris_dataset.keys()))
print("\nНазви відповідей (сортів):\n{}".format(iris_dataset['target_names']))
print("\nНазва ознак:\n{}".format(iris_dataset['feature_names']))
print("\nТип масиву data: {}".format(type(iris_dataset['data'])))
print("Форма масиву data: {}".format(iris_dataset['data'].shape))
print("\nЗначення ознак для перших п'яти прикладів:\n{}".format(iris_dataset['data'][:5]))
print("\nТип масиву target: {}".format(type(iris_dataset['target'])))
print("Відповіді:\n{}".format(iris_dataset['target']))

X = iris_dataset['data']
y = iris_dataset['target']

# КРОК 3. Створення навчального та тестового наборів
X_train, X_validation, Y_train, Y_validation = train_test_split(
    X, y, test_size=0.20, random_state=1)

# КРОК 4. Класифікація (побудова моделі) - порівняння 6 алгоритмів
models = []
models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=200)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

print("\n=== Порівняння алгоритмів (10-кратна стратифікована крос-валідація) ===")
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# КРОК 6-7. Прогноз на контрольній вибірці та оцінка якості для найкращої моделі (SVM)
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

print("\n=== Оцінка якості моделі SVM на контрольній вибірці ===")
print("Accuracy:", accuracy_score(Y_validation, predictions))
print("\nМатриця помилок:\n", confusion_matrix(Y_validation, predictions))
print("\nЗвіт про класифікацію:\n", classification_report(Y_validation, predictions, target_names=iris_dataset['target_names']))

# КРОК 8. Прогноз для нової квітки: чашолистик 5 см x 2.9 см, пелюстка 1 см x 0.2 см
X_new = np.array([[5, 2.9, 1, 0.2]])
print("Форма масиву X_new: {}".format(X_new.shape))

prediction = model.predict(X_new)
print("Прогноз (код класу): {}".format(prediction))
print("Спрогнозований сорт: {}".format(iris_dataset['target_names'][prediction]))
