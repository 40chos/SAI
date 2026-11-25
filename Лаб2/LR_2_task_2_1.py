from sklearn.svm import SVC
from sklearn import metrics
from kernel_common import load_data

# беремо по 1000 точок кожного класу, щоб SVM з нелінійним ядром навчався швидко
X_train, X_test, y_train, y_test = load_data(max_datapoints=1000)

# Поліноміальне ядро (degree=8)
classifier = SVC(kernel='poly', degree=8)
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

accuracy = metrics.accuracy_score(y_test, y_test_pred)
precision = metrics.precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
recall = metrics.recall_score(y_test, y_test_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_test_pred, average='weighted')

print("=== SVM з поліноміальним ядром (degree=8) ===")
print("Accuracy: " + str(round(100 * accuracy, 2)) + "%")
print("Precision: " + str(round(100 * precision, 2)) + "%")
print("Recall: " + str(round(100 * recall, 2)) + "%")
print("F1 score: " + str(round(100 * f1, 2)) + "%")
