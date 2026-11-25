import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, classification_report)

# Вхідні дані: ціни на квитки іспанських високошвидкісних залізниць (Renfe)
df = pd.read_csv('renfe_small.csv')
print("Розмір вихідного набору даних:", df.shape)
print("\nПропущені значення по стовпцях:\n", df.isna().sum())

# Видаляємо рядки з пропущеною ціною або типом тарифу
df = df.dropna(subset=['price', 'fare', 'train_class'])

# Завдання: за ознаками маршруту, типу поїзда, класу обслуговування та ціни
# визначити тип тарифу квитка - Promo чи Flexible (дві домінуючі категорії).
df = df[df['fare'].isin(['Promo', 'Flexible'])].copy()
print("\nРозмір після відбору класів Promo/Flexible:", df.shape)
print(df['fare'].value_counts())

# Кодування категоріальних ознак
encoders = {}
for col in ['origin', 'destination', 'train_type', 'train_class']:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])
    encoders[col] = le

feature_cols = ['origin_enc', 'destination_enc', 'train_type_enc', 'train_class_enc', 'price']
X = df[feature_cols].values
y = LabelEncoder().fit_transform(df['fare'])  # Promo=1, Flexible=0 (за алфавітом)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

# Наївний баєсівський класифікатор (Гаусів, оскільки серед ознак є неперервна price)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n=== Результати класифікації типу тарифу (Promo/Flexible) ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Precision:", round(precision_score(y_test, y_pred, average='weighted'), 4))
print("Recall:", round(recall_score(y_test, y_pred, average='weighted'), 4))
print("F1 score:", round(f1_score(y_test, y_pred, average='weighted'), 4))
print("\nМатриця помилок:\n", confusion_matrix(y_test, y_pred))
print("\nЗвіт про класифікацію:\n",
      classification_report(y_test, y_pred, target_names=['Flexible', 'Promo']))

# Апріорні ймовірності класів, виявлені моделлю
print("\nАпріорні ймовірності класів (class_prior_):", np.round(model.class_prior_, 4))
print("(Flexible, Promo)")

# Приклад прогнозу для нової точки даних
example = pd.DataFrame([{
    'origin': 'MADRID', 'destination': 'BARCELONA',
    'train_type': 'AVE', 'train_class': 'Turista', 'price': 75.0
}])
example_encoded = [
    encoders['origin'].transform(example['origin'])[0],
    encoders['destination'].transform(example['destination'])[0],
    encoders['train_type'].transform(example['train_type'])[0],
    encoders['train_class'].transform(example['train_class'])[0],
    example['price'][0],
]
pred = model.predict([example_encoded])[0]
proba = model.predict_proba([example_encoded])[0]
print(f"\nПриклад: MADRID -> BARCELONA, AVE, Turista, ціна 75.0 EUR")
print(f"Прогнозований тариф: {'Promo' if pred == 1 else 'Flexible'}")
print(f"Ймовірності (Flexible, Promo): {np.round(proba, 4)}")
