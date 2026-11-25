import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
import yfinance as yf

# matplotlib.finance давно видалено, тому котирування отримуємо через yfinance

# Вхідний файл із символічними позначеннями компаній
input_file = 'company_symbol_mapping.json'

# Завантаження з файлу масиву відповідності символів компаній їх повним назвам
with open(input_file, 'r') as f:
    company_symbols_map = json.loads(f.read())

symbols, names = np.array(list(company_symbols_map.items())).T

# Завантаження архівних даних котирувань за останні 2 роки
start_date = "2023-01-01"
end_date = "2025-01-01"

print(f"Завантаження котирувань {len(symbols)} компаній за період "
      f"{start_date} - {end_date}...")
data = yf.download(list(symbols), start=start_date, end=end_date,
                    progress=False, auto_adjust=False)

# Вилучення котирувань, що відповідають відкриттю та закриттю біржі
opening_quotes = data['Open']
closing_quotes = data['Close']

# Видаляємо компанії, для яких не вдалося завантажити повний ряд даних
valid_mask = ~(opening_quotes.isna().any() | closing_quotes.isna().any())
opening_quotes = opening_quotes.loc[:, valid_mask]
closing_quotes = closing_quotes.loc[:, valid_mask]
valid_symbols = opening_quotes.columns.to_numpy()
valid_names = np.array([company_symbols_map[s] for s in valid_symbols])
print(f"Успішно завантажено дані для {len(valid_symbols)} із {len(symbols)} компаній")

# Обчислення різниці між двома видами котирувань
quotes_diff = (closing_quotes - opening_quotes).to_numpy().T

# Нормалізація даних
X = quotes_diff.copy()
X /= X.std(axis=1)[:, np.newaxis]

# Створення моделі графа
edge_model = covariance.GraphicalLassoCV()

# Навчання моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X.T)

# Створення моделі кластеризації на основі поширення подібності,
# використовуючи щойно навчену крайову модель
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
num_labels = labels.max()

print(f"\nЗнайдено {num_labels + 1} кластерів на фондовому ринку:\n")
for i in range(num_labels + 1):
    print("Cluster", i + 1, "==>", ', '.join(valid_names[labels == i]))
