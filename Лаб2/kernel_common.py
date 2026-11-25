import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def load_data(input_file='income_data.txt', max_datapoints=1000):
    X = []
    count_class1 = 0
    count_class2 = 0
    with open(input_file, 'r') as f:
        for line in f.readlines():
            if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
                break
            if '?' in line:
                continue
            data = line[:-1].split(', ')
            if data[-1] == '<=50K' and count_class1 < max_datapoints:
                X.append(data)
                count_class1 += 1
            if data[-1] == '>50K' and count_class2 < max_datapoints:
                X.append(data)
                count_class2 += 1

    X = np.array(X)
    label_encoder = []
    X_encoded = np.empty(X.shape)
    for i, item in enumerate(X[0]):
        if item.isdigit():
            X_encoded[:, i] = X[:, i]
        else:
            label_encoder.append(preprocessing.LabelEncoder())
            X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

    X_final = X_encoded[:, :-1].astype(int)
    y_final = X_encoded[:, -1].astype(int)
    return train_test_split(X_final, y_final, test_size=0.2, random_state=5)
