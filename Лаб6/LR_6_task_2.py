import pandas as pd
import numpy as np
from collections import Counter

# Набір даних з методички (Завдання 2)
data = {
    'Day': [f'D{i}' for i in range(1, 15)],
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain', 'Rain', 'Overcast',
                'Sunny', 'Sunny', 'Rain', 'Sunny', 'Overcast', 'Overcast', 'Rain'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
                 'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 'Strong',
             'Weak', 'Weak', 'Weak', 'Strong', 'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes',
             'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'],
}
df = pd.DataFrame(data)
print(df)

n = len(df)
n_yes = (df['Play'] == 'Yes').sum()
n_no = (df['Play'] == 'No').sum()
print(f"\nВсього днів: {n}, Yes: {n_yes}, No: {n_no}")
print(f"P(Yes) = {n_yes}/{n} = {n_yes / n:.4f}")
print(f"P(No) = {n_no}/{n} = {n_no / n:.4f}")

print("\n=== Крок 1-2. Частотні таблиці та таблиці правдоподібності ===")
for attr in ['Outlook', 'Humidity', 'Wind']:
    print(f"\n--- {attr} ---")
    freq = pd.crosstab(df[attr], df['Play'])
    print("Частотна таблиця:\n", freq)
    likelihood_yes = freq['Yes'] / n_yes
    likelihood_no = freq['No'] / n_no
    print(f"P(x|Yes): \n{likelihood_yes.round(4)}")
    print(f"P(x|No): \n{likelihood_no.round(4)}")

print("\n=== Перевірка прикладу з методички: Sunny ===")
p_sunny_yes = (df[(df.Outlook == 'Sunny') & (df.Play == 'Yes')].shape[0]) / n_yes
p_sunny = (df.Outlook == 'Sunny').sum() / n
p_yes = n_yes / n
p_yes_given_sunny = (p_sunny_yes * p_yes) / p_sunny
print(f"P(Sunny|Yes) = {p_sunny_yes:.4f}")
print(f"P(Sunny) = {p_sunny:.4f}")
print(f"P(Yes|Sunny) = P(Sunny|Yes)*P(Yes)/P(Sunny) = {p_yes_given_sunny:.4f}  "
      f"(очікується 0.591)")

p_sunny_no = (df[(df.Outlook == 'Sunny') & (df.Play == 'No')].shape[0]) / n_no
p_no = n_no / n
p_no_given_sunny = (p_sunny_no * p_no) / p_sunny
print(f"P(No|Sunny) = {p_no_given_sunny:.4f}  (очікується 0.40)")


def predict(outlook, humidity, wind, verbose=True):
    """Наївний баєсівський розрахунок ймовірності Play=Yes/No за трьома ознаками."""
    def cond_prob(attr, value, play_value, n_play):
        count = df[(df[attr] == value) & (df['Play'] == play_value)].shape[0]
        return count / n_play

    p_outlook_yes = cond_prob('Outlook', outlook, 'Yes', n_yes)
    p_humidity_yes = cond_prob('Humidity', humidity, 'Yes', n_yes)
    p_wind_yes = cond_prob('Wind', wind, 'Yes', n_yes)
    score_yes = p_outlook_yes * p_humidity_yes * p_wind_yes * (n_yes / n)

    p_outlook_no = cond_prob('Outlook', outlook, 'No', n_no)
    p_humidity_no = cond_prob('Humidity', humidity, 'No', n_no)
    p_wind_no = cond_prob('Wind', wind, 'No', n_no)
    score_no = p_outlook_no * p_humidity_no * p_wind_no * (n_no / n)

    total = score_yes + score_no
    p_yes_final = score_yes / total
    p_no_final = score_no / total

    if verbose:
        print(f"\nУмови: Outlook={outlook}, Humidity={humidity}, Wind={wind}")
        print(f"Score(Yes) (ненормований) = {score_yes:.4f}")
        print(f"Score(No)  (ненормований) = {score_no:.4f}")
        print(f"P(Yes) = {p_yes_final:.4f}")
        print(f"P(No)  = {p_no_final:.4f}")
        print(f"Прогноз: {'Play' if p_yes_final > p_no_final else 'No play'} "
              f"(ймовірність {max(p_yes_final, p_no_final) * 100:.1f}%)")
    return p_yes_final, p_no_final


print("\n=== Перевірка прикладу з методички: Outlook=Rain, Humidity=High, Wind=Weak ===")
predict('Rain', 'High', 'Weak')
print("Очікується: P(Yes) ≈ 0.55, P(No) ≈ 0.45")
