import pandas as pd

# Набір даних з Завдання 2 (класичний приклад PlayTennis)
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

n = len(df)
n_yes = (df['Play'] == 'Yes').sum()
n_no = (df['Play'] == 'No').sum()


def predict(outlook, humidity, wind):
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
    p_yes = score_yes / total
    p_no = score_no / total

    print(f"\nУмови: Outlook={outlook}, Humidity={humidity}, Wind={wind}")
    print(f"P(Outlook={outlook}|Yes)={p_outlook_yes:.4f}  "
          f"P(Humidity={humidity}|Yes)={p_humidity_yes:.4f}  "
          f"P(Wind={wind}|Yes)={p_wind_yes:.4f}  P(Yes)={n_yes / n:.4f}")
    print(f"Score(Yes) = {score_yes:.4f}")
    print(f"P(Outlook={outlook}|No)={p_outlook_no:.4f}  "
          f"P(Humidity={humidity}|No)={p_humidity_no:.4f}  "
          f"P(Wind={wind}|No)={p_wind_no:.4f}  P(No)={n_no / n:.4f}")
    print(f"Score(No)  = {score_no:.4f}")
    print(f"\nНормалізовані ймовірності:")
    print(f"P(Yes) = {p_yes:.4f}  ({p_yes * 100:.1f}%)")
    print(f"P(No)  = {p_no:.4f}  ({p_no * 100:.1f}%)")
    print(f"Прогноз: матч {'ВІДБУДЕТЬСЯ' if p_yes > p_no else 'НЕ відбудеться'}")
    return p_yes, p_no


# Варіант за списком № 24: 24 mod 5 = 4 -> варіант "4, 9, 14"
# Outlook = Sunny, Humidity = Normal, Wind = Strong
print("=" * 60)
print("Варіант 4, 9, 14 (№ за списком 24): "
      "Outlook=Sunny, Humidity=Normal, Wind=Strong")
print("=" * 60)
predict('Sunny', 'Normal', 'Strong')
