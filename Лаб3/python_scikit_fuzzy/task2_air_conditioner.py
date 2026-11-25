import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ===================== Вхідні лінгвістичні змінні =====================
# Температура повітря в приміщенні, °C
temp_universe = np.linspace(10, 35, 251)
temperature = ctrl.Antecedent(temp_universe, 'temperature')
temperature['дуже холодна'] = fuzz.trapmf(temperature.universe, [10, 10, 14, 18])
temperature['холодна'] = fuzz.trimf(temperature.universe, [15, 19, 23])
temperature['в нормі'] = fuzz.trimf(temperature.universe, [20, 23, 26])
temperature['тепла'] = fuzz.trimf(temperature.universe, [24, 27, 30])
temperature['дуже тепла'] = fuzz.trapmf(temperature.universe, [27, 31, 35, 35])

# Швидкість зміни температури, °C/хв
rate_universe = np.linspace(-2, 2, 201)
rate = ctrl.Antecedent(rate_universe, 'rate')
rate['від’ємна'] = fuzz.trapmf(rate.universe, [-2, -2, -0.6, -0.1])
rate['дорівнює 0'] = fuzz.trimf(rate.universe, [-0.3, 0, 0.3])
rate['додатня'] = fuzz.trapmf(rate.universe, [0.1, 0.6, 2, 2])

# ===================== Вихідна лінгвістична змінна =====================
# Кут повороту регулятора кондиціонера, градуси [-90, 90]
# (від'ємний - вліво - режим "холод"; додатний - вправо - режим "тепло"; 0 - вимкнено)
angle_universe = np.linspace(-90, 90, 361)
regulator = ctrl.Consequent(angle_universe, 'regulator')
regulator['великий_вліво'] = fuzz.trapmf(regulator.universe, [-90, -90, -70, -45])
regulator['невеликий_вліво'] = fuzz.trimf(regulator.universe, [-30, -12, 0])
regulator['нуль'] = fuzz.trimf(regulator.universe, [-10, 0, 10])
regulator['невеликий_вправо'] = fuzz.trimf(regulator.universe, [0, 12, 30])
regulator['великий_вправо'] = fuzz.trapmf(regulator.universe, [45, 70, 90, 90])

# ===================== Візуалізація функцій належності =====================
fig, axes = plt.subplots(3, 1, figsize=(8, 10))
for name in ['дуже холодна', 'холодна', 'в нормі', 'тепла', 'дуже тепла']:
    axes[0].plot(temperature.universe, temperature[name].mf, label=name)
axes[0].set_title("Температура повітря в приміщенні, °C")
axes[0].legend(fontsize=8)

for name in ['від’ємна', 'дорівнює 0', 'додатня']:
    axes[1].plot(rate.universe, rate[name].mf, label=name)
axes[1].set_title("Швидкість зміни температури, °C/хв")
axes[1].legend(fontsize=8)

for name in ['великий_вліво', 'невеликий_вліво', 'нуль', 'невеликий_вправо', 'великий_вправо']:
    axes[2].plot(regulator.universe, regulator[name].mf, label=name)
axes[2].set_title("Кут повороту регулятора кондиціонера")
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig("task2_membership_functions.png", dpi=150)
print("Збережено task2_membership_functions.png")

# ===================== База правил (15 правил) =====================
# у правилі 7 вихідне завдання містило друкарську помилку (мало бути "вправо"
# для режиму "тепло", як і в решті правил) - тут вже виправлено
rules = [
    ctrl.Rule(temperature['дуже тепла'] & rate['додатня'], regulator['великий_вліво']),       # 1
    ctrl.Rule(temperature['дуже тепла'] & rate['від’ємна'], regulator['невеликий_вліво']),     # 2
    ctrl.Rule(temperature['тепла'] & rate['додатня'], regulator['великий_вліво']),             # 3
    ctrl.Rule(temperature['тепла'] & rate['від’ємна'], regulator['нуль']),                     # 4 (режим "холод" вимкнути)
    ctrl.Rule(temperature['дуже холодна'] & rate['від’ємна'], regulator['великий_вправо']),    # 5
    ctrl.Rule(temperature['дуже холодна'] & rate['додатня'], regulator['невеликий_вправо']),   # 6
    ctrl.Rule(temperature['холодна'] & rate['від’ємна'], regulator['великий_вправо']),         # 7 (виправлено)
    ctrl.Rule(temperature['холодна'] & rate['додатня'], regulator['нуль']),                    # 8 (кондиціонер вимкнути)
    ctrl.Rule(temperature['дуже тепла'] & rate['дорівнює 0'], regulator['великий_вліво']),     # 9
    ctrl.Rule(temperature['тепла'] & rate['дорівнює 0'], regulator['невеликий_вліво']),        # 10
    ctrl.Rule(temperature['дуже холодна'] & rate['дорівнює 0'], regulator['великий_вправо']),  # 11
    ctrl.Rule(temperature['холодна'] & rate['дорівнює 0'], regulator['невеликий_вправо']),     # 12
    ctrl.Rule(temperature['в нормі'] & rate['додатня'], regulator['невеликий_вліво']),         # 13
    ctrl.Rule(temperature['в нормі'] & rate['від’ємна'], regulator['невеликий_вправо']),       # 14
    ctrl.Rule(temperature['в нормі'] & rate['дорівнює 0'], regulator['нуль']),                 # 15 (вимкнути)
]

system = ctrl.ControlSystem(rules)

# ===================== Перевірка роботи системи для набору тестових точок =====================
test_points = [
    (33, 1.0, "дуже тепло, температура зростає"),
    (33, -1.0, "дуже тепло, температура спадає"),
    (12, -1.0, "дуже холодно, температура спадає"),
    (12, 1.0, "дуже холодно, температура зростає"),
    (23, 0.0, "кімнатна температура в нормі, стабільно"),
    (28, 0.8, "тепло, температура зростає"),
    (18, -0.8, "холодно, температура спадає"),
]

print("\n=== Перевірка нечіткого виводу для тестових точок ===")
print(f"{'t, °C':>8}{'dt/dt':>8}{'Опис':>40}{'Кут регулятора':>18}")
for t, r, desc in test_points:
    sim = ctrl.ControlSystemSimulation(system)
    sim.input['temperature'] = t
    sim.input['rate'] = r
    sim.compute()
    print(f"{t:>8}{r:>8}{desc:>40}{sim.output['regulator']:>18.2f}")
