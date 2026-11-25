import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ===================== Вхідні лінгвістичні змінні =====================
# Температура води, умовна шкала [0, 100] (0 - дуже холодна, 100 - дуже гаряча)
temp_universe = np.linspace(0, 100, 201)
temperature = ctrl.Antecedent(temp_universe, 'temperature')
temperature['холодна'] = fuzz.trapmf(temperature.universe, [0, 0, 10, 30])
temperature['прохолодна'] = fuzz.trimf(temperature.universe, [15, 30, 45])
temperature['тепла'] = fuzz.trimf(temperature.universe, [35, 50, 65])
temperature['не дуже гаряча'] = fuzz.trimf(temperature.universe, [55, 70, 85])
temperature['гаряча'] = fuzz.trapmf(temperature.universe, [75, 90, 100, 100])

# Напір (відносна інтенсивність потоку), умовна шкала [0, 10]
pressure_universe = np.linspace(0, 10, 201)
pressure = ctrl.Antecedent(pressure_universe, 'pressure')
pressure['слабкий'] = fuzz.trapmf(pressure.universe, [0, 0, 2, 5])
pressure['не дуже сильний'] = fuzz.trimf(pressure.universe, [3, 5, 7])
pressure['сильний'] = fuzz.trapmf(pressure.universe, [5, 8, 10, 10])

# ===================== Вихідні лінгвістичні змінні =====================
# Кут повороту крана, градуси [-90, 90] (від'ємний = вліво, додатний = вправо)
angle_universe = np.linspace(-90, 90, 361)


def make_angle_output(name):
    out = ctrl.Consequent(angle_universe, name)
    out['великий_вліво'] = fuzz.trapmf(out.universe, [-90, -90, -70, -45])
    out['середній_вліво'] = fuzz.trimf(out.universe, [-60, -40, -15])
    out['невеликий_вліво'] = fuzz.trimf(out.universe, [-25, -12, 0])
    out['нуль'] = fuzz.trimf(out.universe, [-10, 0, 10])
    out['невеликий_вправо'] = fuzz.trimf(out.universe, [0, 12, 25])
    out['середній_вправо'] = fuzz.trimf(out.universe, [15, 40, 60])
    out['великий_вправо'] = fuzz.trapmf(out.universe, [45, 70, 90, 90])
    return out


hot_angle = make_angle_output('hot_angle')
cold_angle = make_angle_output('cold_angle')

# ===================== Візуалізація функцій належності =====================
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for name in ['холодна', 'прохолодна', 'тепла', 'не дуже гаряча', 'гаряча']:
    axes[0, 0].plot(temperature.universe, temperature[name].mf, label=name)
axes[0, 0].set_title("Температура води")
axes[0, 0].legend(fontsize=8)

for name in ['слабкий', 'не дуже сильний', 'сильний']:
    axes[0, 1].plot(pressure.universe, pressure[name].mf, label=name)
axes[0, 1].set_title("Напір води")
axes[0, 1].legend(fontsize=8)

for name in ['великий_вліво', 'середній_вліво', 'невеликий_вліво', 'нуль',
             'невеликий_вправо', 'середній_вправо', 'великий_вправо']:
    axes[1, 0].plot(hot_angle.universe, hot_angle[name].mf, label=name)
axes[1, 0].set_title("Кут повороту крана гарячої води")
axes[1, 0].legend(fontsize=7)

for name in ['великий_вліво', 'середній_вліво', 'невеликий_вліво', 'нуль',
             'невеликий_вправо', 'середній_вправо', 'великий_вправо']:
    axes[1, 1].plot(cold_angle.universe, cold_angle[name].mf, label=name)
axes[1, 1].set_title("Кут повороту крана холодної води")
axes[1, 1].legend(fontsize=7)
plt.tight_layout()
plt.savefig("task1_membership_functions.png", dpi=150)
print("Збережено task1_membership_functions.png")

# ===================== База правил (11 правил з методички) =====================
Z = 'нуль'
rules = [
    # 1. гаряча & сильний -> hot=середній вліво, cold=середній вправо
    ctrl.Rule(temperature['гаряча'] & pressure['сильний'],
              [hot_angle['середній_вліво'], cold_angle['середній_вправо']]),
    # 2. гаряча & не дуже сильний -> cold=середній вправо
    ctrl.Rule(temperature['гаряча'] & pressure['не дуже сильний'],
              [cold_angle['середній_вправо'], hot_angle[Z]]),
    # 3. не дуже гаряча & сильний -> hot=невеликий вліво
    ctrl.Rule(temperature['не дуже гаряча'] & pressure['сильний'],
              [hot_angle['невеликий_вліво'], cold_angle[Z]]),
    # 4. не дуже гаряча & слабкий -> hot=невеликий вправо, cold=невеликий вправо
    ctrl.Rule(temperature['не дуже гаряча'] & pressure['слабкий'],
              [hot_angle['невеликий_вправо'], cold_angle['невеликий_вправо']]),
    # 5. тепла & не дуже сильний -> залишити (0,0)
    ctrl.Rule(temperature['тепла'] & pressure['не дуже сильний'],
              [hot_angle[Z], cold_angle[Z]]),
    # 6. прохолодна & сильний -> hot=середній вправо, cold=середній вліво
    ctrl.Rule(temperature['прохолодна'] & pressure['сильний'],
              [hot_angle['середній_вправо'], cold_angle['середній_вліво']]),
    # 7. прохолодна & не дуже сильний -> hot=середній вправо, cold=невеликий вліво
    ctrl.Rule(temperature['прохолодна'] & pressure['не дуже сильний'],
              [hot_angle['середній_вправо'], cold_angle['невеликий_вліво']]),
    # 8. холодна & слабкий -> hot=великий вправо
    ctrl.Rule(temperature['холодна'] & pressure['слабкий'],
              [hot_angle['великий_вправо'], cold_angle[Z]]),
    # 9. холодна & сильний -> hot=середній вліво, cold=середній вправо
    ctrl.Rule(temperature['холодна'] & pressure['сильний'],
              [hot_angle['середній_вліво'], cold_angle['середній_вправо']]),
    # 10. тепла & сильний -> hot=невеликий вліво, cold=невеликий вліво
    ctrl.Rule(temperature['тепла'] & pressure['сильний'],
              [hot_angle['невеликий_вліво'], cold_angle['невеликий_вліво']]),
    # 11. тепла & слабкий -> hot=невеликий вправо, cold=невеликий вправо
    ctrl.Rule(temperature['тепла'] & pressure['слабкий'],
              [hot_angle['невеликий_вправо'], cold_angle['невеликий_вправо']]),
]

system = ctrl.ControlSystem(rules)

# ===================== Перевірка роботи системи для набору тестових точок =====================
test_points = [
    (95, 9, "дуже гаряча вода, сильний напір"),
    (95, 1, "дуже гаряча вода, слабкий напір"),
    (50, 5, "тепла вода, середній напір"),
    (20, 9, "холодна вода, сильний напір"),
    (5, 1, "дуже холодна вода, слабкий напір"),
    (30, 4, "прохолодна вода, помірний напір"),
]

print("\n=== Перевірка нечіткого виводу для тестових точок ===")
print(f"{'Температура':>12}{'Напір':>8}{'Опис':>35}{'Кут гар.':>12}{'Кут хол.':>12}")
for t, p, desc in test_points:
    sim = ctrl.ControlSystemSimulation(system)
    sim.input['temperature'] = t
    sim.input['pressure'] = p
    try:
        sim.compute()
        hot_v = f"{sim.output['hot_angle']:>12.2f}"
        cold_v = f"{sim.output['cold_angle']:>12.2f}"
    except KeyError:
        hot_v = f"{'—':>12}"
        cold_v = f"{'—':>12}"
    print(f"{t:>12}{p:>8}{desc:>35}{hot_v}{cold_v}")
