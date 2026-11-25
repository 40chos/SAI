import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ===================== Визначення вхідних/вихідної змінних =====================
x1_universe = np.linspace(0, 4, 201)
x2_universe = np.linspace(0, 4, 201)
y_universe = np.linspace(-10, 10, 401)

x1 = ctrl.Antecedent(x1_universe, 'x1')
x2 = ctrl.Antecedent(x2_universe, 'x2')
y = ctrl.Consequent(y_universe, 'y')

# x1: 3 трикутні терми (як у Кроці 9 методички, стандартне рівномірне розбиття)
x1['L'] = fuzz.trimf(x1.universe, [-2, 0, 2])
x1['A'] = fuzz.trimf(x1.universe, [0, 2, 4])
x1['H'] = fuzz.trimf(x1.universe, [2, 4, 6])

# x2: 5 гаусових термів (як у Кроці 11 методички)
centers_x2 = np.linspace(0, 4, 5)
sigma_x2 = 0.4247  # типове стандартне відхилення MATLAB для 5 рівномірних gaussmf
names_x2 = ['L', 'LA', 'A', 'HA', 'H']
for name, c in zip(names_x2, centers_x2):
    x2[name] = fuzz.gaussmf(x2.universe, c, sigma_x2)

# y: 5 трикутних термів (як у Кроці 13 методички)
names_y = ['L', 'LA', 'A', 'HA', 'H']
centers_y = np.linspace(-10, 10, 5)
width_y = 5
for name, c in zip(names_y, centers_y):
    y[name] = fuzz.trimf(y.universe, [c - width_y, c, c + width_y])

# ===================== Візуалізація функцій належності =====================
fig, axes = plt.subplots(3, 1, figsize=(8, 10))
for name in ['L', 'A', 'H']:
    axes[0].plot(x1.universe, x1[name].mf, label=name)
axes[0].set_title("Функції належності змінної x1 (3 трикутних терми)")
axes[0].legend()

for name in names_x2:
    axes[1].plot(x2.universe, x2[name].mf, label=name)
axes[1].set_title("Функції належності змінної x2 (5 гаусових термів)")
axes[1].legend()

for name in names_y:
    axes[2].plot(y.universe, y[name].mf, label=name)
axes[2].set_title("Функції належності вихідної змінної y (5 трикутних термів)")
axes[2].legend()
plt.tight_layout()
plt.savefig("fig2_membership_functions.png", dpi=150)
print("Збережено fig2_membership_functions.png")

# ===================== Формування бази правил (9 правил) =====================
rules = [
    ctrl.Rule(x1['L'] & x2['L'], y['L']),       # 1
    ctrl.Rule(x1['L'] & x2['H'], y['A']),       # 2
    ctrl.Rule(x1['L'] & x2['HA'], y['H']),      # 3
    ctrl.Rule(x1['H'] & x2['L'], y['HA']),      # 4
    ctrl.Rule(x1['H'] & x2['H'], y['L']),       # 5
    ctrl.Rule(x1['A'] & x2['A'], y['A']),       # 6
    ctrl.Rule(x1['A'] & x2['HA'], y['HA']),     # 7
    ctrl.Rule(x1['L'] & x2['LA'], y['LA']),     # 8
    ctrl.Rule(x1['A'] & x2['HA'], y['A']),      # 9 (дублює антецедент правила 7, інший консеквент)
]

system = ctrl.ControlSystem(rules)
sim = ctrl.ControlSystemSimulation(system)

# ===================== Перевірка нечіткого виводу для x1=2, x2=2 (як на Рис.6) =====================
sim.input['x1'] = 2
sim.input['x2'] = 2
sim.compute()
print(f"\nПеревірка (аналог Рис. 6 методички): x1=2, x2=2 -> y = {sim.output['y']:.4f}")

# ===================== Побудова поверхні «вхід-вихід» (аналог Рис.7) =====================
n_grid = 15
x1_grid = np.linspace(0, 4, n_grid)
x2_grid = np.linspace(0, 4, n_grid)
Y_fis = np.zeros((n_grid, n_grid))

for i, xv1 in enumerate(x1_grid):
    for j, xv2 in enumerate(x2_grid):
        s = ctrl.ControlSystemSimulation(system)
        s.input['x1'] = xv1
        s.input['x2'] = xv2
        try:
            s.compute()
            Y_fis[j, i] = s.output['y']
        except Exception:
            Y_fis[j, i] = np.nan

X1g, X2g = np.meshgrid(x1_grid, x2_grid)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1g, X2g, Y_fis, cmap='jet')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Поверхня системи нечіткого виводу (аналог Surface Viewer)')
plt.tight_layout()
plt.savefig("fig3_fis_surface.png", dpi=150)
print("Збережено fig3_fis_surface.png")
