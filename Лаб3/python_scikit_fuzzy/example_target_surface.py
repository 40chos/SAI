import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Побудова графіка функції y=(x1^2-8)*cos(x2) в області x1,x2 є [0,4]
n = 15
x1 = np.linspace(0, 4, n)
x2 = np.linspace(0, 4, n)
X1, X2 = np.meshgrid(x1, x2)
Y = (X1 ** 2 - 8) * np.cos(X2)

fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, Y, cmap='jet')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Target: y=(x1^2-8)*cos(x2)')
plt.tight_layout()
plt.savefig("fig1_target_surface.png", dpi=150)
print("Збережено fig1_target_surface.png")
