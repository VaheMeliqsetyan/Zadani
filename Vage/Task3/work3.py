import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x1, x2):
    numerator = np.sin(x1**2 - x2**2)**2 - 0.5
    denominator = (1 + 0.001 * (x1**2 + x2**2))**2
    return 0.5 + numerator / denominator

x1_min, x1_max = -2.0, 2.0
x2_min, x2_max = -2.0, 2.0
x10, x20 = 0.0, 0.0  

x1 = np.linspace(x1_min, x1_max, 100)
x2 = np.linspace(x2_min, x2_max, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f(X1, X2)

y_test = f(x10, x20)

fig = plt.figure(figsize=(16, 12))
fig.suptitle(f'Графики функции: f(x₁, x₂) = 0.5 + [sin²(x₁² - x₂²) - 0.5] / (1 + 0.001(x₁² + x₂²))²\n'
             f'Тестовая точка: (x₁₀={x10}, x₂₀={x20}), f(x₁₀,x₂₀)={y_test:.3f}', 
             fontsize=14, y=1.02)

ax1 = fig.add_subplot(221, projection='3d')
surf1 = ax1.plot_surface(X1, X2, Y, cmap='plasma', rstride=1, cstride=1, alpha=0.8)
ax1.set_title('1. 3D поверхность (изометрический вид)', pad=12)
ax1.set_xlabel('x₁', labelpad=10)
ax1.set_ylabel('x₂', labelpad=10)
ax1.set_zlabel('y = f(x₁, x₂)', labelpad=10)
ax1.scatter([x10], [x20], [y_test], color='r', s=50, label='Тестовая точка')
ax1.legend()
fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, label='Значение функции')

ax2 = fig.add_subplot(222, projection='3d')
surf2 = ax2.plot_surface(X1, X2, Y, cmap='plasma', rstride=1, cstride=1, alpha=0.8)
ax2.view_init(elev=90, azim=0)
ax2.set_title('2. 3D поверхность (вид сверху)', pad=12)
ax2.set_xlabel('x₁', labelpad=10)
ax2.set_ylabel('x₂', labelpad=10)
ax2.set_zlabel('y = f(x₁, x₂)', labelpad=10)
ax2.scatter([x10], [x20], [y_test], color='r', s=50)
fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, label='Значение функции')

ax3 = fig.add_subplot(223)
y_x1 = f(x1, x20)
ax3.plot(x1, y_x1, linewidth=2, color='blue')
ax3.set_title(f'3. График y = f(x₁) при x₂ = {x20}', pad=12)
ax3.set_xlabel('x₁', labelpad=10)
ax3.set_ylabel('y = f(x₁)', labelpad=10)
ax3.grid(True, linestyle='--', alpha=0.6)
ax3.axvline(x=x10, color='r', linestyle='--', alpha=0.7)
ax3.axhline(y=y_test, color='r', linestyle='--', alpha=0.7)
ax3.scatter(x10, y_test, color='r', s=80, zorder=5)

ax4 = fig.add_subplot(224)
y_x2 = f(x10, x2)
ax4.plot(x2, y_x2, linewidth=2, color='green')
ax4.set_title(f'4. График y = f(x₂) при x₁ = {x10}', pad=12)
ax4.set_xlabel('x₂', labelpad=10)
ax4.set_ylabel('y = f(x₂)', labelpad=10)
ax4.grid(True, linestyle='--', alpha=0.6)
ax4.axvline(x=x20, color='r', linestyle='--', alpha=0.7)
ax4.axhline(y=y_test, color='r', linestyle='--', alpha=0.7)
ax4.scatter(x20, y_test, color='r', s=80, zorder=5)

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
