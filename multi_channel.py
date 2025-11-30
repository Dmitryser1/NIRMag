import numpy as np
import matplotlib.pyplot as plt

def generate_shifted_mode(Z, shift_x, shift_y):
    """Сдвиг моды через фазовый множитель e^(i(kx x + ky y))"""
    ny, nx = Z.shape
    x = np.linspace(-1, 1, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y)

    kx = shift_x
    ky = shift_y

    return np.abs(np.exp(1j * (kx * X + ky * Y)) * Z)

# ------------------------------
# Генерация базовой моды Z
# ------------------------------
N = 256
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
X, Y = np.meshgrid(x, y)
rho = np.sqrt(X**2 + Y**2)

# Пример — гауссиан + искажение
Z = np.exp(-3 * rho**2) * np.exp(1j * 2 * X * Y)

# ------------------------------
# Создание 4 направлений
# ------------------------------
top_left     = generate_shifted_mode(Z,  -8,  +8)
top_right    = generate_shifted_mode(Z,  +8,  +8)
bottom_left  = generate_shifted_mode(Z,  -8,  -8)
bottom_right = generate_shifted_mode(Z,  +8,  -8)

# ------------------------------
# Построение красивой фигуры
# ------------------------------
fig, ax = plt.subplots(2, 3, figsize=(14, 9))
ax = ax.ravel()

ax[0].imshow(np.abs(Z), cmap='inferno')
ax[0].set_title("Input Mode")
ax[0].axis('off')

ax[1].imshow(top_left, cmap='inferno')
ax[1].set_title("Top Left (mode A)")
ax[1].axis('off')

ax[2].imshow(top_right, cmap='inferno')
ax[2].set_title("Top Right (mode B)")
ax[2].axis('off')

ax[3].imshow(bottom_left, cmap='inferno')
ax[3].set_title("Bottom Left (mode C)")
ax[3].axis('off')

ax[4].imshow(bottom_right, cmap='inferno')
ax[4].set_title("Bottom Right (mode D)")
ax[4].axis('off')

fig.delaxes(ax[5])  # пустая ячейка

plt.tight_layout()
plt.show()
