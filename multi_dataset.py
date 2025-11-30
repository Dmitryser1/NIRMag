import numpy as np
import matplotlib.pyplot as plt
import os
from math import factorial
from scipy.ndimage import rotate

# Создаем директорию для хранения датасета
dataset_dir = 'DataSet_prikol'
os.makedirs(dataset_dir, exist_ok=True)

# ------------------------- ZERNIKE PART -------------------------

def R(n, m, r):
    if (n - m) % 2 != 0:
        return np.zeros_like(r)
    radial_sum = np.zeros_like(r)
    for k in range((n - m)//2 + 1):
        c = (-1)**k * factorial(n-k) / (
            factorial(k) *
            factorial((n + m)//2 - k) *
            factorial((n - m)//2 - k)
        )
        radial_sum += c * r**(n - 2*k)
    return radial_sum


def zernike_polynomial(n, m, rho, theta):
    if m >= 0:
        return R(n, m, rho) * np.cos(m * theta)
    else:
        return R(n, -m, rho) * np.sin(-m * theta)


# ------------------------- UTILS -------------------------

def noise(img, p):
    rows, cols = img.shape
    noise_map = (np.random.rand(rows, cols) < p) * np.random.rand(rows, cols) * (np.max(img)/4)
    return img + noise_map


def crop_center(img, size=128):
    h, w = img.shape
    sx = w//2 - size//2
    sy = h//2 - size//2
    return img[sy:sy+size, sx:sx+size]


def crop_and_stack(channels, size=128):
    return np.stack([crop_center(ch, size) for ch in channels], axis=-1)


def augment_multi(img_channels, count):
    """Аугментации применяются ко всем каналам одинаково"""
    augmented = []
    for _ in range(count):
        angle = np.random.randint(0, 360)

        # вращаем по каналам одинаково
        rot = [rotate(ch, angle=angle, reshape=False) for ch in img_channels]

        # шум только в амплитуду
        noise1 = noise(rot[0], 0.04)
        noise2 = noise(rot[0], 0.08)

        # обновляем каналы
        aug1 = rot.copy()
        aug2 = rot.copy()
        aug1[0] = noise1
        aug2[0] = noise2

        augmented.extend([rot, aug1, aug2])
    return augmented


# ------------------------- MAIN GENERATION -------------------------

def generate_images(n, m, a_range, class_name, grid_size=100, num_aug=30):
    y, x = np.linspace(-1, 1, grid_size), np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    rho = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)

    mask = rho <= 1
    rho[~mask] = 0
    theta[~mask] = 0

    class_dir = os.path.join(dataset_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)

    img_counter = 0
    b = 25

    for a in a_range:
        # ----------------- генерируем Zernike -----------------
        Z = zernike_polynomial(n, m, rho, theta)
        Z_field = np.exp(1j * (a * Z + b * np.sqrt(X**2 + Y**2)))

        F = np.fft.fftshift(np.fft.fft2(Z_field, s=[512, 512]))

        # ----------------- создаём каналы -----------------
        amplitude = np.abs(F)
        phase     = np.angle(F)
        real      = np.real(F)
        imag      = np.imag(F)
        intensity = amplitude ** 2

        base_channels = [amplitude, phase, real, imag, intensity]

        # кропим и сохраняем мультиканал
        stacked = crop_and_stack(base_channels, size=128)

        base_name = f"zernike_n{n}_m{m}_a{a:.2f}_b{b:.2f}"
        np.save(os.path.join(class_dir, base_name + ".npy"), stacked)

        # ----------------- создаем аугментации -----------------
        augmented = augment_multi(base_channels, num_aug)

        for i, aug in enumerate(augmented):
            stacked_aug = crop_and_stack(aug, size=128)
            np.save(os.path.join(class_dir, f"{base_name}_aug{i}.npy"), stacked_aug)

        img_counter += 1
        if img_counter % 10 == 0:
            print(f"{img_counter} базовых изображений обработано")


# ------------------------- CONFIG -------------------------

aberration_classes = [
    (2, 2, 'class1'), 
    (3, 1, 'class2'), 
    (3, 3, 'class3'), 
    (4, 2, 'class4'), 
    (4, 4, 'class5'),
    (5, 1, 'class6'), 
    (5, 3, 'class7'),
    (5, 5, 'class8'),
]

a_range_large = np.linspace(0.1, 2, 20)


# ------------------------- RUN -------------------------
if __name__ == "__main__":
    for n, m, cls in aberration_classes:
        print("class:", cls)
        generate_images(n, m, a_range_large, cls)
