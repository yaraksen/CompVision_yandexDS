import numpy as np
from sklearn.metrics import mean_squared_error
import math
import os


def gaussian_kernel(size, sigma):
    """
    Построение ядра фильтра Гаусса.

    @param  size  int    размер фильтра (нечетный)
    @param  sigma float  параметр размытия
    @return numpy array  фильтр Гаусса размером size x size
    """
    x, y = np.meshgrid(np.arange(-size // 2 + 1, size // 2 + 1),
                       np.arange(-size // 2 + 1, size // 2 + 1))
    r = np.sqrt(x**2 + y**2)
    h = 1 / (2 * np.pi * sigma**2) * np.exp(-r**2 / (2 * sigma**2))
    h /= h.sum()
    return h


def fourier_transform(h, shape):
    """
    Получение Фурье-образа искажающей функции

    @param  h            numpy array  искажающая функция h (ядро свертки)
    @param  shape        list         требуемый размер образа
    @return numpy array  H            Фурье-образ искажающей функции h
    """
    return np.fft.fft2(np.pad(h, ((0, shape[0] - h.shape[0]), (0, shape[1] - h.shape[1]))))

def inverse_kernel(H, threshold=1e-10):
    """
    Получение H_inv

    @param  H            numpy array    Фурье-образ искажающей функции h
    @param  threshold    float          порог отсечения для избежания деления на 0
    @return numpy array  H_inv
    """
    H_inv = H.copy()
    H_inv[np.abs(H) > threshold] = 1 / H_inv[np.abs(H) > threshold]
    H_inv[np.abs(H) <= threshold] = 0
    return H_inv


def inverse_filtering(blurred_img, h, threshold=1e-10):
    """
    Метод инверсной фильтрации

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  threshold      float        параметр получения H_inv
    @return numpy array                 восстановленное изображение
    """
    f = np.fft.ifft2(fourier_transform(blurred_img, blurred_img.shape) * inverse_kernel(fourier_transform(h, blurred_img.shape), threshold))
    return np.abs(f)


def wiener_filtering(blurred_img, h, K=0.0001):
    """
    Винеровская фильтрация

    @param  blurred_img    numpy array  искаженное изображение
    @param  h              numpy array  искажающая функция
    @param  K              float        константа из выражения (8)
    @return numpy array                 восстановленное изображение
    """
    fft_h = fourier_transform(h, blurred_img.shape)
    fft_h_conj = np.conj(fft_h)
    return np.abs(np.fft.ifft2(fft_h_conj / (fft_h * fft_h_conj + K) * fourier_transform(blurred_img, blurred_img.shape)))


def compute_psnr(img1, img2):
    """
    PSNR metric

    @param  img1    numpy array   оригинальное изображение
    @param  img2    numpy array   искаженное изображение
    @return float   PSNR(img1, img2)
    """
    return 20 * math.log10(255 / math.sqrt(mean_squared_error(img1, img2)))


if __name__ == '__main__':
    dirname = os.path.dirname(__file__)
    original_img = np.load(os.path.join(dirname, 'original_img.npy'))
    noisy_img = np.load(os.path.join(dirname, 'noisy_img.npy'))

    kernel = gaussian_kernel(size=15, sigma=5)
    for k in np.linspace(1e-7, 1e-4, 50):
        filtered_img = wiener_filtering(noisy_img, kernel, k)
        print(compute_psnr(filtered_img, original_img) - compute_psnr(noisy_img, original_img))
