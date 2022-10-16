import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio


def pca_compression(matrix, p):
    """ Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """
    mean_by_row = matrix.mean(axis=1)
    matrix_centered = matrix - mean_by_row[:, None]
    cov = np.cov(matrix_centered)
    eigen_vals, eigen_vecs = np.linalg.eigh(cov)

    sort = (-eigen_vals).argsort()[:p]
    eigen_vecs = eigen_vecs[:, sort]
    proj = eigen_vecs.T @ matrix_centered
    return eigen_vecs, proj, mean_by_row


def pca_decompression(compressed):
    """ Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    img = []
    for i, comp in enumerate(compressed):
        img.append(np.clip(comp[0] @ comp[1] + comp[2][:, None], 0, 255))
    return np.dstack(img).astype(np.uint8)


def pca_visualize():
    plt.clf()
    img = imread('cat.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            compressed.append(pca_compression(img[..., j], p))

        axes[i // 3, i % 3].imshow(pca_decompression(compressed))
        axes[i // 3, i % 3].set_title('Компонент: {}'.format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """ Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    C_b = 128 - 0.1687 * R - 0.3313 * G + 0.5 * B
    C_r = 128 + 0.5 * R - 0.4187 * G - 0.0813 * B
    return np.dstack((Y, C_b, C_r))


def ycbcr2rgb(img):
    """ Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """
    Y, C_b, C_r = img[..., 0], img[..., 1], img[..., 2]
    C_r -= 128
    C_b -= 128
    R = Y + 1.402 * C_r
    G = Y - 0.34414 * C_b - 0.71414 * C_r
    B = Y + 1.77 * C_b
    return np.dstack((R, G, B))


def get_gauss_1():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    YCbCr_img = rgb2ycbcr(rgb_img)
    Y, C_b, C_r = YCbCr_img[..., 0], YCbCr_img[..., 1], YCbCr_img[..., 2]
    YCbCr_img_filtered = np.dstack((Y,
                                    gaussian_filter(C_b, 10),
                                    gaussian_filter(C_r, 10)))
    plt.imshow(np.clip(ycbcr2rgb(YCbCr_img_filtered), 0, 255).astype(np.uint8))
    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread('Lenna.png')
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    YCbCr_img = rgb2ycbcr(rgb_img)
    Y, C_b, C_r = YCbCr_img[..., 0], YCbCr_img[..., 1], YCbCr_img[..., 2]
    YCbCr_img_filtered = np.dstack((gaussian_filter(Y, 10),
                                    C_b,
                                    C_r))
    plt.imshow(np.clip(ycbcr2rgb(YCbCr_img_filtered), 0, 255).astype(np.uint8))
    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [A // 2, B // 2, 1]
    """
    filtered_component = gaussian_filter(component, 10)
    return filtered_component[::2, ::2]


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """
    block_float = block.astype(np.float64)
    alpha = np.ones_like(block_float)
    alpha[:, 0] = 1.0 / np.sqrt(2)
    by_x_y = ((2 * np.arange(8) + 1) * np.pi / 16).reshape(-1, 1)
    by_u_v = np.arange(8).reshape(1, -1)
    cos = np.cos(by_x_y @ by_u_v) / 2 * alpha
    return cos.T @ block_float @ cos


# Матрица квантования яркости
y_quantization_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Матрица квантования цвета
color_quantization_matrix = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """
    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100
    if 1 <= q < 50:
        s = 5000 / q
    elif 50 <= q <= 99:
        s = 200 - 2 * q
    else:
        s = 1

    new_quantization_matrix = (default_quantization_matrix.astype(np.float64) * s + 50) / 100
    new_quantization_matrix = np.clip(new_quantization_matrix, 0, 255).astype(np.uint8)
    new_quantization_matrix[new_quantization_matrix == 0] = np.ones_like(new_quantization_matrix[new_quantization_matrix == 0])
    return new_quantization_matrix


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """
    return list(np.concatenate([block[::-1].diagonal(i)[::int((-1)**abs(i + 1))] for i in range(-7, 8)]))


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """
    compressed_list = []
    i = 0
    while i < len(zigzag_list):
        compressed_list.append(zigzag_list[i])
        if zigzag_list[i] == 0:
            counter = 1
            i += 1
            while i < len(zigzag_list) and zigzag_list[i] == 0:
                counter += 1
                i += 1
            compressed_list.append(counter)
            continue
        i += 1
    return compressed_list


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """
    # Переходим из RGB в YCbCr
    # Уменьшаем цветовые компоненты
    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    # Применяем ДКП, квантование, зигзаг-сканирование и сжатие
    YCbCr_img = rgb2ycbcr(img)
    Y = YCbCr_img[..., 0]
    C_b = downsampling(YCbCr_img[..., 1])
    C_r = downsampling(YCbCr_img[..., 2])

    compressed_img = []
    is_y = 0
    for c in (Y, C_b, C_r):
        channel_blocks = []
        for m in range(0, c.shape[0] - 7, 8):
            for n in range(0, c.shape[1] - 7, 8):
                channel_blocks.append(compression(zigzag(quantization(dct(c[m:m + 8, n:n + 8] - 128), quantization_matrixes[0 + is_y]))))
        compressed_img.append(channel_blocks)
        is_y = 1
    return compressed_img


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """
    uncompressed_list = []
    i = 0
    while i < len(compressed_list):
        if compressed_list[i] == 0:
            i += 1
            uncompressed_list += [0] * compressed_list[i]
        else:
            uncompressed_list.append(compressed_list[i])
        i += 1
    return uncompressed_list


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """
    original_block = np.empty((8, 8))
    # checker = np.zeros((8, 8))
    idx = 0
    row = 0
    col = 0
    after_diag = False
    while idx < 64:
        # up diagonal
        while row >= 0 and col < 8:
            original_block[row][col] = input[idx]
            # checker[row][col] += 1
            idx += 1
            row -= 1
            col += 1
        if after_diag:
            col -= 1
            row += 2
        else:
            row += 1

        # down diagonal
        while col >= 0 and row < 8:
            original_block[row][col] = input[idx]
            # checker[row][col] += 1
            idx += 1
            row += 1
            col -= 1
        if row == 8 and col == -1:
            after_diag = True
        if after_diag:
            col += 2
            row -= 1
        else:
            col += 1
    return original_block


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """
    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """
    block_float = block.astype(np.float64)
    alpha = np.ones_like(block_float)
    alpha[:, 0] = 1.0 / np.sqrt(2)
    by_x_y = ((2 * np.arange(8) + 1) * np.pi / 16).reshape(-1, 1)
    by_u_v = np.arange(8).reshape(1, -1)
    cos = np.cos(by_x_y @ by_u_v) / 2 * alpha
    return np.round(cos @ block_float @ cos.T)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """
    return np.repeat(np.repeat(component, repeats=2, axis=1), repeats=2, axis=0)


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """
    Y = np.empty((result_shape[0], result_shape[1]))
    C_b = np.empty((result_shape[0] // 2, result_shape[1] // 2))
    C_r = np.empty((result_shape[0] // 2, result_shape[1] // 2))

    is_y = 0
    channel = 0
    for c in (Y, C_b, C_r):
        block_num = 0
        for m in range(0, c.shape[0] - 7, 8):
            for n in range(0, c.shape[1] - 7, 8):
                c[m:m + 8, n:n + 8] = inverse_dct(inverse_quantization(inverse_zigzag(inverse_compression(result[channel][block_num])), quantization_matrixes[0 + is_y])) + 128
                block_num += 1
        is_y = 1
        channel += 1
    return np.clip(ycbcr2rgb(np.dstack((Y, upsampling(C_b), upsampling(C_r)))), 0, 255).astype(np.uint8)


def jpeg_visualize():
    plt.clf()
    img = imread('cutie.jpg')
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        quantization_matrices = (own_quantization_matrix(y_quantization_matrix, p),
                                 own_quantization_matrix(color_quantization_matrix, p))
        compressed_img = jpeg_compression(img, quantization_matrices)
        decompressed_img = jpeg_decompression(compressed_img, img.shape, quantization_matrices)

        axes[i // 3, i % 3].imshow(decompressed_img)
        axes[i // 3, i % 3].set_title('Quality Factor: {}'.format(p))

    fig.savefig("jpeg_visualization.png")


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    if c_type.lower() == 'jpeg':
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
    elif c_type.lower() == 'pca':
        compressed = []
        for j in range(0, 3):
            compressed.append((pca_compression(img[:, :, j].astype(np.float64).copy(), param)))

        img = pca_decompression(compressed)
        compressed.extend([np.mean(img[:, :, 0], axis=1), np.mean(img[:, :, 1], axis=1), np.mean(img[:, :, 2], axis=1)])

    if 'tmp' not in os.listdir() or not os.path.isdir('tmp'):
        os.mkdir('tmp')

    np.savez_compressed(os.path.join('tmp', 'tmp.npz'), compressed)
    size = os.stat(os.path.join('tmp', 'tmp.npz')).st_size * 8
    os.remove(os.path.join('tmp', 'tmp.npz'))

    return img, size / (img.shape[0] * img.shape[1])


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Rate-Distortion для PCA и JPEG. Построение графиков
    Вход: путь до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == 'jpeg' or c_type.lower() == 'pca'

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    rate = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title('PSNR for {}'.format(c_type.upper()))
    ax1.plot(param_list, psnr, 'tab:orange')
    ax1.set_xlabel('Quality Factor')
    ax1.set_ylabel('PSNR')

    ax2.set_title('Rate-Distortion for {}'.format(c_type.upper()))
    ax2.plot(psnr, rate, 'tab:red')
    ax2.set_xlabel('Distortion')
    ax2.set_ylabel('Rate')
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'pca', [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics('Lenna.png', 'jpeg', [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == '__main__':
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
