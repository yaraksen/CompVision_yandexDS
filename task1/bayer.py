import numpy as np
import math


def get_bayer_masks(n_rows, n_cols):
    red = np.array([[0, 1], [0, 0]])
    red = np.tile(red, (math.ceil(n_rows / 2), math.ceil(n_cols / 2)))[:n_rows, :n_cols]
    green = np.array([[1, 0], [0, 1]])
    green = np.tile(green, (math.ceil(n_rows / 2), math.ceil(n_cols / 2)))[:n_rows, :n_cols]
    blue = np.array([[0, 0], [1, 0]])
    blue = np.tile(blue, (math.ceil(n_rows / 2), math.ceil(n_cols / 2)))[:n_rows, :n_cols]
    return np.dstack((red, green, blue)).astype('bool')


def get_colored_img(raw_img):
    return np.dstack((raw_img, raw_img, raw_img)) * get_bayer_masks(raw_img.shape[0], raw_img.shape[1])


def bilinear_interpolation(colored_img):
    colored_img = colored_img.astype(np.float64)
    height = colored_img.shape[0]
    width = colored_img.shape[1]
    red = colored_img[..., 0]
    green = colored_img[..., 1]
    blue = colored_img[..., 2]
    # red
    for m in range(0, height, 2):  # rows
        for n in range(1, width, 2):  # cols
            # red[m][n] = red[m][n]
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    if x == 0 and y == 0:
                        continue
                    if 0 <= m + x < height and 0 <= n + y < width:
                        if (m + x + n + y) % 2 == 0:
                            red[m + x][n + y] += red[m][n] / 2
                        else:
                            red[m + x][n + y] += red[m][n] / 4
    # green
    for m in range(height):  # rows
        for n in range(m % 2, width, 2):  # cols
            # green[m][n] = green[m][n]
            for x, y in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                if 0 <= m + x < height and 0 <= n + y < width:
                    # print(f"base_cords: ({m},{n}) ---- dot_cords: ({m + x},{n + y})")
                    # print(green)
                    green[m + x][n + y] += green[m][n] / 4
    # blue
    for m in range(1, height, 2):  # rows
        for n in range(0, width, 2):  # cols
            # blue[m][n] = blue[m][n]
            for x in [-1, 0, 1]:
                for y in [-1, 0, 1]:
                    if x == 0 and y == 0:
                        continue
                    if 0 <= m + x < height and 0 <= n + y < width:
                        if (m + x + n + y) % 2 == 0:
                            blue[m + x][n + y] += blue[m][n] / 2
                        else:
                            blue[m + x][n + y] += blue[m][n] / 4
    return np.dstack((red, green, blue)).astype(np.uint8)


def improved_interpolation(raw_img):



def mse(I1, I2):
    tmp = ((I1 - I2)**2).sum(axis=None) / (I1.shape[0] * I1.shape[1] * I1.shape[2])
    if tmp:
        return tmp
    else:
        raise ValueError


def compute_psnr(img_pred, img_gt):
    img_pred = img_pred.astype('float64')
    img_gt = img_gt.astype('float64')
    return 10 * math.log(np.amax(img_gt**2) / mse(img_pred, img_gt), 10)

