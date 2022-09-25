import numpy as np
from numpy.fft import fft2, ifft2
from skimage.io import imread, imsave
import scipy.signal as sig


def cross_correlate(i1, i2):
    return ifft2(fft2(i1) * np.conjugate(fft2(i2)))


def get_shift(i1, i2):
    channel_height = i1.shape[0]
    channel_width = i1.shape[1]
    i1 -= np.mean(i1)
    i2 -= np.mean(i2)
    c = cross_correlate(i1, i2)
    # print(c.shape, i1.shape)
    u, v = np.unravel_index(np.argmax(c), c.shape)
    return u - channel_height, v - channel_width


def get_real_shift(i1, i2):  # for testing
    i1 -= np.mean(i1)
    i2 -= np.mean(i2)
    u, v = np.unravel_index(
        np.argmax(sig.fftconvolve(i1, i2[::-1, ::-1], 'same')), i1.shape)
    return u - i1.shape[0] // 2, v - i1.shape[1] // 2


def crop5perc(channel):
    height = channel.shape[0]
    width = channel.shape[1]
    return channel[int(height * 0.05): int(height * 0.95), int(width * 0.05): int(width * 0.95)]


def align(raw_img, green_cords):
    raw_img = raw_img.astype(np.float16)
    channel_height = raw_img.shape[0] // 3
    blue = crop5perc(raw_img[0:channel_height, :])
    green = crop5perc(raw_img[channel_height:channel_height * 2, :])
    red = crop5perc(raw_img[channel_height * 2:channel_height * 3, :])

    u1, v1 = get_real_shift(green, red)
    u2, v2 = get_real_shift(green, blue)

    # u1r, v1r = get_real_shift(green, red)
    # u2r, v2r = get_real_shift(green, blue)
    # print(u1, v1, 'real:', u1r, v1r)
    # print(u2, v2, 'real:', u2r, v2r)

    red = np.roll(red, (u1, v1), axis=(0, 1))
    blue = np.roll(blue, (u2, v2), axis=(0, 1))

    # imsave('red.png', red)
    # imsave('green.png', green)
    # imsave('blue.png', blue)

    red_cords = np.array(green_cords) - (u1 - channel_height, v1)
    blue_cords = np.array(green_cords) - (u2 + channel_height, v2)
    return np.dstack((red, green, blue)), blue_cords, red_cords


if __name__ == '__main__':
    image_num = '00'
    image = imread(f'public_tests/{image_num}_test_img_input/img.png')
    coord = open(
        f'public_tests/{image_num}_test_img_input/g_coord.csv').read().rstrip('\n').split(',')
    g_coord = (int(coord[0]), int(coord[1]))
    aligned, blue_c, red_c = align(image, g_coord)
    # print(blue_c, "____", red_c)
    imsave('aligned.png', aligned)

    with open(f'public_tests/{image_num}_test_img_gt/gt.csv') as fhandle:
        parts = fhandle.read().rstrip('\n').split(',')
        coords = map(int, parts[1:])
        gt_b_row, gt_b_col, _, _, gt_r_row, gt_r_col, diff_max = coords
        print(f'[{gt_b_row}, {gt_b_col}] ==== [{blue_c[0]}, {blue_c[1]}]')
        print(f'[{gt_r_row}, {gt_r_col}] ==== [{red_c[0]}, {red_c[1]}]')
