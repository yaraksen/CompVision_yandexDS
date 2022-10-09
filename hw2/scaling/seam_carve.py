import numpy as np
from scipy.signal import convolve2d
from skimage.io import imread, imsave, imshow


def get_y(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b


def delete_seam(img, mask, is_hor):
    if not is_hor:
        return img.T[~mask.T.astype('bool')].reshape(img.shape[1], img.shape[0] - 1).T
    else:
        return img[~mask.astype('bool')].reshape(img.shape[0], img.shape[1] - 1)


def add_vert_seam(img_, seam_mask_):
    img = np.pad(img_, ((0, 0), (0, 1)), 'edge')
    seam_mask = np.pad(seam_mask_, ((0, 0), (0, 1)), 'constant')
    new_seam_mask = np.roll(seam_mask, 1, axis=1)
    new_seam = np.sum((seam_mask + new_seam_mask) * img, axis=1) / 2
    img = np.insert(img.ravel(), np.flatnonzero(new_seam_mask), new_seam).reshape(img.shape[0], -1)[:, :-1]
    return img


def add_seam(img, mask, is_hor):
    if is_hor:
        return add_vert_seam(img, mask)
    else:
        return add_vert_seam(img.T, mask.T).T


def get_grad(img_y):
    return np.sqrt(convolve2d(img_y, np.array([[0, 0, 0],
                                               [1, 0, -1],
                                               [0, 0, 0]]), mode='same')[1:-1, 1:-1]**2 +
                   convolve2d(img_y, np.array([[0, 1, 0],
                                               [0, 0, 0],
                                               [0, -1, 0]]), mode='same')[1:-1, 1:-1]**2)


def get_grad_sums(gradient):
    grad_sums = gradient.copy()
    for m in range(1, grad_sums.shape[0]):
        for n in range(grad_sums.shape[1]):
            if n - 1 < 0:
                grad_sums[m][n] += min(grad_sums[m - 1][n], grad_sums[m - 1][n + 1])
            elif n + 1 >= grad_sums.shape[1]:
                grad_sums[m][n] += min(grad_sums[m - 1][n - 1], grad_sums[m - 1][n])
            else:
                grad_sums[m][n] += min(grad_sums[m - 1][n - 1],
                                       grad_sums[m - 1][n], grad_sums[m - 1][n + 1])
    return grad_sums


def get_seam_mask(img_y, mask):
    grad = get_grad(np.pad(img_y, 1, 'edge'))
    grad += mask.astype(np.float64) * grad.shape[0] * grad.shape[1] * 256
    grad_sums = get_grad_sums(grad)

    seam_mask = np.zeros_like(img_y)
    ind = np.argmin(grad_sums[-1])
    seam_mask[grad_sums.shape[0] - 1, ind] = 1
    for row in range(grad_sums.shape[0] - 2, -1, -1):
        if ind - 1 < 0:
            ind = np.argmin(grad_sums[row][ind: ind + 2])
            seam_mask[row, ind] = 1
        elif ind + 1 >= grad_sums.shape[1]:
            ind = np.argmin(grad_sums[row][ind - 1: ind + 1]) + ind - 1
            seam_mask[row, ind] = 1
        else:
            ind = np.argmin(grad_sums[row][ind - 1: ind + 2]) + ind - 1
            seam_mask[row, ind] = 1
    return seam_mask


def get_seam(img, mask, is_hor):
    if is_hor:
        return get_seam_mask(get_y(img[..., 0], img[..., 1], img[..., 2]), mask)
    else:
        return get_seam_mask(get_y(img[..., 0], img[..., 1], img[..., 2]).T, mask.T).T


def seam_carve(img, regime=None, mask=None):
    img = img.astype(np.float64)
    if mask is None:
        mask = np.zeros_like(img[..., 0])

    is_hor = False
    if regime.split()[0] == 'horizontal':
        is_hor = True

    seam_mask = get_seam(img, mask, is_hor)
    if regime.split()[1] == 'expand':
        return np.dstack((
            add_seam(img[..., 0], seam_mask, is_hor),
            add_seam(img[..., 1], seam_mask, is_hor),
            add_seam(img[..., 2], seam_mask, is_hor)
        )), add_seam(mask, seam_mask, is_hor), seam_mask
    else:
        return np.dstack((
            delete_seam(img[..., 0], seam_mask, is_hor),
            delete_seam(img[..., 1], seam_mask, is_hor),
            delete_seam(img[..., 2], seam_mask, is_hor)
        )), delete_seam(mask, seam_mask, is_hor), seam_mask
