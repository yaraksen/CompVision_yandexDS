from bayer import compute_psnr
from numpy import array
from pytest import approx, raises


def test_exception():
    img_pred = array([[[100, 117,  56],
                       [118, 156, 106]],
                      [[ 93,  13, 201],
                       [206,  15,  29]]], dtype='uint8')
    img_gt = img_pred.copy()

    with raises(ValueError):
        compute_psnr(img_pred, img_gt)


def test_psnr():
    img_pred = array([[[146, 222, 187],
                       [254, 123,  38],
                       [ 57, 255, 135]],
                      [[230, 176, 213],
                       [114,  38, 184],
                       [ 47, 212,  52]],
                      [[100, 111, 170],
                       [ 52, 230, 182],
                       [213,  50, 197]]], dtype='uint8')
    img_gt = array([[[254,  60,   6],
                     [216,  53,  14],
                     [106, 185, 239]],
                    [[121,  34,  29],
                     [ 49, 139, 149],
                     [  6, 159, 221]],
                    [[240,  53, 124],
                     [  3, 194, 227],
                     [ 84,  12, 218]]], dtype='uint8')

    assert compute_psnr(img_pred, img_gt) == approx(8.266646)


def test_psnr_2():
    img_pred = array([[[ 87,  78, 248],
                       [ 92, 239, 239],
                       [119, 239,  48]],
                      [[210, 168, 125],
                       [244, 125, 121],
                       [ 12, 196, 166]],
                      [[ 62,   8,  80],
                       [206, 133, 244],
                       [ 36, 181,  19]]], dtype='uint8')
    img_gt = array([[[103,  87,  58],
                     [ 14, 236,  41],
                     [222,  19,  65]],
                    [[ 13, 167, 106],
                     [ 40,  27,  95],
                     [147, 223,  38]],
                    [[234, 129, 110],
                     [ 35, 225,   0],
                     [ 78, 232,  16]]], dtype='uint8')

    assert compute_psnr(img_pred, img_gt) == approx(5.600017)
