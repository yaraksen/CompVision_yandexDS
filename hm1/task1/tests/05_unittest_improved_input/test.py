from bayer import improved_interpolation
from common import assert_ndarray_equal
from numpy import array, zeros, zeros_like
from skimage import img_as_ubyte

def test_improved_interpolation():
    raw_img = array([[8, 5, 3, 7, 1, 3],
                     [5, 2, 6, 8, 8, 1],
                     [9, 9, 8, 1, 6, 4],
                     [9, 4, 2, 3, 6, 8],
                     [5, 4, 3, 2, 8, 7],
                     [7, 3, 3, 6, 9, 3]], dtype='uint8')

    gt_img = zeros((6, 6, 3), 'uint8')
    r = slice(2, -2), slice(2, -2)
    gt_img[r + (0,)] = array([[6, 1],
                              [1, 0]])
    gt_img[r + (1,)] = array([[8, 4],
                              [2, 3]])
    gt_img[r + (2,)] = array([[7, 2],
                              [2, 2]])
    img = img_as_ubyte(improved_interpolation(raw_img))
    assert_ndarray_equal(actual=img[r],
                         correct=gt_img[r], atol=1)

def test_improved_interpolation_2():
    raw_img = array([[153, 180,  37, 114,  16,   8,   6],
                     [225,  18,  91, 196, 117,  70, 115],
                     [ 61, 214,  56,  74, 196, 248,   6],
                     [179,  63,   5, 137, 246,  55, 109],
                     [ 80, 223, 248,  72,  85,  97, 173],
                     [203, 201,  58, 199, 102, 191, 131],
                     [181, 168, 198, 173, 208, 253, 161]], dtype='uint8')
    gt_img = zeros((7, 7, 3), 'uint8')
    r = slice(2, -2), slice(2, -2)
    gt_img[r + (0,)] = array([[112,  74, 224],
                              [ 41,  43, 244],
                              [222,  72,  37]])
    gt_img[r + (1,)] = array([[ 56, 102, 196],
                              [ 56, 137, 199],
                              [248, 132,  85]])
    gt_img[r + (2,)] = array([[ 11,  48, 238],
                              [  5, 147, 246],
                              [ 90,  50, 130]])
    img = img_as_ubyte(improved_interpolation(raw_img))
    assert_ndarray_equal(actual=img[r],
                         correct=gt_img[r], atol=1)
