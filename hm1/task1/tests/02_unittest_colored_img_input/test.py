from bayer import get_colored_img
from common import assert_ndarray_equal
from numpy import array

def test_colored_img():
    raw_img = array([[1, 2], [3, 4]], dtype='uint8')
    gt_colored_img = array([[[0, 1, 0], [2, 0, 0]],
                            [[0, 0, 3], [0, 4, 0]]], dtype='uint8')
    colored_img = get_colored_img(raw_img)
    assert_ndarray_equal(actual=colored_img, correct=gt_colored_img)

def test_colored_img_2():
    raw_img = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='uint8')
    gt_colored_img = array([[[0, 1, 0], [2, 0, 0], [0, 3, 0]],
                            [[0, 0, 4], [0, 5, 0], [0, 0, 6]],
                            [[0, 7, 0], [8, 0, 0], [0, 9, 0]]], dtype='uint8')
    colored_img = get_colored_img(raw_img)
    assert_ndarray_equal(actual=colored_img, correct=gt_colored_img)
