from bayer import bilinear_interpolation, get_colored_img
from common import assert_ndarray_equal
from numpy import array, zeros
from skimage import img_as_ubyte

def test_bilinear_interpolation():
    raw_img = array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='uint8')
    colored_img = get_colored_img(raw_img)

    img = img_as_ubyte(bilinear_interpolation(colored_img))
    assert (img[1, 1] == [5, 5, 5]).all()

def test_bilinear_interpolation_2():
    raw_img = array([[202, 150, 137, 121, 195],
                     [ 94, 113, 217,  68, 248],
                     [208, 170, 109,  67,  22],
                     [ 20,  93, 222,  54,  50],
                     [254, 252,  10, 187, 203]], dtype='uint8')
    colored_img = get_colored_img(raw_img)
    gt_img = zeros((5, 5, 3), 'uint8')
    r = slice(1, -1), slice(1, -1)
    gt_img[r + (0,)] = array([[160, 127,  94],
                              [170, 118,  67],
                              [211, 169, 127]])
    gt_img[r + (1,)] = array([[113, 106,  68],
                              [130, 109,  63],
                              [93,  66,  54]])
    gt_img[r + (2,)] = array([[155, 217, 232],
                              [138, 219, 184],
                              [121, 222, 136]])

    img = img_as_ubyte(bilinear_interpolation(colored_img))
    assert_ndarray_equal(actual=img[r],
                         correct=gt_img[r], atol=1)
