from bayer import improved_interpolation
from common import assert_ndarray_equal
from glob import glob
from os.path import abspath, basename, dirname, join
from skimage import img_as_ubyte
from skimage.io import imread, imsave

def test():
    test_dir = dirname(abspath(__file__))
    for img_filename in sorted(glob(join(test_dir, '[0-9][0-9].png'))):
        raw_img = img_as_ubyte(imread(join(test_dir, img_filename)))
        img = img_as_ubyte(improved_interpolation(raw_img))
        out_filename = join(test_dir, 'gt_' + basename(img_filename))
        gt_img = img_as_ubyte(imread(out_filename))
        r = slice(2, -2), slice(2, -2)
        assert_ndarray_equal(actual=img[r],
                             correct=gt_img[r], atol=1,
                             err_msg=f'Testing on img {img_filename} failed')
