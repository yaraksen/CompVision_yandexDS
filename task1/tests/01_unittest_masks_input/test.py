from common import assert_ndarray_equal
from numpy import array, dsplit, zeros
from bayer import get_bayer_masks

def test_masks():
    masks = get_bayer_masks(2, 2)
    gt_masks = zeros((2, 2, 3), 'bool')
    gt_masks[..., 0] = array([[0, 1], [0, 0]])
    gt_masks[..., 1] = array([[1, 0], [0, 1]])
    gt_masks[..., 2] = array([[0, 0], [1, 0]])
    assert_ndarray_equal(actual=masks, correct=gt_masks)

def test_masks_2():
    masks = get_bayer_masks(3, 3)
    gt_masks = zeros((3, 3, 3), 'bool')
    gt_masks[..., 0] = array([[0, 1, 0],
                              [0, 0, 0],
                              [0, 1, 0]])
    gt_masks[..., 1] = array([[1, 0, 1],
                              [0, 1, 0],
                              [1, 0, 1]])
    gt_masks[..., 2] = array([[0, 0, 0],
                              [1, 0, 1],
                              [0, 0, 0]])
    assert_ndarray_equal(actual=masks, correct=gt_masks)

def test_masks_3():
    masks = get_bayer_masks(4, 4)
    gt_masks = zeros((4, 4, 3), 'bool')
    gt_masks[..., 0] = array([[0, 1, 0, 1],
                              [0, 0, 0, 0],
                              [0, 1, 0, 1],
                              [0, 0, 0, 0]])
    gt_masks[..., 1] = array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [1, 0, 1, 0],
                              [0, 1, 0, 1]])
    gt_masks[..., 2] = array([[0, 0, 0, 0],
                              [1, 0, 1, 0],
                              [0, 0, 0, 0],
                              [1, 0, 1, 0]])
    assert_ndarray_equal(actual=masks, correct=gt_masks)
