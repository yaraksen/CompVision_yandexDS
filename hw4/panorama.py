import numpy as np
from skimage.feature import ORB, match_descriptors
from skimage.color import rgb2gray
from skimage.transform import ProjectiveTransform
from skimage.transform import warp
from skimage.filters import gaussian

DEFAULT_TRANSFORM = ProjectiveTransform


def find_orb(img, n_keypoints=500):
    """Find keypoints and their descriptors in image.

    img ((W, H, 3)  np.ndarray) : 3-channel image
    n_keypoints (int) : number of keypoints to find

    Returns:
        (N, 2)  np.ndarray : keypoints
        (N, 256)  np.ndarray, type=np.bool  : descriptors
    """
    detector_extractor = ORB(n_keypoints=n_keypoints)
    detector_extractor.detect_and_extract(rgb2gray(img))
    return detector_extractor.keypoints, detector_extractor.descriptors


def center_and_normalize_points(points):
    """Center the image points, such that the new coordinate system has its
    origin at the centroid of the image points.

    Normalize the image points, such that the mean distance from the points
    to the origin of the coordinate system is sqrt(2).

    points ((N, 2) np.ndarray) : the coordinates of the image points

    Returns:
        (3, 3) np.ndarray : the transformation matrix to obtain the new points
        (N, 2) np.ndarray : the transformed image points
    """

    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    matrix = np.zeros((3, 3))
    C_x = points[:, 0].mean()
    C_y = points[:, 1].mean()
    N = np.sqrt(2) / np.sqrt(C_x**2 + C_y**2)
    matrix[0][0] = N
    matrix[1][1] = N
    matrix[2][2] = 1
    matrix[0][2] = -N * C_x
    matrix[1][2] = -N * C_y
    return matrix, (matrix @ pointsh)[:2, :].T


def find_homography(src_keypoints, dest_keypoints):
    """Estimate homography matrix from two sets of N (4+) corresponding points.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates

    Returns:
        ((3, 3) np.ndarray) : homography matrix
    """

    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    A = []
    for s, d in zip(src, dest):
        a_x = [-s[0], -s[1], -1, 0, 0, 0, d[0] * s[0], d[0] * s[1], d[0]]
        a_y = [0, 0, 0, -s[0], -s[1], -1, d[1] * s[0], d[1] * s[1], d[1]]
        A.append(a_x)
        A.append(a_y)
    A = np.array(A).astype(np.float64)
    _, _, VT = np.linalg.svd(A)
    H = VT[-1, :].reshape((3, 3))
    return np.linalg.inv(dest_matrix) @ H @ src_matrix


def ransac_transform(src_keypoints, src_descriptors, dest_keypoints, dest_descriptors, max_trials=1000, residual_threshold=5, return_matches=False):
    """Match keypoints of 2 images and find ProjectiveTransform using RANSAC algorithm.

    src_keypoints ((N, 2) np.ndarray) : source coordinates
    src_descriptors ((N, 256) np.ndarray) : source descriptors
    dest_keypoints ((N, 2) np.ndarray) : destination coordinates
    dest_descriptors ((N, 256) np.ndarray) : destination descriptors
    max_trials (int) : maximum number of iterations for random sample selection.
    residual_threshold (float) : maximum distance for a data point to be classified as an inlier.
    return_matches (bool) : if True function returns matches

    Returns:
        skimage.transform.ProjectiveTransform : transform of source image to destination image
        (Optional)(N, 2) np.ndarray : inliers' indexes of source and destination images
    """
    match = match_descriptors(src_descriptors, dest_descriptors)
    src_keypoints = src_keypoints[match[:, 0]]
    dest_keypoints = dest_keypoints[match[:, 1]]
    best_transform_count = 0
    inliers = []

    for _ in range(max_trials):
        rand = np.random.randint(src_keypoints.shape[0], size=4)
        H = find_homography(src_keypoints[rand], dest_keypoints[rand])
        pred_dest_points = (H @ np.row_stack([src_keypoints.T, np.ones((src_keypoints.shape[0]), )]))
        pred_dest_points /= pred_dest_points[2]
        pred_dest_points = pred_dest_points[:2, :].T
        count = 0
        good_points = []
        for i in range(len(src_keypoints)):
            if np.linalg.norm(dest_keypoints[i] - pred_dest_points[i]) < residual_threshold:
                count += 1
                good_points.append(i)

        if count > best_transform_count:
            best_transform_count = count
            inliers = good_points

    H = find_homography(src_keypoints[inliers], dest_keypoints[inliers])
    if return_matches:
        return ProjectiveTransform(H), match[inliers]
    else:
        return ProjectiveTransform(H)


def find_simple_center_warps(forward_transforms):
    """Find transformations that transform each image to plane of the central image.

    forward_transforms (Tuple[N]) : - pairwise transformations

    Returns:
        Tuple[N + 1] : transformations to the plane of central image
    """
    image_count = len(forward_transforms) + 1
    center_index = (image_count - 1) // 2

    result = [None] * image_count
    result[center_index] = DEFAULT_TRANSFORM()

    for i in range(center_index + 1, image_count):
        result[i] = result[i - 1] + ProjectiveTransform(np.linalg.inv(forward_transforms[i - 1].params))
    for i in range(center_index - 1, -1, -1):
        result[i] = result[i + 1] + forward_transforms[i]

    return tuple(result)


def get_corners(image_collection, center_warps):
    """Get corners' coordinates after transformation."""
    for img, transform in zip(image_collection, center_warps):
        height, width, _ = img.shape
        corners = np.array([[0, 0],
                            [height, 0],
                            [height, width],
                            [0, width]])

        yield transform(corners)[:, ::-1]


def get_min_max_coords(corners):
    """Get minimum and maximum coordinates of corners."""
    corners = np.concatenate(corners)
    return corners.min(axis=0), corners.max(axis=0)


def get_final_center_warps(image_collection, simple_center_warps):
    """Find final transformations.
        image_collection (Tuple[N]) : list of all images
        simple_center_warps (Tuple[N])  : transformations unadjusted for shift

        Returns:
            Tuple[N] : final transformations
    """
    min_coords, max_coords = get_min_max_coords(tuple(get_corners(image_collection, simple_center_warps)))
    shift_matrix = np.eye(3)
    shift_matrix[0][2] = -min_coords[1]
    shift_matrix[1][2] = -min_coords[0]
    transform = ProjectiveTransform(shift_matrix)
    result = tuple(map(lambda w: w + transform, simple_center_warps))
    coords = (int(max_coords[1] - min_coords[1]) + 1, int(max_coords[0] - min_coords[0]) + 1)
    return result, coords


def rotate_transform_matrix(transform):
    """Rotate matrix so it can be applied to row:col coordinates."""
    matrix = transform.params[(1, 0, 2), :][:, (1, 0, 2)]
    return type(transform)(matrix)


def warp_image(image, transform, output_shape):
    """Apply transformation to an image and its mask

    image ((W, H, 3)  np.ndarray) : image for transformation
    transform (skimage.transform.ProjectiveTransform): transformation to apply
    output_shape (int, int) : shape of the final pano

    Returns:
        (W, H, 3)  np.ndarray : warped image
        (W, H)  np.ndarray : warped mask
    """
    img_ = warp(image, rotate_transform_matrix(transform), output_shape=output_shape)
    return img_, (img_[..., 1] > 0) | (img_[..., 0] > 0) | (img_[..., 2] > 0)


def merge_pano(image_collection, final_center_warps, output_shape):
    """ Merge the whole panorama

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)
    for i in range(len(final_center_warps)):
        transformer = ProjectiveTransform(np.linalg.inv(final_center_warps[i].params))
        img, mask = warp_image(image_collection[i], transformer, output_shape)
        mask = mask * ~result_mask
        img = np.dstack((img[..., 0] * mask, img[..., 1] * mask, img[..., 2] * mask))
        result += img
        result_mask += mask
    return np.clip(np.rint(result * 255), 0, 255).astype(np.uint8)


def get_gaussian_pyramid(image, n_layers=6, sigma=2):
    """Get Gaussian pyramid.

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Gaussian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Gaussian pyramid

    """
    layer = image.copy()
    pyramid = [image.copy()]
    for _ in range(n_layers - 1):
        layer = gaussian(layer, sigma)
        pyramid.append(layer)
    return tuple(pyramid)


def get_laplacian_pyramid(image, n_layers=6, sigma=2):
    """Get Laplacian pyramid

    image ((W, H, 3)  np.ndarray) : original image
    n_layers (int) : number of layers in Laplacian pyramid
    sigma (int) : Gaussian sigma

    Returns:
        tuple(n_layers) Laplacian pyramid
    """
    pyramid = list(get_gaussian_pyramid(image, n_layers, sigma))
    for i in range(1, len(pyramid)):
        pyramid[i - 1] -= pyramid[i]
    return tuple(pyramid)


def merge_laplacian_pyramid(laplacian_pyramid):
    """Recreate original image from Laplacian pyramid

    laplacian pyramid: tuple of np.array (h, w, 3)

    Returns:
        np.array (h, w, 3)
    """
    return sum(laplacian_pyramid)


def increase_contrast(image_collection):
    """Increase contrast of the images in collection"""
    result = []

    for img in image_collection:
        img = img.copy()
        for i in range(img.shape[-1]):
            img[:, :, i] -= img[:, :, i].min()
            img[:, :, i] /= img[:, :, i].max()
        result.append(img)

    return result


def gaussian_merge_pano(image_collection, final_center_warps, output_shape, n_layers=6, image_sigma=4, merge_sigma=10):
    """ Merge the whole panorama using Laplacian pyramid

    image_collection (Tuple[N]) : list of all images
    final_center_warps (Tuple[N])  : transformations
    output_shape (int, int) : shape of the final pano
    n_layers (int) : number of layers in Laplacian pyramid
    image_sigma (int) :  sigma for Gaussian filter for images
    merge_sigma (int) : sigma for Gaussian filter for masks

    Returns:
        (output_shape) np.ndarray: final pano
    """
    result = np.zeros(output_shape + (3,))
    result_mask = np.zeros(output_shape, dtype=np.bool8)
    for img_col, fin_center in zip(image_collection, final_center_warps):
        transformer = ProjectiveTransform(np.linalg.inv(fin_center.params))
        img, mask = warp_image(img_col, transformer, output_shape)
        mm = np.any(mask * result_mask, axis=0).nonzero()
        mask_m = np.ones(output_shape, dtype=np.bool8)
        if not mm[0].size:
            mask_m[:, :0] = 0
        else:
            mask_m[:, :(mm[0][-1] + mm[0][0]) // 2] = 0
        lap_pyramid = []
        for g, a, b in zip(get_gaussian_pyramid(mask_m.astype(np.float64), n_layers, merge_sigma),
                           get_laplacian_pyramid(img, n_layers, image_sigma),
                           get_laplacian_pyramid(result, n_layers, image_sigma)):
            gauss_inv = 1 - g
            lap_pyramid.append(np.dstack((b[..., 0] * gauss_inv, b[..., 1] * gauss_inv, b[..., 2] * gauss_inv)) +
                               np.dstack((a[..., 0] * g, a[..., 1] * g, a[..., 2] * g)))
        result = merge_laplacian_pyramid(lap_pyramid)
        result_mask += mask * ~result_mask
    return np.clip(np.rint(result * 255), 0, 255).astype(np.uint8)


def cylindrical_inverse_map(coords, h, w, scale):
    """Function that transform coordinates in the output image
    to their corresponding coordinates in the input image
    according to cylindrical transform.

    Use it in skimage.transform.warp as `inverse_map` argument

    coords ((M, 2) np.ndarray) : coordinates of output image (M == col * row)
    h (int) : height (number of rows) of input image
    w (int) : width (number of cols) of input image
    scale (int or float) : scaling parameter

    Returns:
        (M, 2) np.ndarray : corresponding coordinates of input image (M == col * row) according to cylindrical transform
    """
    # your code here
    pass


def warp_cylindrical(img, scale=None, crop=True):
    """Warp image to cylindrical coordinates

    img ((H, W, 3)  np.ndarray) : image for transformation
    scale (int or None) : scaling parameter. If None, defaults to W * 0.5
    crop (bool) : crop image to fit (remove unnecessary zero-padding of image)

    Returns:
        (H, W, 3)  np.ndarray : warped image (H and W may differ from original)
    """
    # your code here
    pass


# Pick a good scale value for the 5 test image sets
cylindrical_scales = {
    0: None,
    1: None,
    2: None,
    3: None,
    4: None,
}
