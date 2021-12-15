import os
import queue

import numpy as np
import scipy
from scipy import ndimage

import nowandfuture.util.preproccess as prp
from nowandfuture.surface_distance import lookup_tables


# todo use A start
def find_nearest_path(data, start_point, stop_point):
    def is_valid(point, shape):
        for i, limit in zip(tuple(point), shape):
            if i >= limit or i < 0:
                return False

        return True

    directions = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
    visit = np.zeros_like(data)
    q = queue.Queue()
    length = 0
    q.put((start_point, length))

    path = queue.Queue()

    max_length, farthest_point = 0, start_point
    path.put(start_point)

    while not q.empty():
        cur_point, cur_length = q.get()

        for d in directions:
            next_point = [i + j for i, j in zip(cur_point, d)]

            if is_valid(next_point, data.shape) and visit[next_point[0], next_point[1], next_point[2]] == 0 and data[
                next_point[0], next_point[1], next_point[2]] == 1:

                if cur_length + 1 > max_length:
                    max_length = cur_length + 1
                    path.put(next_point)

                if next_point == stop_point:
                    break

                visit[next_point[0], next_point[1], next_point[2]] = 1
                q.put((next_point, cur_length + 1))

    return path


def find_endpoint(data, start_point, stop_points):
    def is_valid(point, shape):
        for i, limit in zip(tuple(point), shape):
            if i >= limit or i < 0:
                return False

        return True

    directions = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
    visit = np.zeros_like(data)
    q = queue.Queue()
    length = 0
    q.put((start_point, length))

    max_length, farthest_point = 0, start_point

    while not q.empty():
        cur_point, cur_length = q.get()

        for d in directions:
            next_point = [i + j for i, j in zip(cur_point, d)]

            if is_valid(next_point, data.shape) and visit[next_point[0], next_point[1], next_point[2]] == 0 and data[
                next_point[0], next_point[1], next_point[2]] == 1:

                if next_point in stop_points:
                    if cur_length + 1 > max_length:
                        max_length = cur_length + 1
                        # print(cur_length)
                        farthest_point = next_point

                visit[next_point[0], next_point[1], next_point[2]] = 1
                q.put((next_point, cur_length + 1))

    return max_length, farthest_point


def get_contour(mask_gt, spacing_mm):
    # _assert_is_bool_numpy_array("mask_gt", mask_gt)
    # _assert_is_bool_numpy_array("mask_pred", mask_pred)

    # if not len(mask_gt.shape)  == len(spacing_mm):
    #     raise ValueError("The arguments must be of compatible shape. Got mask_gt "
    #                      "with {} dimensions ({}) and mask_pred with {} dimensions "
    #                      "({}), while the spacing_mm was {} elements.".format(
    #         len(mask_gt.shape),
    #         mask_gt.shape, len(mask_pred.shape), mask_pred.shape,
    #         len(spacing_mm)))

    num_dims = len(spacing_mm)
    if num_dims == 2:
        # _check_2d_numpy_array("mask_gt", mask_gt)
        # _check_2d_numpy_array("mask_pred", mask_pred)

        # compute the area for all 16 possible surface elements
        # (given a 2x2 neighbourhood) according to the spacing_mm
        neighbour_code_to_surface_area = (
            lookup_tables.create_table_neighbour_code_to_contour_length(spacing_mm))
        kernel = lookup_tables.ENCODE_NEIGHBOURHOOD_2D_KERNEL
        full_true_neighbours = 0b1111
    elif num_dims == 3:
        # _check_3d_numpy_array("mask_gt", mask_gt)
        # _check_3d_numpy_array("mask_pred", mask_pred)

        # compute the area for all 256 possible surface elements
        # (given a 2x2x2 neighbourhood) according to the spacing_mm
        neighbour_code_to_surface_area = (
            lookup_tables.create_table_neighbour_code_to_surface_area(spacing_mm))
        kernel = lookup_tables.ENCODE_NEIGHBOURHOOD_3D_KERNEL
        full_true_neighbours = 0b11111111
    else:
        raise ValueError("Only 2D and 3D masks are supported, not "
                         "{}D.".format(num_dims))

    # compute the bounding box of the masks to trim the volume to the smallest
    # possible processing subvolume
    # bbox_min, bbox_max = _compute_bounding_box(mask_gt | mask_pred)
    # Both the min/max bbox are None at the same time, so we only check one.
    # if bbox_min is None:
    #     return {
    #         "distances_gt_to_pred": np.array([]),
    #         "distances_pred_to_gt": np.array([]),
    #         "surfel_areas_gt": np.array([]),
    #         "surfel_areas_pred": np.array([]),
    #     }

    # crop the processing subvolume.
    # cropmask_gt = _crop_to_bounding_box(mask_gt, bbox_min, bbox_max)
    # cropmask_pred = _crop_to_bounding_box(mask_pred, bbox_min, bbox_max)

    # compute the neighbour code (local binary pattern) for each voxel
    # the resulting arrays are spacially shifted by minus half a voxel in each
    # axis.
    # i.e. the points are located at the corners of the original voxels
    neighbour_code_map_gt = ndimage.filters.correlate(
        mask_gt.astype(np.uint8), kernel, mode="constant", cval=0)

    # create masks with the surface voxels
    borders_gt = ((neighbour_code_map_gt != 0) &
                  (neighbour_code_map_gt != full_true_neighbours))
    # surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
    # surfel_areas_gt = surface_area_map_gt[borders_gt]

    print(borders_gt.shape)
    return borders_gt


def find2endpoints(data):
    border = get_contour(data, (1, 1, 1))
    border = np.argwhere(border == 1)
    start_point = border[0]
    # print(start_point)
    _, endpoint0 = find_endpoint(data, start_point, border)
    _, endpoint1 = find_endpoint(data, endpoint0, border)
    # print(endpoint0, endpoint1)
    return endpoint0, endpoint1


def recolor(start_point, data, r, color):
    def is_valid(point, shape):
        for i, limit in zip(tuple(point), shape):
            if i >= limit or i < 0:
                return False

        return True

    directions = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]

    visit = np.zeros_like(data)
    q = queue.Queue()
    length = 0
    q.put((start_point, length))

    while not q.empty():
        cur_point, cur_length = q.get()

        for d in directions:
            next_point = [i + j for i, j in zip(cur_point, d)]

            if is_valid(next_point, data.shape) and visit[next_point[0], next_point[1], next_point[2]] == 0 and data[
                next_point[0], next_point[1], next_point[2]] >= 1 and cur_length + 1 < r:
                visit[next_point[0], next_point[1], next_point[2]] = 1
                q.put((next_point, cur_length + 1))

                data[next_point[0], next_point[1], next_point[2]] = color


def get_landmarks_by_file(file_path, segment_count=8):
    return get_landmarks(data=prp.read_one_voxel(file_path), segment_count=segment_count)


def get_landmarks(data, segment_count=8, recolored_file_path=None):
    p0, p1 = find2endpoints(data)
    # [187, 5, 11][2, 22, 84]

    # p0 = [187, 5, 11]
    # p1 = [2, 22, 84]
    border = get_contour(data, (1, 1, 1))
    border = np.argwhere(border == 1)

    path = find_nearest_path(data, p0, p1)
    # print(path)
    i = 0
    path_list = []
    while not path.empty():
        p = path.get()
        i += 1
        recolor(p, data, 3, 3)
        path_list.append(p)

    recolor(p0, data, 20, 2)
    recolor(p1, data, 20, 2)

    segment = i // segment_count

    landmarks = []

    for j in range(0, i, segment):
        landmarks.append(path_list[j])

    normal_vectors = []
    prelandmarks = landmarks[1: -1]
    for idx, lm in enumerate(prelandmarks):
        nv = [l - k for l, k in zip(lm, landmarks[idx - 1])]
        normal_vectors.append(nv)

    poslandmark = []
    for idx, lm in enumerate(prelandmarks):
        outline = []
        sumX, sumY, sumZ = 0, 0, 0
        for p in border:
            nv = normal_vectors[idx]
            v = [l - k for l, k in zip(lm, p)]
            if abs(v[0] * nv[0] + v[1] * nv[1] + v[2] * nv[2]) < 5:
                outline.append(p)
                sumX += p[0]
                sumY += p[1]
                sumZ += p[2]
                recolor(p, data, 3, 4)

        lgt = len(outline)
        center = [sumX // lgt, sumY // lgt, sumZ // lgt]

        poslandmark.append(center)

    landmarks = [p0] + poslandmark + [p1]

    a = np.array(landmarks).reshape((1, -1))

    if recolored_file_path:
        file_name = os.path.basename(recolored_file_path)
        dir_name = os.path.dirname(recolored_file_path)

        prp.write2nii(data, "{}colored_{}".format(dir_name, file_name))

    return a


def get_agv_var4landmarks(voxel_pathes, agv_save_path, eigen_save_path):
    sum_lm = None
    landmarks_list = []
    for p in voxel_pathes:
        lm_array = get_landmarks(p)
        landmarks_list.append(lm_array)

        if sum_lm is None:
            sum_lm = lm_array
        else:
            sum_lm += lm_array

    a_bar = sum_lm / len(voxel_pathes)

    np.save(agv_save_path, a_bar)

    cvm = None
    # cvm
    for a in landmarks_list:
        d_a = (a - a_bar)
        cov = d_a.T @ d_a
        if cvm is None:
            cvm = cov
        else:
            cvm += cov

    eigenVal, eigenVec = scipy.linalg.eig(cvm / len(landmarks_list))

    # [[ 188.    3.5  35.5  176.5  16.5  33.5 138.5  35.5  25.5 123.5  44.5  28.5
    #   104.   63.   37.   75.5  71.5  53.5  47.5  64.   64.   22.5  44.5  77.5
    #   1.   32.   89.5]]

    # print(eigenVec)
    # print(eigenVal)
    # [8.30663285e+04 + 0.00000000e+00j - 6.41337537e-13 + 0.00000000e+00j
    #  1.04542150e+03 + 0.00000000e+00j  1.20504223e-12 + 1.09766033e-12j
    #  1.20504223e-12 - 1.09766033e-12j - 1.47020274e-13 + 1.27391980e-12j
    #  - 1.47020274e-13 - 1.27391980e-12j - 8.96980006e-13 + 0.00000000e+00j
    #  8.38014155e-13 + 0.00000000e+00j - 1.74869246e-13 + 3.85598433e-13j
    #  - 1.74869246e-13 - 3.85598433e-13j - 3.04450547e-13 + 2.14939001e-13j
    #  - 3.04450547e-13 - 2.14939001e-13j  6.66103903e-14 + 2.75202668e-13j
    #  6.66103903e-14 - 2.75202668e-13j  2.93550122e-13 + 4.53744646e-15j
    #  2.93550122e-13 - 4.53744646e-15j - 7.84299834e-14 + 8.85828358e-14j
    #  - 7.84299834e-14 - 8.85828358e-14j - 6.94688339e-15 + 9.69975628e-14j
    #  - 6.94688339e-15 - 9.69975628e-14j  4.48156261e-14 + 4.95415920e-14j
    #  4.48156261e-14 - 4.95415920e-14j - 2.11578486e-14 + 4.14298171e-14j
    #  - 2.11578486e-14 - 4.14298171e-14j  2.46578613e-14 + 0.00000000e+00j
    #  - 4.15031377e-16 + 0.00000000e+00j]

    np.savez(eigen_save_path, eigenVal=eigenVal, eigenVec=eigenVec)

    return a_bar, eigenVal, eigenVec

#
# if __name__ == '__main__':
#     pathes = [r'D:\download\labels_clip\0001.nii.gz', r'D:\download\labels_clip\0002.nii.gz']
#     a_bar, eigenVal, eigenVec = get_agv_var4landmarks(pathes, "a_bar.npy", "eigen_datas.npz")
