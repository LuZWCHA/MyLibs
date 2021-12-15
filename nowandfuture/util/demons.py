###################################################
# this file is to implement the Demons arithmetic #
###################################################
import math

import numpy as np
import cv2


def transform(img: np.ndarray, t_x, t_y):
    shape = img.shape
    assert len(shape) >= 2
    if len(shape) == 2:
        np.expand_dims(img, axis=2)

    H, W, C = shape
    assert C == 1
    H, W, C = shape
    t_img = np.zeros_like(img)
    grid_x, grid_y = np.meshgrid(range(W), range(H))

    t_idx_x = np.clip(np.squeeze(t_x) + grid_x, 0, W - 1)
    t_idx_y = np.clip(np.squeeze(t_y) + grid_y, 0, H - 1)

    for i in range(W):
        for j in range(H):
            x, y = t_idx_x[j, i], t_idx_y[j, i]
            l_x = int(math.floor(x))
            r_x = min(l_x + 1, W - 1)
            t_y = int(math.floor(y))
            b_y = min(t_y + 1, H - 1)
            i0, i1, i2, i3 = img[t_y, l_x], img[t_y, r_x], img[b_y, l_x], img[b_y, r_x]
            t_img[j, i, ] = linear_interpolate(i0, i1, i2, i3, x - l_x, y - t_y)

    return t_img


def linear_interpolate(i0, i1, i2, i3, dx, dy):
    return i0 * dx * (1 - dy) + i1 * dx * dy + i2 * (1 - dx)*(1 - dy) + i3 * (1 - dx) * dy


def demons(moving_image: np.ndarray, fixed_image: np.ndarray):

    m_shape = moving_image.shape
    f_shape = fixed_image.shape
    assert m_shape == f_shape

    if len(m_shape) < 3:
        moving_image = np.expand_dims(moving_image, -1)
        fixed_image = np.expand_dims(fixed_image, -1)

    t_x = np.zeros(moving_image.shape, dtype='float')
    t_y = np.zeros(moving_image.shape, dtype='float')
    org_image = moving_image.copy()
    for i in range(200):
        if (i % 5) == 0:
            cv2.imwrite(r'D:\Projects\Python\DL\ClockInDemo\data\res_' + str(i) + '.jpg', moving_image)
            # cv2.imwrite(r'D:\Projects\Python\DL\ClockInDemo\data\res_fixed_' + str(i) + '.jpg', fixed_image)
        diff = -np.squeeze(moving_image - fixed_image)
        [d_m_y, d_m_x] = np.gradient(np.squeeze(moving_image))

        d2 = d_m_x**2 + d_m_y**2

        u_x = -(diff * d_m_x) / (d2 + diff**2)
        u_y = -(diff * d_m_y) / (d2 + diff**2)

        u_x[np.isnan(u_x)] = 0
        u_y[np.isnan(u_y)] = 0
        u_x[np.isinf(u_x)] = 0
        u_y[np.isinf(u_y)] = 0

        u_x_s = gaussian_filter(u_x, 3)
        u_y_s = gaussian_filter(u_y, 3)

        u_x_s /= np.max(u_x_s) * 2
        u_y_s /= np.max(u_y_s) * 2

        t_x = t_x + u_x_s
        t_y = t_y + u_y_s

        # t_x = gaussian_filter(t_x, 3, 1)
        # t_y = gaussian_filter(t_y, 3, 1)

        moving_image = transform(org_image, t_x, t_y)
    return moving_image


def gaussian_filter(img: np.ndarray, K_size=3, sigma=1.3):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
        ## Zero padding

    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))

    K /= (2 * np.pi * sigma * sigma)
    K /= K.sum()
    tmp = out.copy()

    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])

    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W, ...]

    return out


if __name__ == "__main__":
    m_img = cv2.imread(r'D:\Projects\Python\DL\ClockInDemo\data\test_1.jpg')

    m_img_gray = cv2.cvtColor(m_img, cv2.COLOR_RGB2GRAY)
    f_img = cv2.imread(r'D:\Projects\Python\DL\ClockInDemo\data\test_2.jpg')
    f_img_gray = cv2.cvtColor(f_img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("image", f_img_gray)
    # cv2.waitKey(0)

    img = demons(m_img_gray, f_img_gray)

    cv2.namedWindow("image")
    cv2.imshow("image", img)
    cv2.waitKey(0)
