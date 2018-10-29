import numpy as np
from numba import jit


@jit(nopython=True)
def extend_same(image, border_width):
    h, w, d = image.shape
    s = border_width

    new_shape = (h + 2 * s, w + 2 * s, d)
    out_image = np.zeros(new_shape, dtype=np.float32)

    out_image[s:h + s, s:w + s, :] = image

    out_image[:s, :s, :] = np.ones((s, s, d)) * image[0, 0, :]
    out_image[:s, s + w:, :] = np.ones((s, s, d)) * image[0, -1, :]
    out_image[s + h:, :s, :] = np.ones((s, s, d)) * image[-1, 0, :]
    out_image[s + h:, s + w:, :] = np.ones((s, s, d)) * image[-1, -1, :]

    for x in range(h):
        target_row = s + x
        value_left = image[x, 0, :]
        value_right = image[x, -1, :]
        for y in range(s):
            out_image[target_row, y, :] = value_left
            out_image[target_row, y - s, :] = value_right

    for y in range(w):
        target_column = s + y
        value_up = image[0, y, :]
        value_low = image[-1, y, :]
        for x in range(s):
            out_image[x, target_column, :] = value_up
            out_image[x - s, target_column, :] = value_low

    return out_image


@jit(nopython=True)
def extend_mean(image, border_width):
    h, w, d = image.shape
    s = border_width

    mean = np.mean(image)

    new_shape = (h + 2 * s, w + 2 * s, d)
    out_image = np.ones(new_shape, dtype=np.float32) * mean

    out_image[s:h + s, s:w + s, :] = image
    return out_image


@jit(nopython=True)
def __sw_convolution(img, filter):
    fs = filter.shape[0]
    hfs = int(fs / 2)
    h, w, d = img.shape

    out_image = np.zeros((h - 2 * hfs, w - 2 * hfs, d), dtype=np.float32)

    for x in range(0, h - 2 * hfs):
        for y in range(0, w - 2 * hfs):
            for z in range(d):
                v = 0.0
                for x_ in range(fs):
                    for y_ in range(fs):
                        v += img[x + x_, y + y_, z] * filter[x_, y_]

                out_image[x, y, z] = v

    return out_image


@jit(nopython=True)
def __fast_sw_convolution_sv(img, u0, v0):
    fs = u0.shape[0]
    hfs = int(fs / 2)

    h, w, d = img.shape

    mid_image = np.zeros((h - 2 * hfs, w - 2 * hfs, d), dtype=np.float32)
    for x in range(0, h - 2 * hfs):
        for y in range(0, w - 2 * hfs):
            for z in range(d):
                v = 0.0
                for x_ in range(fs):
                    v += img[x + x_, y + hfs, z] * u0[x_]
                mid_image[x, y, z] = v

    mid_image = extend_same(mid_image, hfs)

    out_image = np.zeros((h - 2 * hfs, w - 2 * hfs, d), dtype=np.float32)

    for x in range(0, h - 2 * hfs):
        for y in range(0, w - 2 * hfs):
            for z in range(d):
                v = 0.0
                for y_ in range(fs):
                    v += mid_image[x + hfs, y + y_, z] * v0[y_]
                out_image[x, y, z] = v

    return out_image


@jit(nopython=True)
def convolution(I, H):
    """Convolves an image with a filter matrix. Returns the filtered image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image to convolve
        H : ndarray of float64
            2D array, filter to convolve with

        Returns
        -------
        out : ndarray of float64
            3D array, convolved image

        Notes
        -----
        Input array will be extended with border values,
        so the resulting array will have the same shape as the input array.
        Function accepts single or multi channel images.
        Convolution will be performed channel-wise, so the filtered image
        will have the same number of channels as the input image.
    """

    assert I.ndim == 3, "Image to convolve must have three dimensions."
    fs = H.shape[0]
    assert fs % 2 != 0, "Filter size must be odd!"
    assert H.shape[0] == H.shape[1], "Function only supports square filter matrices."
    hfs = int(fs / 2)
    image = extend_same(I, hfs)
    u, s, vh = np.linalg.svd(H, True)
    if s[1] < 1e-15:
        s0_root = np.sqrt(s[0])
        u0 = u[:, 0] * s0_root
        v0 = vh[0, :] * s0_root
        return __fast_sw_convolution_sv(image, u0, v0)
    if s[2] < 1e-15:
        s0_root = np.sqrt(s[0])
        u0 = u[:, 0] * s0_root
        v0 = vh[0, :] * s0_root
        s1_root = np.sqrt(s[1])
        u1 = u[:, 1] * s1_root
        v1 = vh[1, :] * s1_root
        return __fast_sw_convolution_sv(image, u0, v0) + \
               __fast_sw_convolution_sv(image, u1, v1)
    else:
        return __sw_convolution(image, H)


@jit(nopython=True)
def gaussian_blur(I, sigma):
    """Convolves an image with a gaussian filter. Returns the filtered image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing the image to convolve
        sigma : float64
            Standard deviation of the gaussian filter

        Returns
        -------
        out : ndarray of float64
            3D array, convolved image

        Notes
        -----
        Input array will be extended with border values,
        so the resulting array will have the same shape as the input array.
        Function accepts single and multi channel images.
    """

    # code cryptified (Exercise in Lab)

    q = int(6 * sigma)
    t = sigma ** 2 * 2
    q += (1 if q % 2 == 0 else 0)
    f = np.zeros((q, q), dtype=np.float32)
    o = q // 2
    k = o + 1
    for a in range(k):
        for b in range(a, k):
            g = 1.0 / (np.pi * t) * np.exp(-((a - o) ** 2 + (b - o) ** 2) / t)
            f[a, b] = f[-a - 1, b] = f[-a - 1, -b - 1] = f[a, -b - 1] = g
            if a != b:
                f[b, a] = f[-b - 1, a] = f[-b - 1, -a - 1] = f[b, -a - 1] = g

    return convolution(I, f)


@jit(nopython=True)
def median_filter(I, n):
    if n % 2 == 0:
        n += 1
    q = np.zeros_like(I)
    s = n // 2
    k = extend_same(I, s)
    h, w, d = k.shape
    for x in range(0, h - 2 * s):
        for y in range(0, w - 2 * s):
            for z in range(d):
                m = k[x:x + n, y:y + n, z]
                q[x, y, z] = np.median(m)
    return q


@jit(nopython=True)
def sub_pixel(I, x, y, allow_oob=False):
    assert I.ndim == 3, 'Image has to be a 3D-array!'
    h, w = I.shape[:2]

    if not allow_oob:
        assert x >= 0, 'x must not be negative'
        assert x < h - 1, 'x exceeds image height'
        assert y >= 0, 'y must not be negative'
        assert y < w - 1, 'y exceeds image width'
    else:
        x = min(max(x, 0.0), h - 1.0)
        y = min(max(y, 0.0), w - 1.0)

    x_low = int(x)
    if x % 1.0 == 0:
        x_high = x_low
    else:
        x_high = x_low + 1

    y_low = int(y)
    if x % 1.0 == 0:
        y_high = y_low
    else:
        y_high = y_low + 1

    s = x - x_low
    t = y - y_low

    s_ = 1 - s
    t_ = 1 - t

    v00 = I[x_low, y_low, :]
    v10 = I[x_high, y_low, :]
    v01 = I[x_low, y_high, :]
    v11 = I[x_high, y_high, :]

    return (s_ * t_ * v00 + s * t_ * v10 + s_ * t * v01 + s * t * v11).astype(np.float32)


@jit(nopython=True)
def rescale(I, factor, interpolate):
    h, w, d = I.shape
    ext_image = extend_same(I, 1)

    h_ = int(h * factor)
    w_ = int(w * factor)
    d_ = d

    result = np.zeros((h_, w_, d_), dtype=np.float32)

    for x_ in range(h_):
        for y_ in range(w_):
            x = x_ / factor
            y = y_ / factor

            if interpolate:
                result[x_, y_, :] = sub_pixel(ext_image, 1 + x, 1 + y)
            else:
                result[x_, y_, :] = ext_image[1 + int(np.round(x)), 1 + int(np.round(y))]

    return result


@jit(nopython=True)
def extract_patch(I, rad, cx, cy, patchsize, interpolate, flipx, flipy):
    h, w, d = I.shape
    result = np.zeros((patchsize, patchsize, d), dtype=np.float32)

    sr = np.sin(rad)
    cr = np.cos(rad)

    for x_ in range(patchsize):
        rx = x_ - patchsize / 2
        for y_ in range(patchsize):
            ry = y_ - patchsize / 2

            dx = cr * rx + sr * ry
            dy = -1 * sr * rx + cr * ry

            if flipx:
                dx *= -1
            if flipy:
                dy *= -1
            x = cx + dx
            y = cy + dy

            if interpolate:
                result[x_, y_, :] = sub_pixel(I, x, y)
            else:
                xr = min(max(int(np.round(x)), 0), h - 1)
                yr = min(max(int(np.round(y)), 0), w - 1)
                result[x_, y_, :] = I[xr, yr]

    return result


## colorspaces
# Value-ranges:
# RGB: 0.0 - 1.0 (float)
# H:   0.0 - 2pi (float)
#
# TODO continue & correct this!

def RGB2HSI(RGB):
    RGB_normalized = RGB / 255.0  # Normalize values to 0.0 - 1.0 (float64)
    R = RGB_normalized[:, :, 0]  # Split channels
    G = RGB_normalized[:, :, 1]
    B = RGB_normalized[:, :, 2]

    I = np.mean(RGB, axis=2)  # Compute intensity

    M = np.max(RGB_normalized, axis=2)  # Compute max, min & chroma
    m = np.min(RGB_normalized, axis=2)
    C = M - m

    hue_defined = C > 0  # Check if hue can be computed

    r_is_max = np.logical_and(R == M, hue_defined)  # Computation of hue depends on max
    g_is_max = np.logical_and(G == M, hue_defined)
    b_is_max = np.logical_and(B == M, hue_defined)

    H = np.zeros_like(m)  # Compute hue
    H_r = ((G[r_is_max] - B[r_is_max]) / C[r_is_max]) % 6
    H_g = ((B[g_is_max] - R[g_is_max]) / C[g_is_max]) + 2
    H_b = ((R[b_is_max] - G[b_is_max]) / C[b_is_max]) + 4

    H[r_is_max] = H_r
    H[g_is_max] = H_g
    H[b_is_max] = H_b
    H *= 60

    sat_defined = I > 0
    I /= 255.0

    S = np.zeros_like(m)  # Compute saturation
    O = np.ones_like(m)
    S[sat_defined] = O[sat_defined] - (m[sat_defined] / I[sat_defined])

    return np.dstack((H, S, I))  # H[0-360] , S[0-1], I[0-1]


def HSI2RGB(HSI):
    H = HSI[:, :, 0]  # Split attributes
    S = HSI[:, :, 1]
    I = HSI[:, :, 2]

    H_ = H / 60.0  # Normalize hue
    Z = 1 - np.abs(H_ % 2.0 - 1.0)

    C = 3.0 * I * S / (1.0 + Z)  # Compute chroma

    X = C * Z

    H_0_1 = np.logical_and(0 <= H_, H_ <= 1)  # Store color orderings
    H_1_2 = np.logical_and(1 < H_, H_ <= 2)
    H_2_3 = np.logical_and(2 < H_, H_ <= 3)
    H_3_4 = np.logical_and(3 < H_, H_ <= 4)
    H_4_5 = np.logical_and(4 < H_, H_ <= 5)
    H_5_6 = np.logical_and(5 < H_, H_ <= 6)

    R1G1B1 = np.zeros_like(HSI)  # Compute relative color values
    Z = np.zeros_like(H)

    R1G1B1[H_0_1] = np.dstack((C[H_0_1], X[H_0_1], Z[H_0_1]))
    R1G1B1[H_1_2] = np.dstack((X[H_1_2], C[H_1_2], Z[H_1_2]))
    R1G1B1[H_2_3] = np.dstack((Z[H_2_3], C[H_2_3], X[H_2_3]))
    R1G1B1[H_3_4] = np.dstack((Z[H_3_4], X[H_3_4], C[H_3_4]))
    R1G1B1[H_4_5] = np.dstack((X[H_4_5], Z[H_4_5], C[H_4_5]))
    R1G1B1[H_5_6] = np.dstack((C[H_5_6], Z[H_5_6], X[H_5_6]))

    m = I * (1.0 - S)
    RGB = R1G1B1 + np.dstack((m, m, m))  # Adding the value correction

    return (RGB * 255).astype(np.ubyte)


def RGB2LAB(RGB):
    HSI = RGB2HSI(RGB)

    H = HSI[:, :, 0] / 180.0 * np.pi  # Split attributes
    S = HSI[:, :, 1]
    I = HSI[:, :, 2]

    L = I
    A = np.cos(H) * S
    B = np.sin(H) * S

    return np.dstack((L, A, B))


def LAB2RGB(LAB):
    L = LAB[:, :, 0]  # Split attributes
    A = LAB[:, :, 1]
    B = LAB[:, :, 2]

    I = L
    H = np.arctan2(B, A) * 180.0 / np.pi
    H[H < 0] += 360
    S = np.sqrt(A ** 2 + B ** 2)
    S = np.minimum(np.sqrt(A ** 2 + B ** 2), 1)  # this should be C, not S

    return HSI2RGB(np.dstack((H, S, I)))


def RGB2HSV(RGB):
    RGB_normalized = RGB / 255.0                           # Normalize values to 0.0 - 1.0 (float64)
    R = RGB_normalized[:, :, 0]                            # Split channels
    G = RGB_normalized[:, :, 1]
    B = RGB_normalized[:, :, 2]

    v_max = np.max(RGB_normalized, axis=2)                 # Compute max, min & chroma
    v_min = np.min(RGB_normalized, axis=2)
    C = v_max - v_min

    hue_defined = C > 0                                    # Check if hue can be computed

    r_is_max = np.logical_and(R == v_max, hue_defined)     # Computation of hue depends on max
    g_is_max = np.logical_and(G == v_max, hue_defined)
    b_is_max = np.logical_and(B == v_max, hue_defined)

    H = np.zeros_like(v_max)                               # Compute hue
    H_r = ((G[r_is_max] - B[r_is_max]) / C[r_is_max]) % 6
    H_g = ((B[g_is_max] - R[g_is_max]) / C[g_is_max]) + 2
    H_b = ((R[b_is_max] - G[b_is_max]) / C[b_is_max]) + 4

    H[r_is_max] = H_r
    H[g_is_max] = H_g
    H[b_is_max] = H_b
    H *= 60

    V = v_max                                              # Compute value

    sat_defined = V > 0

    S = np.zeros_like(v_max)                               # Compute saturation
    S[sat_defined] = C[sat_defined] / V[sat_defined]

    return np.dstack((H, S, V))


def HSV2RGB(HSV):
    H = HSV[:, :, 0]                                           # Split attributes
    S = HSV[:, :, 1]
    V = HSV[:, :, 2]

    C = V * S                                                  # Compute chroma

    H_ = H / 60.0                                              # Normalize hue
    X  = C * (1 - np.abs(H_ % 2 - 1))                          # Compute value of 2nd largest color

    H_0_1 = np.logical_and(0 <= H_, H_<= 1)                    # Store color orderings
    H_1_2 = np.logical_and(1 <  H_, H_<= 2)
    H_2_3 = np.logical_and(2 <  H_, H_<= 3)
    H_3_4 = np.logical_and(3 <  H_, H_<= 4)
    H_4_5 = np.logical_and(4 <  H_, H_<= 5)
    H_5_6 = np.logical_and(5 <  H_, H_<= 6)

    R1G1B1 = np.zeros_like(HSV)                                # Compute relative color values
    Z = np.zeros_like(H)

    R1G1B1[H_0_1] = np.dstack((C[H_0_1], X[H_0_1], Z[H_0_1]))
    R1G1B1[H_1_2] = np.dstack((X[H_1_2], C[H_1_2], Z[H_1_2]))
    R1G1B1[H_2_3] = np.dstack((Z[H_2_3], C[H_2_3], X[H_2_3]))
    R1G1B1[H_3_4] = np.dstack((Z[H_3_4], X[H_3_4], C[H_3_4]))
    R1G1B1[H_4_5] = np.dstack((X[H_4_5], Z[H_4_5], C[H_4_5]))
    R1G1B1[H_5_6] = np.dstack((C[H_5_6], Z[H_5_6], X[H_5_6]))

    m = V - C
    RGB = R1G1B1 + np.dstack((m, m, m))                        # Adding the value correction

    return RGB