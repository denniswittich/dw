import numpy as np, imageio
from numba import *


# ===== IO

def read(path):
    I = imageio.imread(path).astype(np.float32)
    if path.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']:
        I /= 255.0
        v_min = np.min(I)
        v_max = np.max(I)
        assert v_min >= 0.0 and v_max <= 1.0, "out of range while normalization ({},{}) {}".format(v_min, v_max, path)
    return np.atleast_3d(I)


def write(path, I):
    if path.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']:
        v_min = np.min(I)
        v_max = np.max(I)
        assert v_min >= 0.0 and v_max <= 1.0, "out of range to write as ubyte ({},{}) {}".format(v_min, v_max, path)
        I = (I * 255.0).astype(np.ubyte)
    imageio.imwrite(path, I)


# ===== PADDING

@jit(nopython=True)
def extend_same(I, border_width):
    # pads by repeating the border values
    #
    #

    h, w, d = I.shape
    s = border_width

    new_shape = (h + 2 * s, w + 2 * s, d)
    out_image = np.zeros(new_shape, dtype=np.float32)

    out_image[s:h + s, s:w + s, :] = I

    out_image[:s, :s, :] = np.ones((s, s, d)) * I[0, 0, :]
    out_image[:s, s + w:, :] = np.ones((s, s, d)) * I[0, -1, :]
    out_image[s + h:, :s, :] = np.ones((s, s, d)) * I[-1, 0, :]
    out_image[s + h:, s + w:, :] = np.ones((s, s, d)) * I[-1, -1, :]

    for x in range(h):
        target_row = s + x
        value_left = I[x, 0, :]
        value_right = I[x, -1, :]
        for y in range(s):
            out_image[target_row, y, :] = value_left
            out_image[target_row, y - s, :] = value_right

    for y in range(w):
        target_column = s + y
        value_up = I[0, y, :]
        value_low = I[-1, y, :]
        for x in range(s):
            out_image[x, target_column, :] = value_up
            out_image[x - s, target_column, :] = value_low

    return out_image


@jit(nopython=True)
def extend_mean(I, border_width):
    h, w, d = I.shape
    s = border_width

    mean = np.mean(I)

    new_shape = (h + 2 * s, w + 2 * s, d)
    out_image = np.ones(new_shape, dtype=np.float32) * mean

    out_image[s:h + s, s:w + s, :] = I
    return out_image


# ================= CONVERSION ====================

@jit(nopython=True)
def label_map2label_image_avg(image, label_map):
    h, w, d = image.shape

    num_labels = np.max(label_map) + 1
    label_image = np.zeros((h, w, d), dtype=np.float32)
    avgs = np.zeros((num_labels, d), dtype=np.float32)
    pxcounter = np.zeros((num_labels), dtype=np.int64)

    ## CALCULATE AVG. COLOR / GRAY VALUE PER LABEL

    for x in range(h):
        for y in range(w):
            label = label_map[x, y]
            avgs[label, :] += image[x, y, :]
            pxcounter[label] += 1

    for i in range(num_labels):
        avgs[i] /= pxcounter[i]

    ## APPLY AVG. COLOR / GRAY VALUE TO PIXELS

    for x in range(h):
        for y in range(w):
            label = label_map[x, y]
            label_image[x, y, :] = avgs[label]

    return label_image

@jit(nopython=True)
def label_map2label_image(label_map):
    h, w = label_map.shape
    label_image = np.zeros((h, w, 3), dtype=np.float32)
    num_labels = np.max(label_map) + 1

    colors = get_saturated_colors(num_labels)

    for x in range(h):
        for y in range(w):
            label = label_map[x, y]
            label_image[x, y, :] = colors[label, :]

    return label_image


@jit(nopython=True)
def get_saturated_color(hue):
    c = 1.0
    x = 1 - abs((hue / 60) % 2 - 1)
    if hue < 60:
        r_, g_, b_ = c, x, 0
    elif hue < 120:
        r_, g_, b_ = x, c, 0
    elif hue < 180:
        r_, g_, b_ = 0, c, x
    elif hue < 240:
        r_, g_, b_ = 0, x, c
    elif hue < 300:
        r_, g_, b_ = x, 0, c
    else:
        r_, g_, b_ = c, 0, x

    return np.array([r_, g_, b_], dtype=np.float32) * 255


@jit(nopython=True)
def get_random_color():
    return get_saturated_color(np.random.random() * 360)


@jit(nopython=True)
def get_saturated_colors(num_colors):
    colors = np.zeros((num_colors, 3), dtype=np.float32)
    for i in range(num_colors):
        if i == 0:
            colors[i, :] = np.ones(3, dtype=np.float32) * 255
        else:
            hue_i = 57 * i
            colors[i, :] = get_saturated_color(hue_i % 360) * (np.sin(i) / 4 + 0.75)
    return colors


@jit(nopython=True)
def convert_to_1channel(image):
    h, w, d = image.shape
    if d == 1:
        return np.copy(image)

    result = np.sum(image, 2) / 3.0

    return result.reshape((h, w, 1))


@jit(nopython=True)
def get_channel(image, channel):
    return image[:, :, channel:channel + 1:]

@jit(nopython=True)
def convert_to_3channel(image):
    h, w, d = image.shape
    if d == 3:
        return image

    result = np.zeros((h, w, 3), dtype=np.float32)
    result[:, :, 0] = image[:, :, 0]
    result[:, :, 1] = image[:, :, 0]
    result[:, :, 2] = image[:, :, 0]

    return result


@jit(nopython=True)
def convert_to_binary(image):
    gray_image = convert_to_1channel(image)
    return gray_image[:, :, 0] != 0


@jit(nopython=True)
def rgb2hsv(image):
    h, w, d = image.shape
    assert d == 3
    image /= 255

    hsv_image = np.zeros((h, w, 3), dtype=np.float32)
    for x in range(h):
        for y in range(w):
            r, g, b = image[x, y, :]
            v_max = np.max(image[x, y, :])
            v_min = np.min(image[x, y, :])

            # HUE 0 - 360
            hue = 0.0
            if v_max > v_min:
                if r == v_max:
                    hue = 60 * (g - b) / (v_max - v_min)
                elif g == v_max:
                    hue = 120 + 60 * (b - r) / (v_max - v_min)
                elif b == v_max:
                    hue = 240 + 60 * (r - g) / (v_max - v_min)
                if hue < 0:
                    hue += 360

            # SATURATION 0 - 1
            sat = 0.0
            if v_max > 0.0:
                sat = (v_max - v_min) / v_max

            # VALUE 0 - 1
            val = v_max
            hsv_image[x, y, :] = (hue, sat, val)
    return hsv_image


@jit(nopython=True)
def hsv2rgb(hsv_image):
    h_, w_, d_ = hsv_image.shape
    assert d_ == 3

    rgb_image = np.zeros((h_, w_, 3), dtype=np.float32)
    for x in range(h_):
        for y in range(w_):
            h, s, v = hsv_image[x, y]
            h_i = int(h // 60) % 6
            f = h / 60 - h_i
            p = v * (1 - s)
            q = v * (1 - s * f)
            t = v * (1 - s * (1 - f))
            if h_i == 0:
                rgb = (v, t, p)
            elif h_i == 1:
                rgb = (q, v, p)
            elif h_i == 2:
                rgb = (p, v, t)
            elif h_i == 3:
                rgb = (p, q, v)
            elif h_i == 4:
                rgb = (t, p, v)
            else:
                rgb = (v, p, q)
            rgb_image[x, y, :] = rgb

    rgb_image *= 255
    return rgb_image


@jit(nopython=True)
def rgb2hsl(image):
    h, w, d = image.shape
    assert d == 3
    image /= 255

    hsl_image = np.zeros((h, w, 3), dtype=np.float32)
    for x in range(h):
        for y in range(w):
            r, g, b = image[x, y, :]
            v_max = np.max(image[x, y, :])
            v_min = np.min(image[x, y, :])

            # HUE 0 - 360
            hue = 0.0
            if v_max > v_min:
                if r == v_max:
                    hue = 60 * (g - b) / (v_max - v_min)
                elif g == v_max:
                    hue = 120 + 60 * (b - r) / (v_max - v_min)
                elif b == v_max:
                    hue = 240 + 60 * (r - g) / (v_max - v_min)
                if hue < 0:
                    hue += 360

            # SATURATION 0 - 1
            sat = 0.0
            if v_max > 0.0 and v_min < 1:
                sat = (v_max - v_min) / (1 - abs(v_max + v_min - 1))

            # LUMINANCE 0 - 1
            lum = (v_max + v_min) / 2
            hsl_image[x, y, :] = (hue, sat, lum)
    return hsl_image


@jit(nopython=True)
def rgb2lab(image):
    # D65 / 2 deg
    Xn = 95.047
    Yn = 100.0
    Zn = 108.883

    h, w, d = image.shape
    assert d == 3

    lab_image = np.zeros((h, w, 3), dtype=np.float32)
    for x in range(h):
        for y in range(w):
            r, g, b = image[x, y, :]
            X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
            Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
            Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b

            frac_x = X / Xn
            frac_y = Y / Yn
            frac_z = Z / Zn

            if frac_x < 0.008856:
                root_x = 1 / 116 * (24389 / 27 * frac_x + 16)
            else:
                root_x = frac_x ** (1 / 3)

            if frac_y < 0.008856:
                root_y = 1 / 116 * (24389 / 27 * frac_y + 16)
            else:
                root_y = frac_y ** (1 / 3)

            if frac_z < 0.008856:
                root_z = 1 / 116 * (24389 / 27 * frac_z + 16)
            else:
                root_z = frac_z ** (1 / 3)

            L = 116 * root_y - 16
            a = 500 * (root_x - root_y)
            b = 200 * (root_y - root_z)

            lab_image[x, y, :] = (L, a, b)
    return lab_image


@jit(nopython=True)
def binary2gray(mask):
    h, w = mask.shape
    return mask.astype(np.float32).reshape((h, w, 1)) * 255


@jit(nopython=True)
def rgb2pca(image):
    h, w, d = image.shape
    N = h * w
    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])

    ### COMPUTE COVARIANCE MATRIX

    pixels = image.astype(np.float32).reshape((N, 3))
    pixels[:, 0] -= mean_r
    pixels[:, 1] -= mean_g
    pixels[:, 2] -= mean_b

    V = np.dot(pixels.T, pixels) / N

    eigvals, R = np.linalg.eig(V)

    transformed_pixels = np.dot(R.T, pixels.T)
    transformed_image = transformed_pixels.T.reshape(h, w, d)

    return normalize(transformed_image[:, :, 0:1])


### =============== HISTOGRAM =======================

@jit(nopython=True)
def histogram_normalisation(image, outlier_fraction):
    h, w, d = image.shape
    N = h * w
    num_outliers = int(N * (outlier_fraction / 2))
    result = np.zeros(image.shape, dtype=np.float32)

    for z in range(d):
        channel = image[:, :, z]

        for g_low in range(256):
            if np.sum((channel < g_low)) >= num_outliers:
                break

        for g_high in range(255, -1, -1):
            if np.sum((channel > g_high)) >= num_outliers:
                break

        for x in range(h):
            for y in range(w):
                v = image[x, y, z]
                v_n = (v - g_low) * (255 / (g_high - g_low))
                result[x, y, z] = min(max(v_n, 0), 255)

    return result


@jit(nopython=True)
def histogram_equalization(image, alpha):
    h, w, d = image.shape
    result = np.zeros((h, w, d), dtype=np.float32)

    for z in range(d):
        channel = image[:, :, z].astype(np.ubyte)

        cumulative_dist = np.zeros(256, dtype=np.int64)
        cumulative_dist[0] = np.sum((channel == 0).astype(np.int64))
        for i in range(1, 256):
            num_i = np.sum((channel == i).astype(np.int64))

            cumulative_dist[i] = cumulative_dist[i - 1] + num_i

        mapping = cumulative_dist * 255 / cumulative_dist[-1]

        for x in range(h):
            for y in range(w):
                v = channel[x, y]
                result[x, y, z] = mapping[v]

    return result * alpha + image * (1 - alpha)


@jit(nopython=True)
def local_histogram_equalization(image, alpha, M):
    h, w, d = image.shape
    result = np.zeros((h, w, d), dtype=np.float32)

    ### CREATE MAPPING FUNCTIONS PER BLOCK
    ext_image = extend_same(image, 3 * M)

    num_blocks_x = int(h / M) + 2
    num_blocks_y = int(w / M) + 2
    print(num_blocks_x, num_blocks_y)

    mappings = np.zeros((num_blocks_x, num_blocks_y, d, 256), dtype=np.float32)

    for x in range(num_blocks_x):
        for y in range(num_blocks_y):
            for z in range(d):
                local_channel = ext_image[3 * M + x * M - int(M / 2):3 * M + (x + 1) * M - int(M / 2),
                                3 * M + y * M - int(M / 2):3 * M + (y + 1) * M - int(M / 2), z].astype(np.ubyte)
                cumulative_dist = np.zeros(256, dtype=np.int64)
                cumulative_dist[0] = np.sum((local_channel == 0).astype(np.int64))
                for i in range(1, 256):
                    num_i = np.sum((local_channel == i).astype(np.int64))
                    cumulative_dist[i] = cumulative_dist[i - 1] + num_i
                mapping = cumulative_dist * 255 / cumulative_dist[-1]
                mappings[x, y, z, :] = mapping

    for x in range(h):
        for y in range(w):
            for z in range(d):
                v = image[x, y, z]
                xf = x / M
                yf = y / M
                x_low = int(xf)
                y_low = int(yf)
                x_high = x_low + 1
                y_high = y_low + 1
                s = (xf - x_low)
                t = (yf - y_low)
                s_ = 1.0 - s
                t_ = 1.0 - t
                f00 = mappings[x_low, y_low, z, int(v)]
                f10 = mappings[x_high, y_low, z, int(v)]
                f01 = mappings[x_low, y_high, z, int(v)]
                f11 = mappings[x_high, y_high, z, int(v)]
                result[x, y, z] = s_ * t_ * f00 + s * t_ * f10 + s_ * t * f01 + s * t * f11

    return result * alpha + image * (1 - alpha)


# ===== FILTERING

@jit(nopython=True)
def __sw_convolution(I, filter):
    fs = filter.shape[0]
    hfs = int(fs / 2)
    h, w, d = I.shape

    out_image = np.zeros((h - 2 * hfs, w - 2 * hfs, d), dtype=np.float32)

    for x in range(0, h - 2 * hfs):
        for y in range(0, w - 2 * hfs):
            for z in range(d):
                v = 0.0
                for x_ in range(fs):
                    for y_ in range(fs):
                        v += I[x + x_, y + y_, z] * filter[x_, y_]

                out_image[x, y, z] = v

    return out_image


@jit(nopython=True)
def __fast_sw_convolution_sv(I, u0, v0):
    fs = u0.shape[0]
    hfs = int(fs / 2)

    h, w, d = I.shape

    mid_image = np.zeros((h - 2 * hfs, w - 2 * hfs, d), dtype=np.float32)
    for x in range(0, h - 2 * hfs):
        for y in range(0, w - 2 * hfs):
            for z in range(d):
                v = 0.0
                for x_ in range(fs):
                    v += I[x + x_, y + hfs, z] * u0[x_]
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
        I : ndarray of float32
            3D array representing the image to convolve
        H : ndarray of float32
            2D array, filter to convolve with

        Returns
        -------
        out : ndarray of float32
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
        I : ndarray of float32
            3D array representing the image to convolve
        sigma : float32
            Standard deviation of the gaussian filter

        Returns
        -------
        out : ndarray of float32
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


# ===== INTERPOLATION

@jit(float32[:](float32[:, :, :], float32, float32), nopython=True, cache=True)
def get_sub_pixel_2d(I, x, y, check_OOB=True, OOB_value=None):
    """Calculates the value of an arbitrary position in an image using bi-linear interpolation.

    :param I: Float32 array representing the image in HWC convention
    :param x: Sub-pixel x-coordinate
    :param y: Sub-pixel y-coordinate
    :param check_OOB: Whether to check for out-of-boundary
    :param OOB_value: Value to return in case of out-of-boundary
    :return: Interpolated value at sub-pixel coordinate or OOB-value
    """

    h, w = I.shape[:2]

    if check_OOB:
        if (x < 0) or (x >= h - 1) or (y < 0) or (y >= w - 1):
            return OOB_value

    x_low = int(x)
    x_high = x_low + 1
    y_low = int(y)
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
def rescale(I, factor, interpolate=True):
    # new (height, width) will be factor * old (height,width)
    # factor > 1: upsample (if interpolate: bilinear)
    # factor < 1: downsample (if interpolate: gaussian blur + bilinear)
    h, w, d = I.shape
    ext_image = extend_same(I, 1)
    if factor < 1.0 and interpolate:
        ext_image = gaussian_blur(ext_image, 0.5 / factor)

    h_ = round(h * factor)
    w_ = round(w * factor)
    d_ = d

    result = np.zeros((h_, w_, d_), dtype=np.float32)

    for x_ in range(h_):
        for y_ in range(w_):
            x = x_ / factor
            y = y_ / factor
            if interpolate:
                result[x_, y_, :] = get_sub_pixel_2d(ext_image, 1 + x, 1 + y,check_OOB=True,OOB_value=0.0)
            else:
                result[x_, y_, :] = ext_image[1 + int(np.round(x)), 1 + int(np.round(y))]
    return result

@jit(nopython=True)
def rescale_to_shorter_edge(I, shorter_edge_length, interpolate=True):
    # rescales an image such that the shorter edge will have length 'shorter_edge_length'
    h, w, d = I.shape
    factor = shorter_edge_length / min(h, w)

    return rescale(I, factor, interpolate)


# ===== OTHER

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
                result[x_, y_, :] = get_sub_pixel_2d(I, x, y)
            else:
                xr = min(max(int(np.round(x)), 0), h - 1)
                yr = min(max(int(np.round(y)), 0), w - 1)
                result[x_, y_, :] = I[xr, yr]

    return result


@jit(f4[:, :, :](f4[:, :, :], f4, f4, f4, i8, i8, b1, b1, b1), nopython=True, cache=True)
def extract_patch(I, rad, cx, cy, patch_height, patch_width, interpolate, flipx, flipy, out=None): # TODO OUT
    h, w, d = I.shape
    result = np.zeros((patch_height, patch_width, d), dtype=np.float32)

    sr = np.sin(rad)
    cr = np.cos(rad)

    for x_ in range(patch_height):
        rx = x_ - patch_height / 2
        crrx = cr * rx
        negsrrx = -1 * sr * rx
        for y_ in range(patch_width):
            ry = y_ - patch_width / 2
            dx = crrx + sr * ry
            dy = negsrrx + cr * ry
            if flipx:
                x = cx - dx
            else:
                x = cx + dx
            if flipy:
                y = cy - dy
            else:
                y = cy + dy

            if interpolate:
                result[x_, y_, :] = get_sub_pixel_2d(I, x, y)
            else:
                xr = min(max(int(np.round(x)), 0), h - 1)
                yr = min(max(int(np.round(y)), 0), w - 1)
                result[x_, y_, :] = I[xr, yr]
    return result


def describe(I):
    try:
        print('min: {}, max {}, mean {}, shape {}, dtype {}'.format(np.min(I), np.max(I), np.mean(I), I.shape, I.dtype))
    except:
        print('I is not an array')


@jit(nopython=True)
def get_optimal_rotated_shape(h, w, rad):
    rad = abs(rad)
    while rad > 2 * np.pi:
        rad -= 2 * np.pi
    if rad > np.pi:
        rad = rad - 2 * np.pi
        rad = abs(rad)

    swap_extends = False
    if rad > np.pi / 2:
        tmp = w
        w = h
        h = tmp
        swap_extends = True
        rad -= np.pi / 2

    sr = np.sin(rad)
    cr = np.cos(rad)

    a = h / w
    if a < 1:
        a = 1 / a
    best_area = 0.0
    best_height = 0.0
    best_width = 0.0
    for aspect in np.linspace(1, 2 * a, 100):
        for i in range(2):
            if i:
                aspect = 1 / aspect
            # print('\naspect:', aspect)
            h_ = w * aspect
            mbr_w = h_ * sr + w * cr
            mbr_h = h_ * cr + w * sr
            s = min(w / mbr_w, h / mbr_h)
            # print('scale', s)
            h_ = h_ * s
            w_ = w * s
            area = h_ * w_
            # print('h,w,a', h_, w_, area)
            if area > best_area:
                best_area = area
                best_height = h_
                best_width = w_

    return best_height, best_width


@jit(nopython=True)
def get_zoomed_central_rotation(img, rad, interpolate):
    h, w, d = img.shape
    h_, w_ = get_optimal_rotated_shape(h, w, rad)
    return extract_patch(img, rad, h / 2, w / 2, h_, w_, interpolate, False, False)


def draw_mbrs(I, mbrs):
    """Draws minimum bounding rectangles to an image.

        Parameters
        ----------
        I : ndarray of float32
            3D array representing the image

        mbrs : list of tuples of float
            each tuple in list holds the parameters of a minimum bounding rectangle

        Returns
        -------
        out : ndarray of float32
            3D array, image with rectangles

        Notes
        -----
        Definitions
        MBRs : list of tuples of floats : (angle, length, with, center x, center y)
    """
    h, w, _ = I.shape
    J = np.copy(I)
    for phi, L, W, cx, cy in mbrs:
        cp = np.cos(phi)
        sp = np.sin(phi)
        for a in (-L / 2, L / 2):
            for b in np.linspace(-W / 2, W / 2, int(W)):
                x = int(a * cp - b * sp + cx)
                y = int(a * sp + b * cp + cy)
                if x >= 0 and x < h and y >= 0 and y < w:
                    J[x, y, :] = 0.0
                    J[x, y, 0] = 255

        for a in np.linspace(-L / 2, L / 2, int(L)):
            for b in (-W / 2, W / 2):
                x = int(a * cp - b * sp + cx)
                y = int(a * sp + b * cp + cy)
                if x >= 0 and x < h and y >= 0 and y < w:
                    J[x, y, :] = 0.0
                    J[x, y, 0] = 255
    return J


# ================= TRANSFORMATION ================

@jit(nopython=True)
def fourier_transformation(image):
    # radius = Image.shape[0]/2
    # for x in range(0, Image.shape[0]):
    #  for y in range(0, Image.shape[1]):
    #          d = min(((x-radius)**2+(y-radius)**2)**0.5 /radius,1.0)
    #          Image[x,y] =d *0.5 + (1-d)*Image[x,y]

    M, N, d = image.shape
    quotient = 1.0 / np.sqrt(M * N)
    imagFac = -2 * np.pi * 1j

    f_size = int(M / 2)
    f_shape = (2 * f_size + 1, 2 * f_size + 1)

    F = np.zeros(f_shape, dtype=np.complex128)

    for m in range(-f_size, f_size):
        for n in range(0, f_size):
            sum = 0.0j
            for x in range(0, M):
                for y in range(0, N):
                    sum += image[x, y, 0] * np.exp(imagFac * ((m * x) / M + (n * y) / N))
            # print(sum)
            f = sum * quotient
            F[m + f_size, n + f_size] = f
            F[-m + f_size, -n + f_size] = f

    abs_F = np.absolute(F).reshape((2 * f_size + 1, 2 * f_size + 1, 1))

    # trimmed = trim(abs_F,1.0,np.max(abs_F))

    print(np.min(abs_F))
    print(np.max(abs_F))
    return np.log1p(abs_F)


# ================ BASIC TOOLS ====================


@jit(nopython=True)
def invert(image):
    return 255.0 - image


@jit(nopython=True)
def normalize(img):
    #
    #  |-5  3 12|  +5  |0  8 17| *255/13 | 0 120 255|
    #  |-1 -2  1|  ->  |4  3  6|   ->    |60  45  90|
    #  |-1  4  6|      |4  9 11|         |60 135 165|
    #

    min_value = np.min(img)
    max_value = np.max(img)

    assert max_value != 0, "Maximum value of image is zero"

    return (img - min_value) * 255 / (max_value - min_value)  # creates a copy


# ===== SEGMENTATION



@jit(nopython=True)
def watershed_transform(image, sigma, seed_threshold):  # Meyer's flooding algorithm
    h, w, _ = image.shape

    filter_dgx = convolution.derivative_of_gaussian(sigma, 0)
    filter_dgy = convolution.derivative_of_gaussian(sigma, 1)
    dgx = convolution.fast_sw_convolution(image, filter_dgx, convolution.SAME)
    dgy = convolution.fast_sw_convolution(image, filter_dgy, convolution.SAME)
    amplitude_map = (np.sqrt(dgx * dgx + dgy * dgy)).reshape(h, w)

    max_amp_plus = np.max(amplitude_map) + 1

    (gmx, gmy), gmv = get_min_coords_2d_threshold(amplitude_map, max_amp_plus)
    num_candidates = 1

    candidate_map = np.ones((h, w), dtype=np.float32) * max_amp_plus
    visited_map = np.zeros((h, w), dtype=np.bool_)
    label_map = np.zeros((h, w), dtype=np.int64)

    next_label = 1

    candidate_map[gmx, gmy] = gmv
    label_map[gmx, gmy] = next_label
    amplitude_map[gmx, gmy] = max_amp_plus

    # second best minimum
    (gmx, gmy), gmv = get_min_coords_2d_threshold(amplitude_map, max_amp_plus)

    while num_candidates > 0:
        (cx, cy), cv = get_min_coords_2d_threshold(candidate_map, max_amp_plus)

        # check for new seeds
        if cv - seed_threshold > gmv:
            if not (candidate_map[gmx, gmy] < max_amp_plus or visited_map[gmx, gmy]):
                candidate_map[gmx, gmy] = gmv
                num_candidates += 1
                next_label += 1
                label_map[gmx, gmy] = next_label
                cx = gmx
                cy = gmy

            (gmx, gmy), gmv = get_min_coords_2d_threshold(amplitude_map, max_amp_plus)

        # remove candidate from candidates and add to visited
        amplitude_map[cx, cy] = max_amp_plus
        candidate_map[cx, cy] = max_amp_plus
        visited_map[cx, cy] = True
        num_candidates -= 1

        neighbours = get_valid_neighbours(h, w, cx, cy)
        num_neighbours = neighbours.shape[0]

        can_be_labeled = True
        label_vote = 0
        for n in range(num_neighbours):
            nx, ny = neighbours[n, :]
            if not (candidate_map[nx, ny] < max_amp_plus or visited_map[nx, ny]):
                candidate_map[nx, ny] = amplitude_map[nx, ny]
                num_candidates += 1

            label = label_map[nx, ny]

            if label == 0:
                continue

            if label_vote == 0:
                label_vote = label
            elif not label_vote == label:
                can_be_labeled = False

        if can_be_labeled and (not label_map[cx, cy]):
            label_map[cx, cy] = label_vote

    return label_map

@jit(nopython=True)
def k_means(I, k):
    """Performs k-means clustering on an image.

        Parameters
        ----------
        I : ndarray of float64
            3D array representing an image
        k : int
            number of means

        Returns
        -------
        out : ndarray of float64
            3D array, image, where each pixel is set to the value of closest mean

        Notes
        -----
        Distance is computed with L2 norm.
        Function takes single and multi channel images.
    """
    h, w, d = I.shape

    S = np.zeros((h, w), dtype=np.int64)
    k_map = np.zeros_like(S)
    k_map_ = np.copy(k_map)
    counts = np.zeros(k, dtype=np.float64)
    means = np.random.uniform(0.0, np.max(I), (k, d)).astype(np.float64)
    sums = np.zeros((k, d), dtype=np.float64)

    for it in range(100):
        counts *= 0
        sums *= 0
        for x in range(h):
            for y in range(w):
                dist_best = 0.0
                k_best = 0
                for m in range(k):
                    dist_m = np.sum(np.square(means[m] - I[x, y, :]))
                    if dist_m < dist_best or m == 0:
                        k_best = m
                        dist_best = dist_m
                k_map[x, y] = k_best
                sums[k_best, :] += I[x, y, :]
                counts[k_best] += 1

        for m in range(k):
            if counts[m] > 0:
                means[m] = np.copy(sums[m, :] / counts[m])

        if np.sum(k_map - k_map_) == 0:
            if it == 0:
                means = np.random.uniform(0.0, np.max(I), (k, d)).astype(np.float64)
            break
        k_map_ = np.copy(k_map)

    J = np.zeros_like(I)
    for x in range(h):
        for y in range(w):
            J[x, y, :] = means[k_map[x, y]]

    return J


### ========== THRESHOLDING =================

@jit(nopython=True)
def gray_value_thresholding(image, g_min, g_max):
    mask = np.logical_and(image[:, :, 0] >= g_min, image[:, :, 0] <= g_max)
    return mask


@jit(nopython=True)
def rgb_thresholding(image, r_min, r_max, g_min, g_max, b_min, b_max):
    h, w, _ = image.shape
    red_mask = np.logical_and(image[:, :, 0] >= r_min, image[:, :, 0] <= r_max)
    green_mask = np.logical_and(image[:, :, 1] >= g_min, image[:, :, 1] <= g_max)
    blue_mask = np.logical_and(image[:, :, 2] >= b_min, image[:, :, 2] <= b_max)
    mask = np.logical_and(np.logical_and(red_mask, green_mask), blue_mask)
    return mask


@jit(nopython=True)
def hsv_thresholding(hsv_image, h_min, h_max, s_min, s_max, v_min, v_max):
    hsv_image = rgb2hsv(hsv_image)

    if h_min > h_max:
        hue_mask = np.logical_or(hsv_image[:, :, 0] >= h_min, hsv_image[:, :, 0] <= h_max)
    else:
        hue_mask = np.logical_and(hsv_image[:, :, 0] >= h_min, hsv_image[:, :, 0] <= h_max)
    sat_mask = np.logical_and(hsv_image[:, :, 1] >= s_min, hsv_image[:, :, 1] <= s_max)
    val_mask = np.logical_and(hsv_image[:, :, 2] >= v_min, hsv_image[:, :, 2] <= v_max)
    mask = np.logical_and(np.logical_and(hue_mask, sat_mask), val_mask)
    return mask


# ========== REGION GROWING ==============

@jit(nopython=True)
def region_growing_gray(image, h_g):
    h, w, _ = image.shape

    num_unlabeled = h * w
    num_labels = 0
    label_means = np.zeros((h * w), dtype=np.float32)
    label_sizes = np.zeros((h * w), dtype=np.int64)
    label_map = np.ones((h, w), dtype=np.int64) * -1  # -1 means unlabeled
    candidate_list = np.zeros((h * w, 2), dtype=np.int64)  # (index,coordinates)

    labeled_rows = 0

    while num_unlabeled > 0:
        ## SEARCH FOR UNLABELED PIXEL AS SEED

        found_seed = False
        for sx in range(labeled_rows, h):
            for sy in range(w):
                if label_map[sx, sy] == -1:
                    found_seed = True
                    break
            if found_seed:
                break
            labeled_rows += 1

        new_label = num_labels
        num_labels += 1

        label_map[sx, sy] = new_label
        label_means[new_label] = image[sx, sy, 0]
        label_sizes[new_label] = 1

        candidate_list[0, :] = (sx, sy)
        num_candidates = 1
        num_unlabeled -= 1

        while num_candidates > 0:
            num_candidates -= 1
            cx, cy = candidate_list[num_candidates, :]  # pop last candidate
            cg = image[cx, cy, 0]

            valid_neighbours = get_valid_neighbours(h, w, cx, cy)
            num_neighbours = valid_neighbours.shape[0]

            for i in range(num_neighbours):
                nx, ny = valid_neighbours[i, :]
                nl = label_map[nx, ny]
                if nl >= 0:
                    continue  # if neighbour is already labeled

                if abs(label_means[new_label] - cg) <= h_g:
                    label_map[nx, ny] = new_label
                    old_size = label_sizes[new_label]
                    new_size = old_size + 1
                    label_means[new_label] = (image[nx, ny, 0] + label_means[new_label] * old_size) / new_size
                    label_sizes[new_label] = new_size

                    candidate_list[num_candidates, :] = (nx, ny)
                    num_candidates += 1
                    num_unlabeled -= 1

    return label_map


@jit(nopython=True)
def region_growing_color(image, h_g):
    h, w, _ = image.shape

    num_unlabeled = h * w
    num_labels = 0
    label_means = np.zeros((h * w, 3), dtype=np.float32)
    label_sizes = np.zeros((h * w), dtype=np.int64)
    label_map = np.ones((h, w), dtype=np.int64) * -1  # -1 means unlabeled
    candidate_list = np.zeros((h * w, 2), dtype=np.int64)  # (index,coordinates)

    labeled_rows = 0

    while num_unlabeled > 0:
        ## SEARCH FOR UNLABELED PIXEL AS SEED

        found_seed = False
        for sx in range(labeled_rows, h):
            for sy in range(w):
                if label_map[sx, sy] == -1:
                    found_seed = True
                    break
            if found_seed:
                break
            labeled_rows += 1

        new_label = num_labels
        num_labels += 1

        label_map[sx, sy] = new_label
        label_means[new_label] = image[sx, sy, :]
        label_sizes[new_label] = 1

        candidate_list[0, :] = (sx, sy)
        num_candidates = 1
        num_unlabeled -= 1

        while num_candidates > 0:
            num_candidates -= 1
            cx, cy = candidate_list[num_candidates, :]  # pop last candidate
            cg = image[cx, cy, :]

            valid_neighbours = get_valid_neighbours(h, w, cx, cy)
            num_neighbours = valid_neighbours.shape[0]

            for i in range(num_neighbours):
                nx, ny = valid_neighbours[i, :]
                nl = label_map[nx, ny]
                if nl >= 0:
                    continue  # if neighbour is already labeled

                if np.linalg.norm(label_means[new_label] - cg) <= h_g:
                    label_map[nx, ny] = new_label
                    old_size = label_sizes[new_label]
                    new_size = old_size + 1
                    label_means[new_label] = (image[nx, ny, :] + label_means[new_label] * old_size) / new_size
                    label_sizes[new_label] = new_size

                    candidate_list[num_candidates, :] = (nx, ny)
                    num_candidates += 1
                    num_unlabeled -= 1

    return label_map


@jit(nopython=True)
def extend_map_2D(I, n):
    h, w = I.shape
    s = n

    new_shape = (h + 2 * s, w + 2 * s)
    out_image = np.zeros(new_shape, dtype=np.int64)

    out_image[s:h + s, s:w + s] = np.copy(I)

    out_image[:s, :s] = np.ones((s, s)) * I[0, 0]
    out_image[:s, s + w:] = np.ones((s, s)) * I[0, -1]
    out_image[s + h:, :s] = np.ones((s, s)) * I[-1, 0]
    out_image[s + h:, s + w:] = np.ones((s, s)) * I[-1, -1]

    for x in range(h):
        target_row = s + x
        value_left = I[x, 0]
        value_right = I[x, -1]
        for y in range(s):
            out_image[target_row, y] = value_left
            out_image[target_row, y - s] = value_right

    for y in range(w):
        target_column = s + y
        value_up = I[0, y]
        value_low = I[-1, y]
        for x in range(s):
            out_image[x, target_column] = value_up
            out_image[x - s, target_column] = value_low

    return out_image


@jit(nopython=True)
def region_growing_lab_hsv(I, max_dist, min_pixels=1, smoothing=0):
    """Performs region growing on an color image in HSV space.

        Parameters
        ----------
        I : ndarray of float32
            3D array representing the image
        max_dist : float
            maximum hsv distance to add a neighbour to a region [ >= 0.0]
        min_pixels : int
            minimum size of a region [ > 0]
        smoothing : int
            size of smoothing distance in pixels [ >= 0]

        Returns
        -------
        out : ndarray of int64
            region map IDs continuous starting from 0

        Notes
        -----
        The hsv distance is a experimental metric which weights the hue distance,
        depending on the mean saturation and value of a pixel and region mean.
        When smoothing is applied ( > 0), the minimum pixel size is no longer guaranteed!
    """

    max_dist_sq = max_dist ** 2
    h, w, d = I.shape
    assert d == 3, "Only color images supported!"
    HSV = rgb2hsv(I)

    S = np.ones((h, w), dtype=np.int64) * -1
    seeds = [(0, 0)][1:]
    for x in range(h):
        for y in range(w):
            seeds.append((x, y))
    current_seg_id = -1
    while len(seeds) > 0:
        sx, sy = seeds.pop()
        if S[sx, sy] >= 0:
            continue
        current_seg_id += 1
        S[sx, sy] = current_seg_id
        members = [(sx, sy)]
        sum_colors = np.copy(HSV[sx, sy])
        num_pixels = 1
        nearest_segment_distance = -1
        nearest_seg_id = -1
        mean_colors = np.copy(sum_colors)  # / num_pixels

        ### GROWING

        added_members = True
        while added_members:
            added_members = False
            for mx, my in members:
                neighbours = get_valid_neighbours(h, w, mx, my)
                for nx, ny in neighbours:
                    ncol = HSV[nx, ny]
                    ns = S[nx, ny]

                    dist = ncol - mean_colors
                    if dist[0] > 180:
                        dist[0] -= 360
                    elif dist[0] < -180:
                        dist[0] += 360

                    dist[0] *= (ncol[1] + mean_colors[1] + ncol[2] + mean_colors[2]) / (4 * 180)

                    g_dist = np.sum(np.square(dist))
                    if ns != current_seg_id and ns >= 0 and (
                            g_dist < nearest_segment_distance or nearest_seg_id == -1):
                        nearest_segment_distance = g_dist
                        nearest_seg_id = ns

                    if ns < 0 and g_dist < max_dist_sq:
                        S[nx, ny] = current_seg_id
                        num_pixels += 1

                        sum_colors += ncol
                        mean_colors = sum_colors / num_pixels
                        if mean_colors[0] < 0:
                            mean_colors[0] += 360
                        elif mean_colors[0] > 360:
                            mean_colors[0] -= 360
                        members.append((nx, ny))
                        added_members = True

        ### MIN SIZE TEST

        if num_pixels < min_pixels:
            for x in range(h):
                for y in range(w):
                    if S[x, y] == current_seg_id:
                        S[x, y] = nearest_seg_id
            if nearest_seg_id == -1:
                seeds.reverse()
                seeds.append((x, y))
                seeds.reverse()
            current_seg_id -= 1

    # ====== SMOOTHING =========

    if smoothing > 0:
        n = smoothing
        if n % 2 == 0:
            n += 1
        R = np.zeros((h, w, 1), dtype=np.float32)
        hfs = n // 2
        S_ext = extend_map_2D(S, hfs)
        h, w = S_ext.shape
        for x in range(0, h - 2 * hfs):
            for y in range(0, w - 2 * hfs):
                area = S_ext[x:x + n, y:y + n]
                R[x, y] = np.argmax(np.bincount(area.flatten()))
        S = connected_components(R)

    return S


# =========== MEAN SHIFT =================

@jit(nopython=True)
def mean_shift_gray(image, h_g):
    h, w, _ = image.shape
    f = -1.0 / (2.0 * h_g * h_g)

    num_modes = h * w
    modes = np.ones((num_modes, 2), dtype=np.float32)  # gray value, count

    ### INITIALIZE MODES (MERGE CLOSE)

    m_new = 0
    for x in range(h):
        for y in range(w):
            exists = False
            m_new_value = image[x, y, 0]
            m_new_count = 1

            for m_old in range(m_new):
                m_old_value, m_old_count = modes[m_old, :]
                if abs(m_new_value - m_old_value) < h_g:
                    ## MERGE
                    sum_count = (m_old_count + m_new_count)
                    modes[m_old, 0] = (m_old_value * m_old_count + m_new_value * m_new_count) / sum_count
                    modes[m_old, 1] = sum_count

                    exists = True
                    break

            if not exists:
                modes[m_new, 0] = m_new_value
                modes[m_new, 1] = m_new_count
                m_new += 1
    num_modes = m_new

    ### ITERATE

    for i in range(10000):
        mode_changed = False
        print(i, num_modes)

        ## MERGE CLOSE MODES
        m1_index = 0
        while m1_index < num_modes - 1:
            m1_value, m1_count = modes[m1_index, :]
            m2_index = m1_index + 1
            while m2_index < num_modes:
                m2_value, m2_count = modes[m2_index, :]
                if abs(m1_value - m2_value) > h_g:
                    m2_index += 1
                    continue

                # MERGE
                sum_count = (m1_count + m2_count)
                modes[m1_index, 0] = (m1_value * m1_count + m2_value * m2_count) / sum_count
                modes[m1_index, 1] = sum_count

                # REPLACE m2 WITH LAST MODE
                modes[m2_index, :] = modes[num_modes - 1, :]
                num_modes -= 1

                mode_changed = True

            m1_index += 1

        ## GET NEIGHBOURS AND UPDATE MODES
        for m1_index in range(num_modes):
            m1_value = modes[m1_index, 0]
            nomin = 0.0
            denom = 0.0
            for m2_index in range(num_modes):
                m2_value, m2_count = modes[m2_index, :]
                dist = abs(m1_value - m2_value)
                if dist > 3 * h_g:
                    continue
                weight = np.exp(f * dist * dist) * m2_count
                nomin += m2_value * weight
                denom += weight

            new_value = nomin / denom
            if new_value != m1_value:
                mode_changed = True
                modes[m1_index, 0] = new_value

        if not mode_changed:
            break

    ### ASSIGN PIXELS

    for x in range(h):
        for y in range(w):
            gv = image[x, y, 0]
            nearest_dist = 1000000.0
            nearest_mode = 0
            for m in range(num_modes):
                mv = modes[m, 0]
                dist = abs(gv - mv)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_mode = m
            image[x, y, 0] = modes[nearest_mode, 0]

    return image


@jit(nopython=True)
def mean_shift_gray_space(image, h_g, h_s):
    nominator = np.zeros((3), dtype=np.float32)
    p_value = np.zeros((3), dtype=np.float32)
    h, w, _ = image.shape
    f_g = -1.0 / (2.0 * h_g * h_g)
    f_s = -1.0 / (2.0 * h_s * h_s)

    h_s_sq = h_s * h_s

    num_modes = h * w
    modes = np.ones((num_modes, 4), dtype=np.float32)  # gray value,x,y, count

    ### INITIALIZE MODES (MERGE CLOSE)

    m_new_value = np.zeros((3), dtype=np.float32)
    m_new = 0
    for x in range(h):
        for y in range(w):
            exists = False
            m_new_value[0] = image[x, y, 0]
            m_new_value[1] = x
            m_new_value[2] = y
            m_new_count = 1

            for m_old in range(m_new):
                m_old_value = modes[m_old, :3]
                m_old_count = modes[m_old, 3]
                d_g = m_new_value[0] - m_old_value[0]

                if abs(d_g) < h_g:
                    d_xy = m_new_value[1:3] - m_old_value[1:3]
                    d_s_sq = np.sum(d_xy * d_xy)
                    d_s = np.sqrt(d_s_sq)
                    if d_s < h_s:
                        ## MERGE
                        sum_count = (m_old_count + m_new_count)
                        modes[m_old, :3] = (m_old_value * m_old_count + m_new_value * m_new_count) / sum_count
                        modes[m_old, 3] = sum_count

                        exists = True
                        break

            if not exists:
                modes[m_new, :3] = m_new_value
                modes[m_new, 3] = m_new_count
                m_new += 1
    num_modes = m_new

    ### ITERATE

    for i in range(10000):
        modes_changed = False
        print(i, num_modes)

        ### MERGE CLOSE MODES
        m1_index = 0
        while m1_index < num_modes - 1:
            m1_value = modes[m1_index, :3]
            m1_count = modes[m1_index, 3]
            m2_index = m1_index + 1
            while m2_index < num_modes:
                m2_value = modes[m2_index, :3]
                m2_count = modes[m2_index, 3]
                d_g = m1_value[0] - m2_value[0]
                if abs(d_g) > h_g:
                    m2_index += 1
                    continue

                d_xy = m2_value[1:3] - m1_value[1:3]
                d_s_sq = np.sum(d_xy * d_xy)
                if d_s_sq > h_s_sq:
                    m2_index += 1
                    continue

                ## MERGE
                sum_count = (m1_count + m2_count)
                modes[m1_index, :3] = (m1_value * m1_count + m2_value * m2_count) / sum_count
                modes[m1_index, 3] = sum_count

                ## REPLACE m2 WITH LAST MODE
                modes[m2_index, :] = modes[num_modes - 1, :]
                num_modes -= 1

                modes_changed = True

            m1_index += 1

        ### GET NEIGHBOURS AND UPDATE MODES
        for m1_index in range(num_modes):
            m1_value = modes[m1_index, :3]
            nominator *= 0.0
            denom = 0.0
            for m2_index in range(num_modes):
                m2_value = modes[m2_index, :3]
                m2_count = modes[m2_index, 3]
                d_g = m1_value[0] - m2_value[0]
                if abs(d_g) > 3 * h_g:
                    continue
                d_xy = m2_value[1:3] - m1_value[1:3]
                d_s_sq = np.sum(d_xy * d_xy)
                d_s = np.sqrt(d_s_sq)
                if d_s > 3 * h_s:
                    continue
                ## UPDATE
                weight = np.exp(f_g * d_g * d_g) * np.exp(f_s * d_s_sq) * m2_count
                nominator += m2_value * weight
                denom += weight

            new_value = nominator / denom
            if new_value[0] != m1_value[0] or new_value[1] != m1_value[1] or new_value[2] != m1_value[2]:
                modes_changed = True
                modes[m1_index, :3] = new_value

        if not modes_changed:
            break

    ### ASSIGN PIXELS

    for x in range(h):
        for y in range(w):
            p_value[:] = (image[x, y, 0], float(x), float(y))
            best_weight = 0
            best_mode = 0
            for m in range(num_modes):
                dg, dx, dy = p_value[:] - modes[m, :3]
                dist_s_sq = dx ** 2 + dy ** 2
                dist_g_sq = dg * dg
                weight_c = np.exp(f_g * dist_g_sq)
                weight_s = np.exp(f_s * dist_s_sq)
                weight = weight_c * weight_s
                if weight > best_weight or m == 0:
                    best_weight = weight
                    best_mode = m
            image[x, y, 0] = modes[best_mode, 0]

    return image


@jit(nopython=True)
def mean_shift_color(image, h_c):
    nominator = np.zeros((3), dtype=np.float32)
    p_value = np.zeros((3), dtype=np.float32)
    h, w, _ = image.shape
    f_c = -1.0 / (2.0 * h_c * h_c)

    h_c_sq = h_c * h_c

    num_modes = h * w
    modes = np.ones((num_modes, 4), dtype=np.float32)  # r,g,b, count

    ### INITIALIZE MODES (MERGE CLOSE)

    m_new_value = np.zeros((3), dtype=np.float32)
    m_new = 0
    for x in range(h):
        for y in range(w):
            exists = False
            m_new_value[:] = image[x, y, :]
            m_new_count = 1

            for m_old in range(m_new):
                m_old_value = modes[m_old, :3]
                m_old_count = modes[m_old, 3]

                d_rgb = m_new_value[1:3] - m_old_value[1:3]
                d_rgb_sq = np.sum(d_rgb * d_rgb)
                d_c = np.sqrt(d_rgb_sq)
                if d_c < h_c:
                    ## MERGE
                    sum_count = (m_old_count + m_new_count)
                    modes[m_old, :3] = (m_old_value * m_old_count + m_new_value * m_new_count) / sum_count
                    modes[m_old, 3] = sum_count

                    exists = True
                    break

            if not exists:
                modes[m_new, :3] = m_new_value
                modes[m_new, 3] = m_new_count
                m_new += 1
    num_modes = m_new

    ### ITERATE

    for i in range(10000):
        modes_changed = False
        print(i, num_modes)

        ### MERGE CLOSE MODES
        m1_index = 0
        while m1_index < num_modes - 1:
            m1_value = modes[m1_index, :3]
            m1_count = modes[m1_index, 3]
            m2_index = m1_index + 1
            while m2_index < num_modes:
                m2_value = modes[m2_index, :3]
                m2_count = modes[m2_index, 3]

                d_rgb = m2_value[:3] - m1_value[:3]
                d_rgb_sq = np.sum(d_rgb * d_rgb)
                if d_rgb_sq > h_c_sq:
                    m2_index += 1
                    continue

                ## MERGE
                sum_count = (m1_count + m2_count)
                modes[m1_index, :3] = (m1_value * m1_count + m2_value * m2_count) / sum_count
                modes[m1_index, 3] = sum_count

                ## REPLACE m2 WITH LAST MODE
                modes[m2_index, :] = modes[num_modes - 1, :]
                num_modes -= 1

                modes_changed = True

            m1_index += 1

        ### GET NEIGHBOURS AND UPDATE MODES
        for m1_index in range(num_modes):
            m1_value = modes[m1_index, :3]
            nominator *= 0.0
            denom = 0.0
            for m2_index in range(num_modes):
                m2_value = modes[m2_index, :3]
                m2_count = modes[m2_index, 3]

                d_rgb = m2_value[:3] - m1_value[:3]
                d_rgb_sq = np.sum(d_rgb * d_rgb)
                d_c = np.sqrt(d_rgb_sq)
                if d_c > 3 * h_c:
                    continue
                ## UPDATE
                weight = np.exp(f_c * d_c * d_c) * m2_count
                nominator += m2_value * weight
                denom += weight

            new_value = nominator / denom
            if new_value[0] != m1_value[0] or new_value[1] != m1_value[1] or new_value[2] != m1_value[2]:
                modes_changed = True
                modes[m1_index, :3] = new_value

        if not modes_changed:
            break

    ### ASSIGN PIXELS

    for x in range(h):
        for y in range(w):
            p_value[:] = image[x, y, :]
            best_weight = 0
            best_mode = 0
            for m in range(num_modes):
                d_rgb = p_value[:] - modes[m, :3]
                dist_c_sq = np.sum(d_rgb * d_rgb)
                weight = np.exp(f_c * dist_c_sq)
                if weight > best_weight or m == 0:
                    best_weight = weight
                    best_mode = m
            image[x, y, :] = modes[best_mode, :3]

    return image


@jit(nopython=True)
def mean_shift_color_space(image, h_c, h_s):
    nominator = np.zeros((5), dtype=np.float32)
    p_value = np.zeros((5), dtype=np.float32)
    h, w, _ = image.shape
    f_c = -1.0 / (2.0 * h_c * h_c)
    f_s = -1.0 / (2.0 * h_s * h_s)

    h_c_sq = h_c * h_c
    h_s_sq = h_s * h_s

    num_modes = h * w
    modes = np.ones((num_modes, 6), dtype=np.float32)  # gray value,x,y, count

    ### INITIALIZE MODES (MERGE CLOSE)

    m_new_value = np.zeros((5), dtype=np.float32)
    m_new = 0
    for x in range(h):
        for y in range(w):
            exists = False
            m_new_value[0] = image[x, y, 0]
            m_new_value[1] = image[x, y, 1]
            m_new_value[2] = image[x, y, 2]
            m_new_value[3] = x
            m_new_value[4] = y
            m_new_count = 1

            for m_old in range(m_new):
                m_old_value = modes[m_old, :5]
                m_old_count = modes[m_old, 5]

                d_rgb = m_new_value[:3] - m_old_value[:3]
                d_c_sq = np.sum(d_rgb * d_rgb)

                if d_c_sq < h_s_sq:
                    d_xy = m_new_value[3:5] - m_old_value[3:5]
                    d_s_sq = np.sum(d_xy * d_xy)
                    if d_s_sq < h_s_sq:
                        ## MERGE
                        sum_count = (m_old_count + m_new_count)
                        modes[m_old, :5] = (m_old_value * m_old_count + m_new_value * m_new_count) / sum_count
                        modes[m_old, 5] = sum_count

                        exists = True
                        break

            if not exists:
                modes[m_new, :5] = m_new_value
                modes[m_new, 5] = m_new_count
                m_new += 1
    num_modes = m_new

    ### ITERATE

    for i in range(10000):
        modes_changed = False
        print(i, num_modes)

        ### MERGE CLOSE MODES
        m1_index = 0
        while m1_index < num_modes - 1:
            m1_value = modes[m1_index, :5]
            m1_count = modes[m1_index, 5]
            m2_index = m1_index + 1
            while m2_index < num_modes:
                m2_value = modes[m2_index, :5]
                m2_count = modes[m2_index, 5]

                d_rgb = m2_value[:3] - m1_value[:3]
                d_c_sq = np.sum(d_rgb * d_rgb)
                if d_c_sq > h_c_sq:
                    m2_index += 1
                    continue

                d_xy = m2_value[3:5] - m1_value[3:5]
                d_s_sq = np.sum(d_xy * d_xy)
                if d_s_sq > h_s_sq:
                    m2_index += 1
                    continue

                ## MERGE
                sum_count = (m1_count + m2_count)
                modes[m1_index, :5] = (m1_value * m1_count + m2_value * m2_count) / sum_count
                modes[m1_index, 5] = sum_count

                ## REPLACE m2 WITH LAST MODE
                modes[m2_index, :] = modes[num_modes - 1, :]
                num_modes -= 1

                modes_changed = True

            m1_index += 1

        ### GET NEIGHBOURS AND UPDATE MODES
        for m1_index in range(num_modes):
            m1_value = modes[m1_index, :5]
            nominator *= 0.0
            denom = 0.0
            for m2_index in range(num_modes):
                m2_value = modes[m2_index, :5]
                m2_count = modes[m2_index, 5]

                d_rgb = m2_value[:3] - m1_value[:3]
                d_c_sq = np.sum(d_rgb * d_rgb)
                d_c = np.sqrt(d_c_sq)
                if d_c > 3 * h_c:
                    continue

                d_xy = m2_value[3:5] - m1_value[3:5]
                d_s_sq = np.sum(d_xy * d_xy)
                d_s = np.sqrt(d_s_sq)
                if d_s > 3 * h_s:
                    continue

                ## UPDATE
                weight = np.exp(f_c * d_c_sq) * np.exp(f_s * d_s_sq) * m2_count
                nominator += m2_value * weight
                denom += weight

            new_value = nominator / denom
            if new_value[0] != m1_value[0] or new_value[1] != m1_value[1] or new_value[2] != m1_value[2] \
                    or new_value[3] != m1_value[3] or new_value[4] != m1_value[4]:
                modes_changed = True
                modes[m1_index, :5] = new_value

        if not modes_changed:
            break

    ### ASSIGN PIXELS

    for x in range(h):
        for y in range(w):
            p_value[:] = (image[x, y, 0], image[x, y, 1], image[x, y, 2], float(x), float(y))
            best_weight = 0
            best_mode = 0
            for m in range(num_modes):
                dr, dg, db, dx, dy = p_value[:] - modes[m, :5]
                dist_c_sq = dr * dr + dg * dg + db * db
                dist_s_sq = dx * dx + dy * dy
                weight_c = np.exp(f_c * dist_c_sq)
                weight_s = np.exp(f_s * dist_s_sq)
                weight = weight_c * weight_s
                if weight > best_weight or m == 0:
                    best_weight = weight
                    best_mode = m
            image[x, y, :] = modes[best_mode, :3]

    return image


# ============= SUPER PIXELS ==================

@jit(nopython=True)
def slic(image, K, m, E):
    # K = number of superpixels
    # m = compactness

    h, w, d = image.shape
    N = h * w
    S = (N / K) ** 0.5

    lab_image = rgb2lab(image)
    label_map = np.zeros((h, w), dtype=np.int64)

    num_clusters = int((h // S + 1) * (w // S + 1))
    print(num_clusters)
    # S = (N / num_clusters) ** 0.5
    # print(S)

    c = 0

    avgs = np.zeros((num_clusters, 5), dtype=np.float32)
    num_pixels = np.zeros((num_clusters), dtype=np.int64)
    clusters = np.zeros((num_clusters, 5), dtype=np.float32)  # x,y,l,a,b
    xc = 0
    while xc <= h:
        yc = 0
        while yc <= w:
            L, a, b = lab_image[int(round(xc)), int(round(yc)), :]
            clusters[c, :] = (xc, yc, L, a, b)
            c += 1
            yc += S
        xc += S

    i = 0
    while True:
        i += 1
        avgs *= 0
        num_pixels *= 0
        for x in range(h):
            for y in range(w):
                nearest_cluster = 0
                nearest_distance = 9999999
                L, a, b = lab_image[x, y, :]
                for c in range(num_clusters):
                    xc, yc, Lc, ac, bc = clusters[c, :]
                    dx = x - xc
                    dy = y - yc
                    if abs(dx) > S or abs(dy) > S:
                        continue

                    dL = L - Lc
                    da = a - ac
                    db = b - bc

                    Dxy = (dx * dx + dy * dy) ** 0.5
                    # Dxy = abs(dx) + abs(dy)
                    Dlab = (dL * dL + da * da + db * db) ** 0.5
                    Ds = Dlab + m / S * Dxy

                    if Ds < nearest_distance:
                        nearest_distance = Ds
                        nearest_cluster = c

                avgs[nearest_cluster, 0] += x
                avgs[nearest_cluster, 1] += y
                avgs[nearest_cluster, 2] += L
                avgs[nearest_cluster, 3] += a
                avgs[nearest_cluster, 4] += b

                num_pixels[nearest_cluster] += 1
                label_map[x, y] = nearest_cluster

        clusters_pre = np.copy(clusters)
        for c in range(num_clusters):
            clusters[c] = avgs[c, :] / num_pixels[c]
        max_diff = np.max(np.sum(np.abs(clusters - clusters_pre), 1))

        if max_diff < E:
            break

    return label_map


# ============= POST PROCESSING ===============

@jit(nopython=True)
def connected_components(image):
    h, w, d = image.shape

    ### MAP COORDINATES TO HORIZONTAL LABEL
    label_map_horizontal = np.zeros((h, w), dtype=np.int64)
    next_label = 1
    for x in range(h):
        for y in range(w):
            merge_h = y > 0
            if merge_h:
                for z in range(d):
                    if image[x, y - 1, z] != image[x, y, z]:
                        merge_h = False
                        break

            if merge_h:
                label_map_horizontal[x, y] = label_map_horizontal[x, y - 1]

            else:
                merge_v = x > 0
                if merge_v:
                    for z in range(d):
                        if image[x - 1, y, z] != image[x, y, z]:
                            merge_v = False
                            break

                if merge_v:
                    label_map_horizontal[x, y] = label_map_horizontal[x - 1, y]

                else:
                    label_map_horizontal[x, y] = next_label
                    next_label += 1

    ### MAP HORIZONTAL INDICES TO FINAL INDICES
    final_labels_length = next_label
    final_labels = np.ones((final_labels_length), dtype=np.int64) * -1
    next_final_label = 0
    for x in range(h):
        for y in range(w):
            try_merge = x > 0
            h_label = label_map_horizontal[x, y]

            merge = try_merge
            if try_merge:
                upper_h_label = label_map_horizontal[x - 1, y]
                if h_label == upper_h_label or final_labels[h_label] == final_labels[upper_h_label]:
                    merge = False
                else:
                    for z in range(d):
                        if image[x - 1, y, z] != image[x, y, z]:
                            merge = False
                            break
                if merge:
                    if final_labels[h_label] == -1:
                        final_labels[h_label] = final_labels[upper_h_label]
                    elif final_labels[upper_h_label] != final_labels[h_label]:
                        to_change_final = final_labels[h_label]
                        for i in range(final_labels_length):
                            if final_labels[i] == to_change_final:
                                final_labels[i] = final_labels[upper_h_label]

            if not merge and final_labels[h_label] == -1:
                final_labels[h_label] = next_final_label
                next_final_label += 1

    ### MAP FINAL INDICES TO INDICES WITHOUT GAPS (STARTING FROM ZERO)
    no_gap_labels_length = next_final_label
    no_gap_labels = np.ones((no_gap_labels_length), dtype=np.int64) * -1
    no_gap_index = 0
    for x in range(h):
        for y in range(w):
            if no_gap_labels[final_labels[label_map_horizontal[x, y]]] == -1:
                no_gap_labels[final_labels[label_map_horizontal[x, y]]] = no_gap_index
                no_gap_index += 1

    label_map = np.zeros((h, w), dtype=np.int64)
    for x in range(h):
        for y in range(w):
            label_map[x, y] = no_gap_labels[final_labels[label_map_horizontal[x, y]]]

    return label_map

@jit(nopython=True)
def get_neighbouring_segments(S):
    """Return adjacency matrix.

        Parameters
        ----------
        S : ndarray of int64
            2D array representing the segment map

        Returns
        -------
        out : ndarray of bool_
            2D array, entry at i,j = True if segments i and j are neighbours
    """
    h, w = S.shape
    num_s = np.max(S) + 1
    neighbour_matrix = np.zeros((num_s, num_s), dtype=np.bool_)
    for x in range(h):
        for y in range(w):
            s_id = S[x, y]
            N = get_valid_neighbours(h, w, x, y)
            for n in range(N.shape[0]):
                nx, ny = N[n,:]
                n_id = S[nx, ny]
                neighbour_matrix[s_id, n_id] = True
                neighbour_matrix[n_id, s_id] = True
    return neighbour_matrix

@jit(nopython=True, cache=True)
def get_contour_image(S):
    """Computes a contour ID map from a segment map.

        Parameters
        ----------
        S : ndarray of int64
            2D array representing the segment map

        Returns
        -------
        out : ndarray of int64
            2D array representing the contour map

        Notes
        -----
        Each entry of the contour map corresponds to the component ID,
        if the entry it is on the contour of that component, else -1.
        The contour map is computed via contour tracing.
    """

    h, w = S.shape
    num_s = np.max(S) + 1
    C = np.ones((h, w), dtype=np.int64) * -1

    for s in range(num_s):

        # GET CONTOUR SEED FOR CLASS 'c'
        start_pixel = (-1, -1)
        for x in range(h):
            for y in range(w):
                if S[x, y] == s:
                    start_pixel = (x, y)
                    break
            if start_pixel[0] >= 0:
                break

        if start_pixel == (-1, -1):
            continue

        # FOLLOW CONTOUR OF SEGMENT 's'
        search_dir = 0
        current_pixel = start_pixel
        done = False
        while True:
            cpx, cpy = current_pixel
            C[cpx, cpy] = s
            for i in range(8):  # while(True) failes with jit..
                if search_dir == 0 and cpx > 0 and S[cpx - 1, cpy] == s:  # start at pixel above
                    current_pixel = (cpx - 1, cpy)
                    break
                elif search_dir == 45 and cpx > 0 and cpy < w - 1 and S[cpx - 1, cpy + 1] == s:
                    current_pixel = (cpx - 1, cpy + 1)
                    break
                elif search_dir == 90 and cpy < w - 1 and S[cpx, cpy + 1] == s:
                    current_pixel = (cpx, cpy + 1)
                    break
                elif search_dir == 135 and cpx < h - 1 and cpy < w - 1 and S[cpx + 1, cpy + 1] == s:
                    current_pixel = (cpx + 1, cpy + 1)
                    break
                elif search_dir == 180 and cpx < h - 1 and S[cpx + 1, cpy] == s:
                    current_pixel = (cpx + 1, cpy)
                    break
                elif search_dir == 225 and cpx < h - 1 and cpy > 0 and S[cpx + 1, cpy - 1] == s:
                    current_pixel = (cpx + 1, cpy - 1)
                    break
                elif search_dir == 270 and cpy > 0 and S[cpx, cpy - 1] == s:
                    current_pixel = (cpx, cpy - 1)
                    break
                elif search_dir == 315 and cpx > 0 and cpy > 0 and S[cpx - 1, cpy - 1] == s:
                    current_pixel = (cpx - 1, cpy - 1)
                    break
                search_dir = (search_dir + 45) % 360
                if search_dir == 0 and start_pixel == current_pixel:
                    done = True
                    break
            search_dir = (search_dir + 270) % 360

            if done:
                break
    return C

# only 2 neighbours
def get_border_pixels(C):
    h, w = C.shape
    num_classes = np.max(C)
    B = np.zeros((h, w), dtype=np.int64)

    for c in range(1, num_classes):

        # GET CONTOUR SEED FOR CLASS 'c'
        start_pixel = (-1, -1)
        for x in range(h):
            for y in range(w):
                if C[x, y] == c:
                    start_pixel = (x, y)
                    break
            if start_pixel[0] >= 0:
                break

        if start_pixel == (-1, -1):
            continue

        # FOLLOW CONTOUR OF CLASS 'c'
        search_dir = 0
        current_pixel = start_pixel
        while True:
            cpx, cpy = current_pixel
            B[cpx, cpy] = c
            while True:
                if search_dir == 0 and cpx > 0 and C[cpx - 1, cpy] == c:
                    current_pixel = (cpx - 1, cpy)
                    break
                elif search_dir == 45 and cpx > 0 and cpy < w - 1 and C[cpx - 1, cpy + 1] == c:
                    current_pixel = (cpx - 1, cpy + 1)
                    break
                elif search_dir == 90 and cpy < w - 1 and C[cpx, cpy + 1] == c:
                    current_pixel = (cpx, cpy + 1)
                    break
                elif search_dir == 135 and cpx < h - 1 and cpy < w - 1 and C[cpx + 1, cpy + 1] == c:
                    current_pixel = (cpx + 1, cpy + 1)
                    break
                elif search_dir == 180 and cpx < h - 1 and C[cpx + 1, cpy] == c:
                    current_pixel = (cpx + 1, cpy)
                    break
                elif search_dir == 225 and cpx < h - 1 and cpy > 0 and C[cpx + 1, cpy - 1] == c:
                    current_pixel = (cpx + 1, cpy - 1)
                    break
                elif search_dir == 270 and cpy > 0 and C[cpx, cpy - 1] == c:
                    current_pixel = (cpx, cpy - 1)
                    break
                elif search_dir == 315 and cpx > 0 and cpy > 0 and C[cpx - 1, cpy - 1] == c:
                    current_pixel = (cpx - 1, cpy - 1)
                    break
                search_dir = (search_dir + 45) % 360
            search_dir = (search_dir + 270) % 360

            if current_pixel == start_pixel:
                break
    return B


@jit(nopython=True)
def get_valid_neighbours(h, w, x, y, neighbourhood = 8):
    has_upper = x > 0
    has_lower = x < h - 1
    has_left = y > 0
    has_right = y < w - 1

    neighbours = np.zeros((8, 2), dtype=np.int64)
    neighbour_index = 0

    if neighbourhood == 8:

        if has_upper:
            neighbours[neighbour_index, :] = (x - 1, y)
            neighbour_index += 1
        if has_lower:
            neighbours[neighbour_index, :] = (x + 1, y)
            neighbour_index += 1
        if has_left:
            neighbours[neighbour_index, :] = (x, y - 1)
            neighbour_index += 1
            if has_upper:
                neighbours[neighbour_index, :] = (x - 1, y - 1)
                neighbour_index += 1
            if has_lower:
                neighbours[neighbour_index, :] = (x + 1, y - 1)
                neighbour_index += 1
        if has_right:
            neighbours[neighbour_index, :] = (x, y + 1)
            neighbour_index += 1
            if has_upper:
                neighbours[neighbour_index, :] = (x - 1, y + 1)
                neighbour_index += 1
            if has_lower:
                neighbours[neighbour_index, :] = (x + 1, y + 1)
                neighbour_index += 1

        # if has_right and has_upper:
        #     neighbours[neighbour_index, :] = (x - 1, y + 1)
        #     neighbour_index += 1
        # if has_left and has_upper:
        #     neighbours[neighbour_index, :] = (x - 1, y - 1)
        #     neighbour_index += 1
        # if has_left and has_lower:
        #     neighbours[neighbour_index, :] = (x + 1, y - 1)
        #     neighbour_index += 1
        # if has_right and has_lower:
        #     neighbours[neighbour_index, :] = (x + 1, y + 1)
        #     neighbour_index += 1

    elif neighbourhood == 4:

        if has_upper:
            neighbours[neighbour_index, :] = (x - 1, y)
            neighbour_index += 1
        if has_lower:
            neighbours[neighbour_index, :] = (x + 1, y)
            neighbour_index += 1
        if has_left:
            neighbours[neighbour_index, :] = (x, y - 1)
            neighbour_index += 1
        if has_right:
            neighbours[neighbour_index, :] = (x, y + 1)
            neighbour_index += 1

    return neighbours[:neighbour_index, :]


# n neighbours
@jit(nopython=True)
def get_border_pixels_dense(C):
    h, w = C.shape
    num_classes = np.max(C)
    B = np.zeros((h, w), dtype=np.int64)

    for x in range(h):
        for y in range(w):
            c = C[x, y]
            neighbours = get_valid_neighbours(h, w, x, y)
            if len(neighbours) < 8:
                B[x, y] = c
                continue
            for nx, ny in neighbours:
                if C[nx, ny] != c:
                    B[x, y] = c
                    break

    return B


def get_perimeters(C):
    w2 = 2 ** 0.5

    h, w = C.shape
    num_classes = np.max(C)
    perimeters = np.zeros(num_classes, dtype=np.float32)

    for c in range(1, num_classes):

        # GET CONTOUR SEED FOR CLASS 'c'
        start_pixel = (-1, -1)
        for x in range(h):
            for y in range(w):
                if C[x, y] == c:
                    start_pixel = (x, y)
                    break
            if start_pixel[0] >= 0:
                break

        if start_pixel == (-1, -1):
            continue

        # FOLLOW CONTOUR OF CLASS 'c'
        search_dir = 0
        current_pixel = start_pixel
        while True:
            cpx, cpy = current_pixel
            while True:
                if search_dir == 0 and cpx > 0 and C[cpx - 1, cpy] == c:
                    current_pixel = (cpx - 1, cpy)
                    perimeters[c] += 1
                    break
                elif search_dir == 45 and cpx > 0 and cpy < w - 1 and C[cpx - 1, cpy + 1] == c:
                    current_pixel = (cpx - 1, cpy + 1)
                    perimeters[c] += w2
                    break
                elif search_dir == 90 and cpy < w - 1 and C[cpx, cpy + 1] == c:
                    current_pixel = (cpx, cpy + 1)
                    perimeters[c] += 1
                    break
                elif search_dir == 135 and cpx < h - 1 and cpy < w - 1 and C[cpx + 1, cpy + 1] == c:
                    current_pixel = (cpx + 1, cpy + 1)
                    perimeters[c] += w2
                    break
                elif search_dir == 180 and cpx < h - 1 and C[cpx + 1, cpy] == c:
                    current_pixel = (cpx + 1, cpy)
                    perimeters[c] += 1
                    break
                elif search_dir == 225 and cpx < h - 1 and cpy > 0 and C[cpx + 1, cpy - 1] == c:
                    current_pixel = (cpx + 1, cpy - 1)
                    perimeters[c] += w2
                    break
                elif search_dir == 270 and cpy > 0 and C[cpx, cpy - 1] == c:
                    current_pixel = (cpx, cpy - 1)
                    perimeters[c] += 1
                    break
                elif search_dir == 315 and cpx > 0 and cpy > 0 and C[cpx - 1, cpy - 1] == c:
                    current_pixel = (cpx - 1, cpy - 1)
                    perimeters[c] += w2
                    break
                search_dir = (search_dir + 45) % 360
            search_dir = (search_dir + 270) % 360

            if current_pixel == start_pixel:
                break
    return perimeters


@jit(nopython=True, cache=True)
def get_geometric_segment_features(S):
    """Computes the geometric segment features, based on a segment map.

        Parameters
        ----------
        S : ndarray of int64
            2D array representing a segment map

        Returns
        -------
        out : tuple
            areas, compactnesses, MBRs, fill factors, elongations

        Notes
        -----
        Definitions
        areas : list of float
        compactnesses : list of float
        MBRs : list of tuples of floats : (angle, length, with, center x, center y)
        fill factors : list of float
        elongations : list of float
    """

    h, w = S.shape
    num_s = np.max(S) + 1
    C = get_contour_image(S)

    reg_moms = np.zeros((num_s, 2, 2), dtype=np.int64)
    cen_moms = np.zeros((num_s, 3, 3), dtype=np.int64)
    perimeters = np.zeros((num_s), dtype=np.int64)
    mass_centres = np.zeros((num_s, 2), dtype=np.float64)
    mbrs = np.zeros((num_s, 5), dtype=np.float64)  # phi, L, W, x_, y_

    for x in range(h):
        for y in range(w):
            s = S[x, y]
            reg_moms[s, 0, 0] += 1
            reg_moms[s, 1, 0] += x
            reg_moms[s, 0, 1] += y
            if C[x, y] >= 0:
                perimeters[C[x, y]] += 1

    areas = reg_moms[:, 0, 0]
    mass_centres[:, 0] = reg_moms[:, 1, 0] / areas
    mass_centres[:, 1] = reg_moms[:, 0, 1] / areas

    # cent_moms, neighbours

    for x in range(h):
        for y in range(w):
            s = S[x, y]
            x_ = x - mass_centres[s, 0]
            y_ = y - mass_centres[s, 1]
            cen_moms[s, 1, 1] += x_ * y_
            cen_moms[s, 2, 0] += x_ * x_
            cen_moms[s, 0, 2] += y_ * y_
            # cen_moms[s, .. ] += ..

    comps = np.square(perimeters) / (4 * np.pi * reg_moms[:, 0, 0])

    mbrs[:, 0] = 0.5 * np.arctan2(2 * cen_moms[:, 1, 1], (cen_moms[:, 2, 0] - cen_moms[:, 0, 2]))
    c_phis = np.cos(mbrs[:, 0])
    s_phis = np.sin(mbrs[:, 0])

    ab_min = np.ones((num_s, 2), dtype=np.float64) * h * w
    ab_max = np.ones((num_s, 2), dtype=np.float64) * h * w * -1

    for x in range(h):
        for y in range(w):
            s = C[x, y]
            if s >= 0:
                cp = c_phis[s]
                sp = s_phis[s]
                a = x * cp + y * sp
                b = -x * sp + y * cp

                if a > ab_max[s, 0]:
                    ab_max[s, 0] = a
                if a < ab_min[s, 0]:
                    ab_min[s, 0] = a

                if b > ab_max[s, 1]:
                    ab_max[s, 1] = b
                if b < ab_min[s, 1]:
                    ab_min[s, 1] = b

    ab_mean = (ab_max + ab_min) / 2
    mbrs[:, 3] = ab_mean[:, 0] * c_phis - ab_mean[:, 1] * s_phis
    mbrs[:, 4] = ab_mean[:, 0] * s_phis + ab_mean[:, 1] * c_phis

    mbrs[:, 1:3] = np.abs(ab_max - ab_min)

    fill_factors = areas / (mbrs[:, 1] * mbrs[:, 2])
    elongations = np.abs(np.log(np.abs(mbrs[:, 1] / mbrs[:, 2])))

    return areas, comps, mbrs, fill_factors, elongations


@jit(nopython=True, cache=True)
def get_spectral_segment_features(S, I):
    """Computes the spectral features of all segments.

        Parameters
        ----------
        S : ndarray of int64
            2D array representing a segment map

        I : ndarray of float64
            3D array representing the image

        Returns
        -------
        out : tuple
            mean rgb values, means hsv values

        Notes
        -----
        Definitions
        mean rgb values : list of tuples of floats : (mean red, mean green, mean blue)
        mean hsv values : list of tuples of floats : (mean hue, mean saturation, mean value)
    """
    HSV = rgb2hsv(I)
    num_s = np.max(S) + 1

    means_rgb = np.zeros((num_s, 3), dtype=np.float64)
    means_hsv = np.zeros_like(means_rgb)

    for s in range(num_s):
        M = (S == s).astype(np.float64)
        sum_M = np.sum(M)
        means_rgb[s, :] = (
            np.sum(I[:, :, 0] * M) / sum_M, np.sum(I[:, :, 1] * M) / sum_M, np.sum(I[:, :, 2] * M) / sum_M)
        means_hsv[s, :] = (
            np.sum(HSV[:, :, 0] * M) / sum_M, np.sum(HSV[:, :, 1] * M) / sum_M, np.sum(HSV[:, :, 2] * M) / sum_M)

    return means_rgb, means_hsv


# ============= FILTER BANK CLUSTERING ========

def get_filters(sigma_min, sigma_max):
    sl = sigma_min
    sh = sigma_max
    filters = []

    # gaussians

    for sigma in np.linspace(sl, sh, 4):
        N = int(6 * sigma)
        if N % 2 == 0:
            N += 1
        filter = np.zeros((N, N), dtype=np.float32)
        offset = N / 2 - 0.5
        two_sigma_sq = 2 * sigma * sigma
        for x_ in range(N):
            for y_ in range(N):
                x = x_ - offset
                y = y_ - offset
                G_sigma = 1.0 / (np.pi * two_sigma_sq) * np.e ** (-(x * x + y * y) / (two_sigma_sq))
                filter[x_, y_] = G_sigma
        filters += [filter]

    # first, second derivatives

    for sigma in np.linspace(sl, sh, 3):
        N = int(6 * sigma)
        if N % 2 == 0:
            N += 1
        offset = N / 2 - 0.5
        two_sigma_sq = 2 * sigma * sigma
        two_pi_sigma_4 = 2 * np.pi * sigma ** 4
        two_pi_sigma_6 = 2 * np.pi * sigma ** 6

        for alpha in np.linspace(0.0, np.pi, 6, endpoint=False):
            filter_first = np.zeros((N, N), dtype=np.float32)
            filter_second = np.zeros((N, N), dtype=np.float32)
            sa = np.sin(alpha)
            ca = np.cos(alpha)
            for x_ in range(N):
                for y_ in range(N):
                    xo = x_ - offset
                    yo = y_ - offset
                    x = ca * xo - sa * yo
                    y = sa * xo + ca * yo
                    # first derivative
                    G_sigma = - x / two_pi_sigma_4 * \
                              np.exp(-(x * x + y * y) / two_sigma_sq)
                    filter_first[x_, y_] = G_sigma
                    # second derivative
                    G_sigma = (x - sigma) * (x + sigma) / two_pi_sigma_6 * \
                              np.exp(-(x * x + y * y) / two_sigma_sq)
                    filter_second[x_, y_] = G_sigma

            filters += [filter_first]
            filters += [filter_second]

    # 8 laplacians

    for sigma in np.linspace(sl, sh, 8):
        N = int(6 * sigma)
        if N % 2 == 0:
            N += 1
        filter = np.zeros((N, N), dtype=np.float32)
        offset = N / 2 - 0.5
        f0 = -1 / (np.pi * sigma ** 4)
        two_sigma_sq = 2 * sigma ** 2

        for x_ in range(N):
            for y_ in range(N):
                x = x_ - offset
                y = y_ - offset
                xxyy = x * x + y * y
                G_sigma = f0 * (1 - xxyy / two_sigma_sq) * np.exp(-xxyy / two_sigma_sq)
                filter[x_, y_] = G_sigma
        filters += [filter]

    return filters


@jit(nopython=True)
def cluster_features(features, k, iter, th):
    h, w, num_f = features.shape
    means = np.random.uniform(0, 10, (k, num_f)).astype(np.float32)
    sums = np.zeros_like(means)
    counters = np.zeros((k), dtype=np.int64)

    prev_means = np.zeros_like(means)

    print('clustering..')
    for it in range(iter):
        print(it)
        sums *= 0
        counters *= 0

        for x in range(h):
            for y in range(w):
                k_best = -1
                dist_best = 0.0
                feature_vec = features[x, y, :]

                for ki in range(k):
                    dist = np.sum(np.square(feature_vec - means[ki]))
                    if dist < dist_best or k_best == -1:
                        k_best = ki
                        dist_best = dist

                sums[k_best] += feature_vec
                counters[k_best] += 1

        for ki in range(k):
            if counters[ki] > 0:
                means[ki] = sums[ki] / counters[ki]

        if np.max(np.abs(prev_means - means)) < th:
            break
        prev_means = np.copy(means)

    label_map = np.zeros((h, w), dtype=np.int64)

    for x in range(h):
        for y in range(w):
            k_best = -1
            dist_best = 0.0
            feature_vec = features[x, y, :]

            for ki in range(k):
                dist = np.sum(np.square(feature_vec - means[ki]))
                if dist < dist_best or k_best == -1:
                    k_best = ki
                    dist_best = dist

            label_map[x, y] = k_best

    return label_map


def filter_bank_clustering(image, sigma_min, sigma_max, k, th=0.1):
    h, w, d = image.shape

    filters = get_filters(sigma_min, sigma_max)
    num_f = len(filters)
    print('created {} filters'.format(num_f))
    features = np.zeros((h, w, num_f), dtype=np.float32)
    for i in range(num_f):
        print('convolved with {} / {} filters'.format(i, num_f))
        feature_map = convolution.fast_sw_convolution(image, filters[i], convolution.SAME)
        features[:, :, [i]] = feature_map

    features = np.abs(features)

    return cluster_features(features, k, 100, th)


# ================ SUPPRESSION ====================

@jit(nopython=True)
def get_n_best_3d(candidates, value_matrix, num_best):
    num_candidates = candidates.shape[0]
    if num_candidates <= num_best:
        return np.copy(candidates)

    best_candidates = np.zeros((num_best, 3), dtype=np.int64)
    values = np.zeros((num_candidates), dtype=np.float32)
    for i in range(num_candidates):
        c_x, c_y, c_z = candidates[i, :]
        values[i] = value_matrix[c_x, c_y, c_z]

    for i in range(num_best):
        best_candidate_index = np.argmax(values)
        best_candidates[i, :] = candidates[best_candidate_index, :]
        values[best_candidate_index] = -999999

    return best_candidates


@jit(nopython=True)
def get_n_best_4d(candidates, value_matrix, num_best):
    num_candidates = candidates.shape[0]
    if num_candidates <= num_best:
        return np.copy(candidates)

    best_candidates = np.zeros((num_best, 4), dtype=np.int64)
    values = np.zeros((num_candidates), dtype=np.float32)
    for i in range(num_candidates):
        values[i] = value_matrix[candidates[i, 0], candidates[i, 1], candidates[i, 2], candidates[i, 3]]

    for i in range(num_best):
        best_candidate_index = np.argmax(values)
        best_candidates[i, :] = candidates[best_candidate_index, :]
        values[best_candidate_index] = -999999

    return best_candidates


@jit(nopython=True)
def non_max_suppression_3d(matrix, search_width):
    matrix = np.copy(matrix)
    h, w, d = matrix.shape

    num_maximas = 0
    maximas = np.zeros((len(matrix.flat), 3), dtype=np.int64)

    for x in range(search_width, h - 1 - search_width):
        for y in range(search_width, w - 1 - search_width):
            for z in range(d):
                z_low = max(z - search_width, 0)
                z_high = min(z + 1 + search_width, d)

                v = matrix[x, y, z]

                ref_area = matrix[x - search_width: x + 1 + search_width,
                           y - search_width: y + 1 + search_width,
                           z_low: z_high]

                if v < np.max(ref_area):
                    continue
                if v == np.min(ref_area):
                    continue

                maximas[num_maximas, :] = (x, y, z)
                num_maximas += 1

    return maximas[:num_maximas, :]


@jit(nopython=True)
def non_max_suppression_3d_threshold(matrix, search_width, t):
    matrix = np.copy(matrix)
    h, w, d = matrix.shape

    num_maximas = 0
    maximas = np.zeros((len(matrix.flat), 3), dtype=np.int64)

    for x in range(search_width, h - 1 - search_width):
        for y in range(search_width, w - 1 - search_width):
            for z in range(d):
                z_low = max(z - search_width, 0)
                z_high = min(z + 1 + search_width, d)

                v = matrix[x, y, z]
                if v < t:
                    continue

                ref_area = matrix[x - search_width: x + 1 + search_width,
                           y - search_width: y + 1 + search_width,
                           z_low: z_high]

                if v < np.max(ref_area):
                    continue
                if v == np.min(ref_area):
                    continue

                maximas[num_maximas, :] = (x, y, z)
                num_maximas += 1

    return maximas[:num_maximas, :]


@jit(nopython=True)
def get_min_coords_2d_threshold(matrix, t):
    h, w = matrix.shape

    min_coords = np.zeros((2), dtype=np.int64)

    for x in range(h):
        for y in range(w):
            val = matrix[x, y]
            if val < t:
                t = val
                min_coords[:] = (x, y)

    return (min_coords, t)
