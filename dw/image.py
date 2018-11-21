import numpy as np
from numba import jit
import imageio


# ===== IO

def read(path):
    I = imageio.imread(path).astype(np.float)
    if path.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']:
        I /= 255.0
    return np.atleast_3d(I)


def write(path, I):
    if path.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']:
        v_min = np.min(I)
        v_max = np.max(I)
        assert v_min >= 0.0 and v_max <= 1.0, "out of range to write as ubyte ({},{})".format(v_min, v_max)
        I = (I*255.0).astype(np.ubyte)
    imageio.imwrite(path, I)


# ===== PADDING

@jit(nopython=True)
def extend_same(I, border_width):
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


# ===== INTERPOLATION

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
def rescale(I, factor, interpolate=True):
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
                result[x_, y_, :] = sub_pixel(I, x, y)
            else:
                xr = min(max(int(np.round(x)), 0), h - 1)
                yr = min(max(int(np.round(y)), 0), w - 1)
                result[x_, y_, :] = I[xr, yr]

    return result
