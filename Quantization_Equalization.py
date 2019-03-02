import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt

MAX_GRAY_LEVEL_VAL = 255
RGB_REP = 2
GRAYSCALE_REP = 1
RGB_SHAPE = 3

RGB2YIQ_MATRIX = np.float64([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])
YIQ2RGB_MATRIX = np.linalg.inv(RGB2YIQ_MATRIX)


def read_image(fileame, representation):
    """ reads an image file and converts it into a given representation
        representation -  is a code, either 1 or 2 defining whether the
        output should be a grayscale image (1) or an RGB image (2)"""

    if representation == GRAYSCALE_REP:
        return np.float64(imread(fileame, True)) / MAX_GRAY_LEVEL_VAL
    elif representation == RGB_REP:
        return np.float64(imread(fileame)) / MAX_GRAY_LEVEL_VAL


def imdisplay(filename, representation):
    """ Display an image in a given representation """
    im = read_image(filename, representation)
    if representation == GRAYSCALE_REP:
        plt.imshow(im, cmap=plt.cm.gray)
    elif representation == RGB_REP:
        plt.imshow(im)
    plt.show()


def transformingImage(im, matrix):
    x = im[:, :, 0]
    y = im[:, :, 1]
    z = im[:, :, 2]
    transImage = im.copy()
    for i in range(3):
        transImage[:, :, i] = matrix[i][0] * x + matrix[i][1] * y + matrix[i][2] * z
    return transImage


def rgb2yiq(imRGB):
    """ transforming an RGB image to YIQ color space"""
    return transformingImage(imRGB, RGB2YIQ_MATRIX)


def yiq2rgb(imYIQ):
    """ transforming an YIQ image to RGB image"""
    return transformingImage(imYIQ, YIQ2RGB_MATRIX)


def histogram_equalize(im_orig):
    """
    performs histogram equalization of a given grayscale or RGB image.
    :param im_orig: grayscale or RGB image with values [0,1].
    :return: a list [ im_eq, hist_orig, hist_eq] where:
    im_eq - is the equalized image. grayscale or RGB float64 image with values in [0, 1].
    hist_orig - is a 256 bin histogram of the original image.
    hist_eq - is a 256 bin histogram of the equalized image.
    """
    yiq_im = []
    if len(im_orig.shape) == RGB_SHAPE:
        yiq_im = rgb2yiq(im_orig)
        init_im = yiq_im[:, :, 0].copy()
    else:
        init_im = im_orig.copy()

    hist_orig, bin_edges = np.histogram(init_im, 256)

    hist_cum = np.cumsum(hist_orig)
    nonZeroIndices = np.nonzero(hist_cum)
    minNonzero, maxNonZero = hist_cum[nonZeroIndices[0][0]], hist_cum[nonZeroIndices[0][-1]]
    # stretch the image
    stretched_im = np.round(255 * (hist_cum - minNonzero) / (maxNonZero - minNonzero))

    im_eq = np.interp(init_im.flatten(), bin_edges[:-1], stretched_im).reshape(init_im.shape) / MAX_GRAY_LEVEL_VAL
    if len(im_orig.shape) == RGB_SHAPE:
        yiq_im[:, :, 0] = im_eq
        im_eq = yiq2rgb(yiq_im)

    hist_eq, bin_edges2 = np.histogram(im_eq, 256)
    return [np.float64(np.clip(im_eq, 0, 1)), hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """performs optimal quantization of a given grayscale or RGB image."""
    yiq_im = []

    if len(im_orig.shape) == RGB_SHAPE:
        yiq_im = rgb2yiq(im_orig)
        init_im = yiq_im[:, :, 0].copy()
    else:
        init_im = im_orig.copy()

    hist, bin_edges = np.histogram(init_im, 256)
    hist_cum = np.cumsum(hist)
    pixInSegment = int(hist_cum[-1] / n_quant)

    error, qArr, zArr = segmtents_quantization(bin_edges, hist, hist_cum, n_iter, n_quant, pixInSegment)

    for segNum in range(n_quant):
        inSeg = np.logical_and(init_im >= zArr[segNum], init_im < zArr[segNum + 1])
        init_im[inSeg] = qArr[segNum]
    # pixels with intensity 255
    init_im[init_im == 1] = qArr[-1]

    im_quant = init_im
    if len(im_orig.shape) == RGB_SHAPE:
        yiq_im[:, :, 0] = im_quant
        im_quant = yiq2rgb(yiq_im)

    return [im_quant, np.array(error)]


def segmtents_quantization(bin_edges, hist, hist_cum, n_iter, n_quant, pixInSegment):
    # initial segments
    zArr = np.array([0] + [bin_edges[np.where(hist_cum >= pixInSegment * i)[0][0]] for i in range(1, n_quant)] + [1])
    qArr = np.zeros(n_quant)
    error = []
    for i in range(n_iter):
        curZ = zArr.copy()
        curErr = 0

        for k in range(n_quant):
            if k != n_quant - 1:
                curSeg = np.intersect1d(np.where(bin_edges[:-1] >= zArr[k])[0],
                                        np.where(bin_edges[:-1] < zArr[k + 1])[0])
            else:
                curSeg = np.intersect1d(np.where(bin_edges[:-1] >= zArr[k])[0],
                                        np.where(bin_edges[:-1] <= zArr[k + 1])[0])

            qArr[k] = np.dot(bin_edges[curSeg], hist[curSeg]) / np.sum(hist[curSeg])
            curErr += np.dot(np.power(qArr[k] - bin_edges[curSeg], 2), hist[curSeg])
        error.append(curErr)
        zArr = np.array([0] + [(qArr[k] + qArr[k - 1]) / 2 for k in range(1, n_quant)] + [1])
        if np.array_equal(zArr, curZ):
            break
    return error, qArr, zArr


im = read_image("monkey.jpg", 1)
# imq, err = quantize(im, 5, 5)
imq,h1,h2=histogram_equalize(im)
plt.imshow(im, cmap="gray")
plt.show()
plt.imshow(imq, cmap="gray")
plt.show()
