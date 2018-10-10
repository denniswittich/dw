"""
Created on Oct 10 2018
Last edited on Oct 10 2018
@author: Dennis Wittich
"""

from numba import jit
import numpy as np

@jit(nopython=True)
def update_confusion_matrix(confusions, predicted_labels, reference_labels):
    reshaped_pr = np.ravel(predicted_labels)
    reshaped_gt = np.ravel(reference_labels)
    for predicted, actual in zip(reshaped_pr, reshaped_gt):
        confusions[predicted, actual] += 1

def get_confusion_metrics(confusion_matrix):
    """Computes confusion metrics out of a confusion matrix (N classes)

        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            Confusion matrix [N x N]

        Returns
        -------
        out1 : numpy.ndarray
            class distribution [N] with sum = 1.0
        out2 : numpy.ndarray
            precisions [N] TP/(TP+FP)
        out3 : numpy.ndarray
            recall [N] TP/(TP+FN)
        out4 : numpy.ndarray
            f1 [N] harmonic mean of precision and recall
        out5 : numpy.ndarray
            mean of all f1 scores

        Notes
        -----
        'I' will always have 3 dimensions: (rows, columns dimensions).
        Last dimension will be of length 1 or 3, depending on the image.

    """
    tp = np.diag(confusion_matrix)
    tp_fn = np.sum(confusion_matrix, axis=0)
    tp_fp = np.sum(confusion_matrix, axis=1)
    pctgs = tp_fn / np.sum(confusion_matrix)
    precisions = tp / tp_fp
    recall = tp / tp_fn
    f1 = 2 * (precisions * recall) / (precisions + recall)
    f1[np.isnan(f1)] = 0.0
    f1[pctgs == 0.0] = np.nan
    mean_f1 = np.nanmean(f1)
    oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return pctgs, precisions, recall, f1, mean_f1, oa

@jit(nopython=True)
def smooth1d(a, n):
    """Reads an image from disk. Returns the array representation.

        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            Confusion matrix [N x N]

        Returns
        -------
        out : ndarray of float64
            Image as 3D array

        Notes
        -----
        'I' will always have 3 dimensions: (rows, columns dimensions).
        Last dimension will be of length 1 or 3, depending on the image.

    """
    d = n // 2
    b = np.zeros_like(a)
    for i in range(len(a)):
        summ = 0
        for j in range(n):
            k = i - d + j
            if k < 0:
                summ += a[0]
            elif k > len(a) - 1:
                summ += a[len(a) - 1]
            else:
                summ += a[k]
        b[i] = summ / n
    return b