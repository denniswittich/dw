"""
Created on Oct 10 2018
Last edited on Oct 10 2018
@author: Dennis Wittich
"""

from numba import jit, float32, float64, int64, void, types, boolean, uint8, int32
import numpy as np
import matplotlib.pyplot as plt


def get_confusion_metrics(confusion_matrix):
    """Computes confusion metrics out of a confusion matrix (N classes)

        Parameters
        ----------
        confusion_matrix : numpy.ndarray
            Confusion matrix [N x N]

        Returns
        -------
        metrics : dict
            a dictionary holding all computed metrics

        Notes
        -----
        Metrics are: 'percentages', 'precisions', 'recalls', 'f1s', 'mf1', 'oa'

    """

    tp = np.diag(confusion_matrix)
    tp_fn = np.sum(confusion_matrix, axis=0)
    tp_fp = np.sum(confusion_matrix, axis=1)

    has_rp = tp_fn > 0
    has_pp = tp_fp > 0

    tp_fn[has_rp] = 1
    tp_fp[has_pp] = 1

    percentages = tp_fn / np.sum(confusion_matrix)
    precisions = tp / tp_fp
    recalls = tp / tp_fn

    f1s = 2 * (precisions * recalls) / (precisions + recalls)
    ious = tp / (tp_fn + tp_fp - tp)

    precisions[has_pp] *= 0.0
    recalls[has_rp] *= 0.0

    f1s[percentages == 0.0] = np.nan
    ious[percentages == 0.0] = np.nan

    mf1 = np.nanmean(f1s)
    miou = np.nanmean(ious)
    oa = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    metrics = {'percentages': percentages,
               'precisions': precisions,
               'recalls': recalls,
               'f1s': f1s,
               'mf1': mf1,
               'ious': ious,
               'miou': miou,
               'oa': oa}

    return metrics


# =====================================================================

def print_stats(X, y, classes, features):
    Nfeatures = X.shape[0]
    print('Set contains {} features'.format(Nfeatures))
    print('\nClass    |   Num   |   Pct   |')
    for i, class_ in enumerate(classes):
        num = np.sum(y == i)
        pct = num * 100 / Nfeatures
        print('{:<9}|{:^9.0f}|{:^9.2f}|'.format(class_, num, pct))
    print('\nFeature  |   Min   |   Max   |   Mean  |')
    for i, feature in enumerate(features):
        print('{:<9}|{:^9.2f}|{:^9.2f}|{:^9.2f}|'.format(
            feature, np.min(X[:, i]), np.max(X[:, i]), np.mean(X[:, i])))


@jit(nopython=True)
def update_confusion_matrix(confusions, predicted_labels, reference_labels):
    # reference labels with label < 0 will not be considered
    reshaped_pr = np.ravel(predicted_labels)
    reshaped_gt = np.ravel(reference_labels)
    for predicted, actual in zip(reshaped_pr, reshaped_gt):
        if actual >= 0 and predicted >= 0:
            confusions[predicted, actual] += 1


def plot_confusions(confusions):
    num = confusions.shape[0]
    #print('\nConfusion Matrix:')
    plt.figure()
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(False)
    plt.xticks(np.arange(num))
    plt.yticks(np.arange(num))
    plt.imshow(confusions, cmap=plt.cm.jet, interpolation='nearest');
    for i, cas in enumerate(confusions):
        for j, count in enumerate(cas):
            if count > 0:
                xoff = .07 * len(str(count))
                plt.text(j - xoff, i + .2, int(count), fontsize=9, color='white')
    plt.show()


def print_metrics(confusions):
    metrics = get_confusion_metrics(confusions)

    print('\nclass | pct of data | precision |   recall  |    f1     |    iou',
          '\n-----------------------------------------------------------------')

    percentages = metrics["percentages"]
    precisions = metrics["precisions"]
    recall = metrics["recalls"]
    f1 = metrics["f1s"]
    ious = metrics["ious"]
    mf1 = metrics["mf1"]
    miou = metrics["miou"]
    oa = metrics["oa"]

    for i in range(len(percentages)):
        pct = '{:.3%}'.format(percentages[i]).rjust(9)
        p = '{:.3%}'.format(precisions[i]).rjust(7)
        r = '{:.3%}'.format(recall[i]).rjust(7)
        f = '{:.3%}'.format(f1[i]).rjust(7)
        u = '{:.3%}'.format(ious[i]).rjust(7)
        print('   {:2d} |  {}  |  {}  |  {}  |  {}  |  {}\n'.format(i, pct, p, r, f, u))

    print('mean f1-score: {:.3%}'.format(mf1))
    print('mean iou: {:.3%}'.format(miou))
    print('Overall accuracy: {:.3%}'.format(oa))
    print('Samples: {}'.format(np.sum(confusions)))


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


class mova(object):
    # class to keep track of the MOVING AVERAGE of a variable

    def __init__(self, initial_value=None, momentum=0.9, print_accuracy=4):
        assert 0.0 < momentum < 1.0, "momentum has to be between 0.0 and 1.0"
        self.value = None if not initial_value else float(initial_value)
        self.momentum = float(momentum)
        self.inc = 1.0 - momentum
        self.str = '{:.' + str(int(print_accuracy)) + 'f}'

    def __call__(self, other):
        self.value = float(other) if not self.value else self.value * self.momentum + other * self.inc
        return self

    def __str__(self):
        return self.str.format(self.value)


# =============== EXPERIMENTS

def plot_table_chart(data, rows=None, columns=None, y_label='', title='', cell_rule='{}', cmap=plt.cm.Accent):
    n_rows, n_cols = data.shape[:2]

    if not columns:
        columns = [str(i) for i in range(n_cols)]
    if not rows:
        rows = [str(i) for i in range(n_rows)]

    # Get some pastel shades for the colors
    colors = cmap(np.linspace(0, 0.5, len(rows)))

    cell_text = []
    for row in range(n_rows):
        plt.plot(data[row], color=colors[row])
        cell_text.append([cell_rule.format(v) for v in data[row]])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    tab = plt.table(cellText=cell_text,
                    rowLabels=rows,
                    rowColours=colors,
                    colLabels=columns,
                    loc='bottom')
    tab.scale(1, 2)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    plt.ylabel(y_label)
    plt.xticks([])
    plt.title(title)
    plt.show()
