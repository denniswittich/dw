from matplotlib import pyplot as plt
import numpy as np

import dw.eval as dwe
import dw.video as dwv


def mova_demo():
    moments = [0.9, 0.99, 0.999]
    movas = [dwe.mova(1.0, m) for m in moments]

    histories = [[1.0], [1.0], [1.0]]

    for i in range(300):
        for hist, mova in zip(histories, movas):
            mova(0.0)
            hist.append(mova.value)

    for i in range(300):
        for hist, mova in zip(histories, movas):
            mova(0.5)
            hist.append(mova.value)

    for hist, moment in zip(histories, moments):
        plt.plot(hist, label='mom. {}'.format(moment))
    plt.legend()
    plt.show()


def confusion_matrix_demo():
    num_classes = 4

    # 1. initialize a confusion matrix
    CM = np.zeros((num_classes, num_classes))

    # 2. define predicted and reference labels (e.g. 1D arrays)
    predictions = np.random.randint(0, num_classes, 10000)
    reference = np.random.randint(0, num_classes, 10000)

    # 3. update confusion matrix
    dwe.update_confusion_matrix(CM, predictions, reference)

    # (repeat 2,3) update again, this time with 3D arrays
    predictions = np.random.randint(0, num_classes, (3, 4, 5))
    reference = np.random.randint(0, num_classes, (3, 4, 5))
    dwe.update_confusion_matrix(CM, predictions, reference)

    # 4. print CM to console
    dwe.print_metrics(CM)
    #dwe.plot_confusions(CM)


def gif_writer_demo():
    gw = dwv.gif_writer([-3,3])
    for i in range(10):
        img = (np.random.randn(300,300,3))
        gw.append(img)
    gw.write('./gif_demo')

if __name__ == '__main__':
    # mova_demo()
    # confusion_matrix_demo()
    # gif_writer_demo()
    pass