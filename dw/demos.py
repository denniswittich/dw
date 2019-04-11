from matplotlib import pyplot as plt
import dw.eval as dwe


def mova_demo():
    moments = [0.9, 0.99, 0.999]
    movas = [dwe.mova(1.0, m) for m in moments]

    histories = [[1.0], [1.0], [1.0]]

    for i in range(300):
        for hist, mova in zip(histories , movas):
            mova(0.0)
            hist.append(mova.value)

    for i in range(300):
        for hist, mova in zip(histories , movas):
            mova(0.5)
            hist.append(mova.value)

    for hist, moment in zip(histories,moments):
        plt.plot(hist, label='mom. {}'.format(moment))
    plt.legend()
    plt.show()

if __name__=='__main__':
    mova_demo()

