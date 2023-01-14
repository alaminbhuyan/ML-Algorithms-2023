import numpy as np


def sigmoid(num):
    return 1 / (1 + np.exp(-num))

def weightUpdate(x1, x2, w1, w2, wb, y, b=1):
    learning_rate = 0.1

    while True:
        print("weight1: {}, weight2: {}, bias_weight: {}".format(w1, w2, wb))

        result = (x1*w1) + (x2*w2) + (b * wb)

        y_pred = sigmoid(result)

        if y_pred>=0.5:
            y_pred = 1
        else:
            y_pred = 0

        error = y - y_pred

        print("error: {}".format(error))

        if error == 0:
            print("final w1: {}, final w2: {}, final wb: {}".format(w1, w2, wb))
            break
        else:
            w1 = round(w1 + learning_rate * x1 * (error),2)
            w2 = round(w2 + learning_rate * x2 * (error), 2)
            wb = round(wb + learning_rate * b * (error), 2)


weightUpdate(x1=0, x2=1, w1=0.5, w2=0.1, wb=0.5, y=0)