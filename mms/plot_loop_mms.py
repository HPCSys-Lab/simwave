"""Check logarithmic plots of accs values"""


import numpy as np
import matplotlib.pyplot as plt


def get_log_fit(accs):
    log_accs = {}
    for order in accs:
        step = np.log(list(accs[order].keys()))
        error = np.log(list(accs[order].values()))
        log_accs[order] = dict(zip(step, error))
    return log_accs

def log_arrays(log_acc):
    x = [key for key in log_acc.keys()]
    y = [value for value in log_acc.values()]
    return x, y

def get_polyfit(log_acc):
    x_, y_ = log_arrays(log_acc)
    # p = np.polynomial.Polynomial.fit(x_, y_, 1)
    # c, b = p.coef
    b, c = np.polyfit(x_, y_, 1)
    x = np.linspace(x_[0], x_[-1], 100)
    y = c + b*x
    print(f"inclination = {b}")
    return x, y

def get_log_data(order, accs_file):
    accs = np.load(accs_file, allow_pickle=True).item()
    log_accs = get_log_fit(accs)
    x, y = log_arrays(log_accs[order])
    xp, yp = get_polyfit(log_accs[order])

    return x, y, xp, yp

if __name__ == '__main__':
    x, y, xp, yp = get_log_data(2, "accs.npy")
    plt.plot(x, y, 'o', xp, yp)
    plt.show()
    
