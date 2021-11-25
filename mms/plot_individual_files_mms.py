"""Check logarithmic plots of accs values"""


import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    npzfile = "acc_p_2_false.npz"
    accuracies = np.load(npzfile, allow_pickle=True)
    errors = accuracies['acc']
    steps = accuracies['dx']
    log_errors = np.log(errors)
    log_steps = np.log(steps)
    a, b = np.polyfit(log_steps, log_errors, 1)
    x = np.linspace(log_steps[0], log_steps[-1], 100)
    y = a*x + b
    print(f"inclination = {a}")

    plt.plot(log_steps, log_errors, 'o'); 
    plt.plot(x, y)
    plt.show()
