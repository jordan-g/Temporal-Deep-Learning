import numpy as np
import matplotlib.pyplot as plt
import os

def plot_test_errors(folder, suffixes, trial_num=0, save=False):
    colors = ['#f77e90', '#7ecaf7', '#61d873', '#f4bb35', '#4ae0c9', '#ed6aa1']

    errors_list = []
    for i in range(len(suffixes)):
        suffix = suffixes[i]
        k = trial_num
        while k >= 0:
            try:
                errors = np.load(os.path.join("simulations", folder, "trial_{}_errors{}.npy".format(k, ("_" + suffix)*(len(suffix) > 0) )))[:k+1]
                break
            except:
                k -= 1

        errors_list.append(errors)

    mean_errors_list = [ np.mean(errors, axis=0) for errors in errors_list ]
    min_errors_list  = [ np.amin(errors, axis=0) for errors in errors_list ]
    max_errors_list  = [ np.amax(errors, axis=0) for errors in errors_list ]

    x = np.arange(mean_errors_list[0].shape[0])

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(len(suffixes)):
        suffix = suffixes[i]

        ax.fill_between(x, min_errors_list[i], max_errors_list[i], facecolor=colors[i], alpha=0.5)
        ax.plot(x, mean_errors_list[i], colors[i], label=suffixes[i], lw=1)

    plt.title("Test error comparison")
    plt.xlabel("Epoch #")
    plt.ylabel("Test error (%)")
    plt.yscale('log')

    if np.amin(min_errors_list) > 10:
        plt.ylim(10, 100)
    else:
        plt.ylim(1, 100)
    plt.xlim(0, mean_errors_list[0].shape[0])
    plt.legend()

    if save:
        plt.savefig(os.path.join("simulations", folder, "test_error_comparion.png"))
        plt.savefig(os.path.join("simulations", folder, "test_error_comparion.svg"))
    else:
        plt.show()
