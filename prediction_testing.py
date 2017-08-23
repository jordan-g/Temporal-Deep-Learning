import network
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_mean_diffs(folder):
    diffs = np.load(os.path.join(folder, "diffs.npy"))
    colors = ["red", "blue", "green"]

    plt.close('all')
    mean_diffs = np.mean(diffs, axis=-1)
    plt.scatter(np.zeros(diffs.shape[1]) + np.random.normal(0, 0.1, size=diffs.shape[1]), mean_diffs[0], c=colors[0], s=8, alpha=0.5, label='No Backward Updates')
    plt.scatter(np.ones(diffs.shape[1]) + np.random.normal(0, 0.1, size=diffs.shape[1]), mean_diffs[1], c=colors[1], s=8, alpha=0.5, label='Backward Updates')
    plt.hlines(y=np.mean(mean_diffs[0]), xmin=-0.2, xmax=0.2, colors=colors[0], lw=2)
    plt.hlines(y=np.mean(mean_diffs[1]), xmin=0.8, xmax=1.2, colors=colors[1], lw=2)
    plt.xlim(-1, 2)

    plt.legend()
    plt.savefig(os.path.join(folder, "mean_diffs.svg"))
    plt.savefig(os.path.join(folder, "mean_diffs.png"))
    plt.show()

if __name__ == "__main__":
    # set training parameters
    f_etas = [5.0, 0.01]
    b_etas = [0.05]

    n = [500, 200, 3]

    n_epochs      = 100
    n_trials      = 10
    plot_activity = False
    weight_decay  = 0.0

    folder = "Generated Activity Differences (b_eta={})".format(b_etas[0])
    if not os.path.exists(folder):
        os.makedirs(folder)

    # initalize array to hold losses
    losses = np.zeros((2, n_trials, len(n)-1, int((n_epochs-int(plot_activity))*network.sequence_length/100.0)))
    diffs  = np.zeros((2, n_trials, int(network.sequence_length/2.0)))

    for i in range(n_trials):
        print("Trial {:>2d}/{:>2d}. --------------------".format(i+1, n_trials))

        # create the network
        net = network.Network(n=n)

        # train the network
        loss, diff = net.train(f_etas, b_etas, n_epochs, plot_activity=plot_activity, weight_decay=weight_decay, update_b_weights=False, generate_activity=True)

        losses[0, i] = loss
        diffs[0, i] = diff

    for i in range(n_trials):
        print("Trial {:>2d}/{:>2d}. --------------------".format(i+1, n_trials))

        # create the network
        net = network.Network(n=n)

        # train the network
        loss, diff = net.train(f_etas, b_etas, n_epochs, plot_activity=plot_activity, weight_decay=weight_decay, update_b_weights=True, generate_activity=True)

        losses[1, i] = loss
        diffs[1, i] = diff

    plt.ioff()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlabel("Epoch")

    colors = ["red", "blue", "green"]

    diffs_max = diffs[0, :].max(axis=0)
    diffs_min = diffs[0, :].min(axis=0)

    if n_trials > 1:
        ax.fill_between(np.arange(diffs.shape[-1]), diffs_min, diffs_max, facecolor=colors[0], alpha=0.3)

    ax.plot(np.arange(diffs.shape[-1]), np.mean(diffs[0, :], axis=0), colors[0], label='No Backward Updates', alpha=0.5)

    diffs_max = diffs[1, :].max(axis=0)
    diffs_min = diffs[1, :].min(axis=0)

    if n_trials > 1:
        ax.fill_between(np.arange(diffs.shape[-1]), diffs_min, diffs_max, facecolor=colors[1], alpha=0.3)

    ax.plot(np.arange(diffs.shape[-1]), np.mean(diffs[1, :], axis=0), colors[1], label='Backward Updates', alpha=0.5)

    plt.legend()
    plt.savefig(os.path.join(folder, "diffs.svg"))
    plt.savefig(os.path.join(folder, "diffs.png"))
    plt.show()

    np.save(os.path.join(folder, "diffs.npy"), diffs)
    np.save(os.path.join(folder, "losses.npy"), losses)

    plot_mean_diffs(folder)