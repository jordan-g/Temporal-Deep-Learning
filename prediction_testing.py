import network
import torch
import numpy as np
import matplotlib.pyplot as plt

# set training parameters
f_etas = [5.0, 0.01]
b_etas = [0.05]

n = [500, 200, 3]

n_epochs      = 100
n_trials      = 10
plot_activity = False
weight_decay  = 0.0

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
plt.savefig('diffs.svg')
plt.savefig('diffs.png')
plt.show()