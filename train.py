import network_torch as network
import torch
import numpy as np
import matplotlib.pyplot as plt

# set training parameters
f_etas = [0.1, 0.1, 0.1, 0.1]
b_etas = [0.1, 0.1, 0.1, 0.1]

n = [500, 300, 200, 100, 3]

n_epochs      = 100
n_trials      = 1
plot_activity = True
weight_decay  = 0.0001

# initalize array to hold losses
losses = np.zeros((n_trials, len(n)-1, int((n_epochs-int(plot_activity))*network.sequence_length/100.0)))

for i in range(n_trials):
    print("Trial {:>2d}/{:>2d}. --------------------".format(i+1, n_trials))

    # create the network
    net = network.Network(n=n)

    # train the network
    loss = net.train(f_etas, b_etas, n_epochs, plot_activity=plot_activity, weight_decay=weight_decay)

    losses[i] = loss

plt.ioff()
plt.close('all')

fig, ax = plt.subplots()

colors = ["red", "blue", "green", "purple", "brown", "cyan", "orange", "magenta"]

for m in range(len(n)-1):
    losses_max = losses[:, m].max(axis=0)
    losses_min = losses[:, m].min(axis=0)

    if n_trials > 1:
        ax.fill_between(np.arange(losses.shape[-1]), losses_min, losses_max, facecolor=colors[m], alpha=0.5)
    
    ax.plot(np.arange(losses.shape[-1]), np.mean(losses[:, m], axis=0), colors[m], label='Layer {} Loss'.format(m+1), alpha=0.5)

plt.legend()
plt.show()