import network_torch as network
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

for i in range(n_trials):
    print("Trial {:>2d}/{:>2d}. --------------------".format(i+1, n_trials))

    # create the network
    net = network.Network(n=n)

    # train the network
    loss = net.train(f_etas, b_etas, n_epochs, plot_activity=plot_activity, weight_decay=weight_decay, update_b_weights=False)

    losses[0, i] = loss

for i in range(n_trials):
    print("Trial {:>2d}/{:>2d}. --------------------".format(i+1, n_trials))

    # create the network
    net = network.Network(n=n)

    # train the network
    loss = net.train(f_etas, b_etas, n_epochs, plot_activity=plot_activity, weight_decay=weight_decay, update_b_weights=True)

    losses[1, i] = loss

plt.ioff()
plt.close('all')

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlabel("Epoch")

colors = ["red", "blue", "green"]
colors_2 = ["magenta", "cyan", "yellow"]
layer_types = ["Hidden"]*(len(n)-2) + ["Output"]

for m in range(len(n)-1):
    plt.ioff()
    plt.close('all')

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlabel("Epoch")

    colors = ["red", "blue", "green"]
    colors_2 = ["magenta", "cyan", "yellow"]
    layer_types = ["Hidden"]*(len(n)-2) + ["Output"]

    losses_max = losses[0, :, m].max(axis=0)
    losses_min = losses[0, :, m].min(axis=0)

    if n_trials > 1:
        ax.fill_between(np.arange(losses.shape[-1])/20, losses_min, losses_max, facecolor=colors[m], alpha=0.3)
    
    ax.plot(np.arange(losses.shape[-1])/20, np.mean(losses[0, :, m], axis=0), colors[m], label='No Backward Updates ({}) Layer {} Loss'.format(layer_types[m], m+1), alpha=0.5)

    losses_max = losses[1, :, m].max(axis=0)
    losses_min = losses[1, :, m].min(axis=0)

    if n_trials > 1:
        ax.fill_between(np.arange(losses.shape[-1])/20, losses_min, losses_max, facecolor=colors_2[m], alpha=0.3)
    
    ax.plot(np.arange(losses.shape[-1])/20, np.mean(losses[1, :, m], axis=0), colors_2[m], label='Backward Updates ({}) Layer {} Loss'.format(layer_types[m], m+1), alpha=0.5)

    plt.legend()
    plt.savefig('losses_layer_{}.svg'.format(m))
    plt.savefig('losses_layer_{}.png'.format(m))
    plt.show()