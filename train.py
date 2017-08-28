import network
import torch
import numpy as np
import matplotlib.pyplot as plt

# feedforward learning rates
f_etas = [20.0, 0.01]
f_etas = [0.01]

# feedback learning rates
b_etas = [0.0001]

# number of units per layer (including input layer)
n = [500, 300, 3]
n = [500, 3]

# number of epochs of training (one epoch = one complete showing of the input sequence)
n_epochs = 10

# number of trials to repeat training
n_trials = 1

# weight decay constant
weight_decay = 0.0

update_b_weights  = False # whether to update feedback weights
plot_activity     = True  # whether to show a plot of the network activity vs. target activity, before and after training
generate_activity = True  # whether to internally generate activity during the second half of the last epoch

# don't plot activity if we are running multiple trials
if n_trials > 1:
    plot_activity = False

# initalize array to hold losses
losses = np.zeros((n_trials, len(n)-1, int(n_epochs*network.n_sequences*network.sequence_length/100.0)))
errors = np.zeros((n_trials, n_classes))

for i in range(n_trials):
    print("Trial {:>2d}/{:>2d}. --------------------".format(i+1, n_trials))

    # create the network
    net = network.Network(n=n)

    # train the network
    loss, error, _, _, _ = net.train(f_etas, b_etas, n_epochs, plot_activity=plot_activity, weight_decay=weight_decay, update_b_weights=update_b_weights, generate_activity=generate_activity)

    losses[i] = loss
    errors[i] = error

plt.ioff()
plt.close('all')

# create figure
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlabel("Epoch")

colors = ["red", "blue", "green", "purple", "brown", "cyan", "orange", "magenta"]
layer_types = ["Hidden"]*(len(n)-2) + ["Output"]

for m in range(len(n)-1):
    losses_max = losses[:, m].max(axis=0)
    losses_min = losses[:, m].min(axis=0)

    if n_trials > 1:
        ax.fill_between(np.arange(losses.shape[-1])/20, losses_min, losses_max, facecolor=colors[m], alpha=0.3)

        mean_alpha = 0.5
    else:
        mean_alpha = 1
    
    ax.plot(np.arange(losses.shape[-1])/20, np.mean(losses[:, m], axis=0), colors[m], label='({}) Layer {} Loss'.format(layer_types[m], m+1), alpha=mean_alpha)

plt.legend()
plt.savefig('losses.svg')
plt.savefig('losses.png')
plt.show()