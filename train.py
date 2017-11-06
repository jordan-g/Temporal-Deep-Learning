import network
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# number of epochs of training (one epoch = one complete showing of the input sequence)
n_epochs = 100

# weight_decay = 0.001

# number of trials to repeat training
n_trials = 1

folder = "testing_mnist"

if not os.path.exists(folder):
    os.makedirs(folder)

# feedforward learning rates
f_etas_list = [[0.01], [5.0, 0.01], [5.0, 5.0, 0.01], [5.0, 5.0, 5.0, 0.01]]

# number of units per layer (including input layer)
n_list      = [[784, 10], [784, 300, 10], [784, 300, 200, 10], [784, 300, 200, 100, 10]]

for i in range(len(n_list)):
    f_etas = f_etas_list[i]
    n      = n_list[i]

    # initalize array to hold losses
    losses       = np.zeros((n_trials, len(n)-1, int(n_epochs*network.n_sequences)))
    losses_2     = np.zeros((n_trials, len(n)-1, int(n_epochs*network.n_sequences)))
    avg_losses   = np.zeros((n_trials, len(n)-1, n_epochs*network.n_sequences))
    avg_losses_2 = np.zeros((n_trials, len(n)-1, n_epochs*network.n_sequences))
    errors       = np.zeros((n_trials, network.n_classes))

    for j in range(n_trials):
        print("{} hidden layers. Trial {:>2d}/{:>2d}. --------------------".format(len(n)-2, j+1, n_trials))

        # create the network
        net = network.Network(n=n, use_mnist=True)

        loss, loss_2, error, outputs, targets, target_times, test_outputs, test_targets, class_nums = net.train_mnist(f_etas, None, n_epochs, weight_decay=weight_decay, trial=j)

        losses[j] = loss
        losses_2[j] = loss_2
        errors[j] = error

    for l in range(n_epochs*network.n_sequences):
        avg_losses[:, :, l] = np.mean(losses[:, :, l:l+1], axis=-1)
        avg_losses_2[:, :, l] = np.mean(losses_2[:, :, l:l+1], axis=-1)

    suffix = "{}_hidden".format(len(n)-2)

    np.save(os.path.join(folder, "losses_{}.npy".format(suffix)), losses)
    np.save(os.path.join(folder, "avg_losses_{}.npy".format(suffix)), avg_losses)
    np.save(os.path.join(folder, "losses_2_{}.npy".format(suffix)), losses_2)
    np.save(os.path.join(folder, "avg_losses_2_{}.npy".format(suffix)), avg_losses_2)
    np.save(os.path.join(folder, "errors_{}.npy".format(suffix)), errors)
    np.save(os.path.join(folder, "outputs_{}.npy".format(suffix)), outputs)
    np.save(os.path.join(folder, "targets_{}.npy".format(suffix)), targets)
    np.save(os.path.join(folder, "target_times_{}.npy".format(suffix)), target_times)
    np.save(os.path.join(folder, "test_outputs_{}.npy".format(suffix)), test_outputs)
    np.save(os.path.join(folder, "test_targets_{}.npy".format(suffix)), test_targets)
    np.save(os.path.join(folder, "class_nums_{}.npy".format(suffix)), class_nums)