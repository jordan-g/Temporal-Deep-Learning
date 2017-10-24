import network
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# number of epochs of training (one epoch = one complete showing of the input sequence)
n_epochs = 20

# number of trials to repeat training
n_trials = 5

folder = "mnist_test"

if not os.path.exists(folder):
    os.makedirs(folder)

# _, _ = network.create_data(500, 3)

# ----- NO HIDDEN LAYERS ----- #

# # feedforward learning rates
# f_etas = [0.01]

# # number of units per layer (including input layer)
# n = [500, 3]

# # initalize array to hold losses
# losses       = np.zeros((n_trials, len(n)-1, int(n_epochs*network.n_sequences*network.sequence_length/100.0)))
# losses_2     = np.zeros((n_trials, len(n)-1, int(n_epochs*network.n_sequences*network.sequence_length/100.0)))
# avg_losses   = np.zeros((n_trials, len(n)-1, n_epochs*network.n_sequences))
# avg_losses_2 = np.zeros((n_trials, len(n)-1, n_epochs*network.n_sequences))
# errors       = np.zeros((n_trials, network.n_classes))

# for i in range(n_trials):
#     print("No hidden layers. Trial {:>2d}/{:>2d}. --------------------".format(i+1, n_trials))

#     # create the network
#     net = network.Network(n=n)

#     loss, loss_2, error, outputs, targets, target_times, test_outputs, test_targets, class_nums = net.train(f_etas, None, n_epochs, update_b_weights=False, generate_activity=False)

#     losses[i] = loss
#     losses_2[i] = loss_2
#     errors[i] = error

# for l in range(n_epochs*network.n_sequences):
#     avg_losses[:, :, l] = np.mean(losses[:, :, int(l*network.sequence_length/100.0):int((l+1)*network.sequence_length/100.0)], axis=-1)
#     avg_losses_2[:, :, l] = np.mean(losses_2[:, :, int(l*network.sequence_length/100.0):int((l+1)*network.sequence_length/100.0)], axis=-1)

# suffix = "no_hidden"

# np.save(os.path.join(folder, "losses_{}.npy".format(suffix)), losses)
# np.save(os.path.join(folder, "avg_losses_{}.npy".format(suffix)), avg_losses)
# np.save(os.path.join(folder, "losses_2_{}.npy".format(suffix)), losses_2)
# np.save(os.path.join(folder, "avg_losses_2_{}.npy".format(suffix)), avg_losses_2)
# np.save(os.path.join(folder, "errors_{}.npy".format(suffix)), errors)
# np.save(os.path.join(folder, "outputs_{}.npy".format(suffix)), outputs)
# np.save(os.path.join(folder, "targets_{}.npy".format(suffix)), targets)
# np.save(os.path.join(folder, "target_times_{}.npy".format(suffix)), target_times)
# np.save(os.path.join(folder, "test_outputs_{}.npy".format(suffix)), test_outputs)
# np.save(os.path.join(folder, "test_targets_{}.npy".format(suffix)), test_targets)
# np.save(os.path.join(folder, "class_nums_{}.npy".format(suffix)), class_nums)

# # ----- ONE HIDDEN LAYER ----- #

# # feedforward learning rates
# f_etas = [2.0, 0.01]

# # number of units per layer (including input layer)
# n = [500, 300, 3]

# # initalize array to hold losses
# losses       = np.zeros((n_trials, len(n)-1, int(n_epochs*network.n_sequences*network.sequence_length/100.0)))
# losses_2     = np.zeros((n_trials, len(n)-1, int(n_epochs*network.n_sequences*network.sequence_length/100.0)))
# avg_losses   = np.zeros((n_trials, len(n)-1, n_epochs*network.n_sequences))
# avg_losses_2 = np.zeros((n_trials, len(n)-1, n_epochs*network.n_sequences))
# errors       = np.zeros((n_trials, network.n_classes))
# diffs        = np.zeros((n_trials, network.n_classes, int(network.sequence_length/2.0)))

# for i in range(n_trials):
#     print("No hidden layers. Trial {:>2d}/{:>2d}. --------------------".format(i+1, n_trials))

#     # create the network
#     net = network.Network(n=n)

#     loss, loss_2, error, outputs, targets, target_times, test_outputs, test_targets, class_nums, diff, generation_outputs, generation_targets = net.train(f_etas, None, n_epochs, update_b_weights=False, generate_activity=True)

#     losses[i] = loss
#     losses_2[i] = loss_2
#     errors[i] = error
#     diffs[i]  = diff

# for l in range(n_epochs*network.n_sequences):
#     avg_losses[:, :, l] = np.mean(losses[:, :, int(l*network.sequence_length/100.0):int((l+1)*network.sequence_length/100.0)], axis=-1)
#     avg_losses_2[:, :, l] = np.mean(losses_2[:, :, int(l*network.sequence_length/100.0):int((l+1)*network.sequence_length/100.0)], axis=-1)

# suffix = "hidden"

# np.save(os.path.join(folder, "losses_{}.npy".format(suffix)), losses)
# np.save(os.path.join(folder, "avg_losses_{}.npy".format(suffix)), avg_losses)
# np.save(os.path.join(folder, "losses_2_{}.npy".format(suffix)), losses_2)
# np.save(os.path.join(folder, "avg_losses_2_{}.npy".format(suffix)), avg_losses_2)
# np.save(os.path.join(folder, "errors_{}.npy".format(suffix)), errors)
# np.save(os.path.join(folder, "outputs_{}.npy".format(suffix)), outputs)
# np.save(os.path.join(folder, "targets_{}.npy".format(suffix)), targets)
# np.save(os.path.join(folder, "target_times_{}.npy".format(suffix)), target_times)
# np.save(os.path.join(folder, "test_outputs_{}.npy".format(suffix)), test_outputs)
# np.save(os.path.join(folder, "test_targets_{}.npy".format(suffix)), test_targets)
# np.save(os.path.join(folder, "class_nums_{}.npy".format(suffix)), class_nums)
# np.save(os.path.join(folder, "diffs_{}.npy".format(suffix)), diffs)
# np.save(os.path.join(folder, "generation_outputs_{}.npy".format(suffix)), generation_outputs)
# np.save(os.path.join(folder, "generation_targets_{}.npy".format(suffix)), generation_targets)

# # ----- ONE HIDDEN LAYER, OUTPUT ONLY WEIGHT UPDATES ----- #

# # # feedforward learning rates
# # f_etas = [0.2, 0.05]

# # # number of units per layer (including input layer)
# # n = [500, 300, 3]

# # initalize array to hold losses
# losses       = np.zeros((n_trials, len(n)-1, int(n_epochs*network.n_sequences*network.sequence_length/100.0)))
# losses_2     = np.zeros((n_trials, len(n)-1, int(n_epochs*network.n_sequences*network.sequence_length/100.0)))
# avg_losses   = np.zeros((n_trials, len(n)-1, n_epochs*network.n_sequences))
# avg_losses_2 = np.zeros((n_trials, len(n)-1, n_epochs*network.n_sequences))
# errors       = np.zeros((n_trials, network.n_classes))
# diffs        = np.zeros((n_trials, network.n_classes, int(network.sequence_length/2.0)))

# for i in range(n_trials):
#     print("No hidden layers. Trial {:>2d}/{:>2d}. --------------------".format(i+1, n_trials))

#     # create the network
#     net = network.Network(n=n)

#     loss, loss_2, error, outputs, targets, target_times, test_outputs, test_targets, class_nums, diff, generation_outputs, generation_targets = net.train(f_etas, None, n_epochs, update_b_weights=False, generate_activity=True, update_hidden_weights=False)

#     losses[i] = loss
#     losses_2[i] = loss_2
#     errors[i] = error
#     diffs[i]  = diff

# for l in range(n_epochs*network.n_sequences):
#     avg_losses[:, :, l] = np.mean(losses[:, :, int(l*network.sequence_length/100.0):int((l+1)*network.sequence_length/100.0)], axis=-1)
#     avg_losses_2[:, :, l] = np.mean(losses_2[:, :, int(l*network.sequence_length/100.0):int((l+1)*network.sequence_length/100.0)], axis=-1)

# suffix = "hidden_no_update"

# np.save(os.path.join(folder, "losses_{}.npy".format(suffix)), losses)
# np.save(os.path.join(folder, "avg_losses_{}.npy".format(suffix)), avg_losses)
# np.save(os.path.join(folder, "losses_2_{}.npy".format(suffix)), losses_2)
# np.save(os.path.join(folder, "avg_losses_2_{}.npy".format(suffix)), avg_losses_2)
# np.save(os.path.join(folder, "errors_{}.npy".format(suffix)), errors)
# np.save(os.path.join(folder, "outputs_{}.npy".format(suffix)), outputs)
# np.save(os.path.join(folder, "targets_{}.npy".format(suffix)), targets)
# np.save(os.path.join(folder, "target_times_{}.npy".format(suffix)), target_times)
# np.save(os.path.join(folder, "test_outputs_{}.npy".format(suffix)), test_outputs)
# np.save(os.path.join(folder, "test_targets_{}.npy".format(suffix)), test_targets)
# np.save(os.path.join(folder, "class_nums_{}.npy".format(suffix)), class_nums)
# np.save(os.path.join(folder, "diffs_{}.npy".format(suffix)), diffs)
# np.save(os.path.join(folder, "generation_outputs_{}.npy".format(suffix)), generation_outputs)
# np.save(os.path.join(folder, "generation_targets_{}.npy".format(suffix)), generation_targets)

# ----- ONE HIDDEN LAYER, BACKWARD WEIGHT UPDATES ----- #

# # feedforward learning rates
f_etas = [10.0, 0.01]

# feedback learning rates
b_etas = [0.001]

# # number of units per layer (including input layer)
n = [784, 300, 10]

# initalize array to hold losses
losses       = np.zeros((n_trials, len(n)-1, int(n_epochs*network.n_sequences*network.sequence_length/100.0)))
losses_2     = np.zeros((n_trials, len(n)-1, int(n_epochs*network.n_sequences*network.sequence_length/100.0)))
avg_losses   = np.zeros((n_trials, len(n)-1, n_epochs*network.n_sequences))
avg_losses_2 = np.zeros((n_trials, len(n)-1, n_epochs*network.n_sequences))
errors       = np.zeros((n_trials, network.n_classes))
diffs        = np.zeros((n_trials, network.n_classes, int(network.sequence_length/2.0)))

for i in range(n_trials):
    print("1 hidden layer. Trial {:>2d}/{:>2d}. --------------------".format(i+1, n_trials))

    # create the network
    net = network.Network(n=n)

    loss, loss_2, error, outputs, targets, target_times, test_outputs, test_targets, class_nums = net.train(f_etas, b_etas, n_epochs, update_b_weights=False, generate_activity=False, trial=i)

    losses[i] = loss
    losses_2[i] = loss_2
    errors[i] = error

    print(errors)

for l in range(n_epochs*network.n_sequences):
    avg_losses[:, :, l] = np.mean(losses[:, :, int(l*network.sequence_length/100.0):int((l+1)*network.sequence_length/100.0)], axis=-1)
    avg_losses_2[:, :, l] = np.mean(losses_2[:, :, int(l*network.sequence_length/100.0):int((l+1)*network.sequence_length/100.0)], axis=-1)

suffix = ""

np.save(os.path.join(folder, "losses{}.npy".format(suffix)), losses)
np.save(os.path.join(folder, "avg_losses{}.npy".format(suffix)), avg_losses)
np.save(os.path.join(folder, "losses_2{}.npy".format(suffix)), losses_2)
np.save(os.path.join(folder, "avg_losses_2{}.npy".format(suffix)), avg_losses_2)
np.save(os.path.join(folder, "errors{}.npy".format(suffix)), errors)
np.save(os.path.join(folder, "outputs{}.npy".format(suffix)), outputs)
np.save(os.path.join(folder, "targets{}.npy".format(suffix)), targets)
np.save(os.path.join(folder, "target_times{}.npy".format(suffix)), target_times)
np.save(os.path.join(folder, "test_outputs{}.npy".format(suffix)), test_outputs)
np.save(os.path.join(folder, "test_targets{}.npy".format(suffix)), test_targets)
np.save(os.path.join(folder, "class_nums{}.npy".format(suffix)), class_nums)
np.save(os.path.join(folder, "diffs{}.npy".format(suffix)), diffs)
np.save(os.path.join(folder, "generation_outputs{}.npy".format(suffix)), generation_outputs)
np.save(os.path.join(folder, "generation_targets{}.npy".format(suffix)), generation_targets)