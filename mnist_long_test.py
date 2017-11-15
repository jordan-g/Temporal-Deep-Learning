from train import train

# folder in which to save results
folder = "mnist_testing_long_3"

# number of epochs of training
n_epochs = 50

# number of trials to repeat training
n_trials = 5

# initial weight magnitudes
Y_range = 1.0
W_range = 0.1

# --- Train with no hidden layers --- #

n_hidden_units = []         # number of units per hidden layer
f_etas         = [0.1]      # feedforward learning rates
suffix         = "0_hidden" # suffix to append to files

train(n_epochs, f_etas, n_hidden_units, W_range, Y_range, folder, n_trials=n_trials, validation=False, suffix=suffix, cuda=True)

# # --- Train with 1 hidden layer --- #

n_hidden_units = [500]
f_etas         = [0.1, 0.1]
suffix         = "1_hidden"

train(n_epochs, f_etas, n_hidden_units, W_range, Y_range, folder, n_trials=n_trials, validation=False, suffix=suffix, cuda=True)

# --- Train with 2 hidden layers --- #

n_hidden_units = [500, 300]
f_etas         = [0.1, 0.1, 0.1]
suffix         = "2_hidden"

train(n_epochs, f_etas, n_hidden_units, W_range, Y_range, folder, n_trials=n_trials, validation=False, suffix=suffix, cuda=True)