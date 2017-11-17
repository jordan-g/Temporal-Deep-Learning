from train import train
import utils

# folder in which to save results
folder = "cifar10_testing_long_4"

# number of epochs of training
n_epochs = 100

# number of trials to repeat training
n_trials = 1

# initial weight magnitudes
Y_range = 1.0
W_range = 0.1

x_set, t_set, x_test_set, t_test_set = utils.load_cifar10_data(50000, 10000, validation=False, cuda=True)

# --- Train with no hidden layers --- #

# n_hidden_units = []         # number of units per hidden layer
# f_etas         = [0.1]      # feedforward learning rates
# suffix         = "0_hidden" # suffix to append to files

# train(n_epochs, f_etas, n_hidden_units, W_range, Y_range, folder, n_trials=n_trials, suffix=suffix, validation=False, dataset="CIFAR10", cuda=True, x_set=x_set, t_set=t_set, x_test_set=x_test_set, t_test_set=t_test_set)

# --- Train with 1 hidden layer --- #

# n_hidden_units = [1000]
# f_etas         = [0.1, 0.1]
# suffix         = "1_hidden"

# train(n_epochs, f_etas, n_hidden_units, W_range, Y_range, folder, n_trials=n_trials, suffix=suffix, validation=False, dataset="CIFAR10", cuda=True, x_set=x_set, t_set=t_set, x_test_set=x_test_set, t_test_set=t_test_set)

# --- Train with 2 hidden layers --- #

n_hidden_units = [1000, 1000]
f_etas         = [0.1, 0.1, 0.1]
suffix         = "2_hidden"

train(n_epochs, f_etas, n_hidden_units, W_range, Y_range, folder, n_trials=n_trials, suffix=suffix, validation=False, dataset="CIFAR10", cuda=True, x_set=x_set, t_set=t_set, x_test_set=x_test_set, t_test_set=t_test_set)
