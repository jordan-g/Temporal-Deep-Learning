import network_simple as network
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import utils
import time
import datetime

def test(net, x_test_set, t_test_set, n_test_examples, n_layers, trial_num, epoch_num):
    # make a list of testing example indices
    test_example_indices = np.arange(n_test_examples)

    # initialize error
    error = 0

    for test_example_num in range(n_test_examples):
        test_example_index = test_example_indices[test_example_num]

        # get input and target for this test example
        x = x_test_set[:, test_example_index]
        t = t_test_set[:, test_example_index]

        # do a forward pass
        net.forward(x)

        # get the predicted & target class
        _, predicted_class = torch.max(net.layers[-1].event_rate, 0)
        _, target_class    = torch.max(t, 0)

        # update the test error
        if predicted_class != target_class:
            error += 1

    return 100.0*error/n_test_examples

def train(n_epochs, f_etas, n_hidden_units, W_range, Y_range, folder, suffix="", n_trials=1, validation=True, dataset="MNIST", cuda=False):
    if dataset == "MNIST":
        # number of input & output neurons
        n_in  = 784
        n_out = 10

        # number of training & testing examples
        if validation:
            n_examples      = 50000
            n_test_examples = 10000
        else:
            n_examples      = 60000
            n_test_examples = 10000

        # load MNIST data
        x_set, t_set, x_test_set, t_test_set = utils.load_mnist_data(n_examples, n_test_examples, validation=validation, cuda=cuda)
    elif dataset == "CIFAR10":
        n_in  = 3072
        n_out = 10

        # number of training & testing examples
        if validation:
            n_examples      = 40000
            n_test_examples = 10000
        else:
            n_examples      = 50000
            n_test_examples = 10000

        # load CIFAR10 data
        x_set, t_set, x_test_set, t_test_set = utils.load_cifar10_data(n_examples, n_test_examples, validation=validation, cuda=cuda)

    n_units = n_hidden_units + [n_out]

    print("Layer sizes: {}.".format(n_units))

    # number of layers
    n_layers = len(n_units)

    if not os.path.exists(os.path.join("simulations", folder)):
        os.makedirs(os.path.join("simulations", folder))

    with open(os.path.join("simulations", folder, "params.txt"), "a+") as f:
        f.write("Simulation run @ {}\n".format(datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")))
        f.write("Number of epochs: {}\n".format(n_epochs))
        f.write("Feedforward learning rates: {}\n".format(f_etas))
        f.write("Number of units in each layer: {}\n".format(n_units))
        f.write("W range: {}\n".format(W_range))
        f.write("Y range: {}\n".format(Y_range))
        f.write("Number of trials: {}\n\n".format(n_trials))

    # initialize recording arrays
    losses = np.zeros((n_trials, n_layers, n_epochs*n_examples))
    errors = np.zeros((n_trials, n_epochs+1))

    for trial_num in range(n_trials):
        print("Trial {:>2d}/{:>2d}. --------------------".format(trial_num+1, n_trials))

        # create the network
        net = network.Network(n_units, n_in, W_range, Y_range, cuda=cuda)

        # make a list of training and testing example indices
        example_indices = np.arange(n_examples)

        # calculate the initial test error as a percentage
        errors[trial_num, 0] = test(net, x_test_set, t_test_set, n_test_examples, n_layers, trial_num, 0)

        # print test error
        print("Initial test error: {}.".format(errors[trial_num, 0]))

        # save initial variables
        np.save(os.path.join("simulations", folder, "trial_{}_errors{}.npy".format(trial_num, "_"*(len(suffix)>0) + suffix)), errors)
        for layer_num in range(n_layers):
            np.save(os.path.join("simulations", folder, "trial_{}_f_weights_layer_{}{}.npy".format(trial_num, layer_num, "_"*(len(suffix)>0) + suffix)), net.layers[layer_num].W)
            np.save(os.path.join("simulations", folder, "trial_{}_f_biases_layer_{}{}.npy".format(trial_num, layer_num, "_"*(len(suffix)>0) + suffix)), net.layers[layer_num].b)
            if layer_num != n_layers-1:
                np.save(os.path.join("simulations", folder, "trial_{}_b_weights_layer_{}{}.npy".format(trial_num, layer_num, "_"*(len(suffix)>0) + suffix)), net.layers[layer_num].Y)

        # train the network
        for epoch_num in range(n_epochs):
            start_time = time.time()

            # shuffle which examples to show
            np.random.shuffle(example_indices)

            for example_num in range(n_examples):
                example_index = example_indices[example_num]

                # get input and target for this example
                x = x_set[:, example_index]
                t = t_set[:, example_index]

                # do a forward pass
                net.forward(x)

                # do a backward pass (with weight updates) and record the loss at each layer
                losses[trial_num, :, epoch_num*n_examples + example_num] = net.backward(x, t, f_etas)

                # print progress every 1000 examples
                if (example_num+1) % 1000 == 0:
                    if n_layers > 1:
                        print("{}Trial {:>3d}, epoch {:>3d}, example {:>5d}. Avg. output loss: {:.10f}. Last hidden loss: {:.10f}.".format(suffix + ". "*(len(suffix)>0), trial_num+1, epoch_num+1, example_num+1, np.mean(losses[trial_num, -1, epoch_num*n_examples + example_num - 999:epoch_num*n_examples + example_num]), np.mean(losses[trial_num, -2, epoch_num*n_examples + example_num - 999:epoch_num*n_examples + example_num])))
                    else:
                        print("{}Trial {:>3d}, epoch {:>3d}, example {:>5d}. Avg. output loss: {:.10f}.".format(suffix + ". "*(len(suffix)>0), trial_num+1, epoch_num+1, example_num+1, np.mean(losses[trial_num, -1, epoch_num*n_examples + example_num - 999:epoch_num*n_examples + example_num])))

            # calculate the test error as a percentage
            errors[trial_num, epoch_num+1] = test(net, x_test_set, t_test_set, n_test_examples, n_layers, trial_num, epoch_num+1)

            # print test error
            print("Epoch {} test error: {}.".format(epoch_num+1, errors[trial_num, epoch_num+1]))

            # save variables
            np.save(os.path.join("simulations", folder, "trial_{}_errors{}.npy".format(trial_num, "_"*(len(suffix)>0) + suffix)), errors)
            for layer_num in range(n_layers):
                np.save(os.path.join("simulations", folder, "trial_{}_f_weights_layer_{}{}.npy".format(trial_num, layer_num, "_"*(len(suffix)>0) + suffix)), net.layers[layer_num].W)
                np.save(os.path.join("simulations", folder, "trial_{}_f_biases_layer_{}{}.npy".format(trial_num, layer_num, "_"*(len(suffix)>0) + suffix)), net.layers[layer_num].b)
                if layer_num != n_layers-1:
                    np.save(os.path.join("simulations", folder, "trial_{}_b_weights_layer_{}{}.npy".format(trial_num, layer_num, "_"*(len(suffix)>0) + suffix)), net.layers[layer_num].Y)

            end_time = time.time()

            print("Total time for this epoch: {}s.".format(end_time - start_time))

if __name__ == "__main__":
    # folder in which to save results
    folder = "mnist_testing_simple"

    # number of epochs of training
    n_epochs = 50

    # number of trials to repeat training
    n_trials = 1

    # initial weight magnitudes
    Y_range = 1.0
    W_range = 0.1

    n_hidden_units = [500]      # number of units per hidden layer
    f_etas         = [0.1, 0.1] # feedforward learning rates
    suffix         = "1_hidden" # suffix to append to files

    # train
    train(n_epochs, f_etas, n_hidden_units, W_range, Y_range, folder, n_trials=n_trials, validation=True, suffix=suffix)