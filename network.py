import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

import utils
from plotter import Plotter, SigmoidLimitsPlotter
import os
import datetime
import torch
from torch.autograd import Variable
from comet_ml import Experiment
import time
import misc
import json
import shutil

use_comet             = False # whether to use Comet.ml
comet_experiment_name = "Multiplexing"

cuda = torch.cuda.is_available()
if cuda:
    dtype = torch.cuda.FloatTensor
    print("Using CUDA.")
else:
    dtype = torch.FloatTensor
    print("Not using CUDA.")

# training variables
validation   = True  # whether to use validation set
dynamic_plot = False # whether to plot variables as training occurs
n_epochs     = 50    # number of epochs

# number of layers
n_layers = 3

# hyperparameters
if n_layers == 2: # No hidden layers
    n_units           = [784, 10]
    W_std             = [0, 0.01]
    Z_std             = [0]
    Y_std             = [0]
    f_etas            = [0, 0.1]
    r_etas            = [0]
    b_etas            = [0]
    output_burst_prob = 0.2
    min_Z             = 0.1
    u_range           = 2
    W_decay           = 0
elif n_layers == 3: # One hidden layer
    n_units           = [784, 500, 10]
    W_std             = [0, 0.05, 0.01]
    Z_std             = [0, 0.01]
    Y_std             = [0, 0.01]
    f_etas            = [0, 2.0, 0.01]
    r_etas            = [0, 0.0]
    b_etas            = [0, 0.0]
    output_burst_prob = 0.2
    min_Z             = 0.1
    u_range           = 2
    W_decay           = 0
elif n_layers == 4: # Two hidden layers
    n_units           = [784, 500, 300, 10]
    W_std             = [0, 0.05, 0.05, 0.01]
    Z_std             = [0, 0.01, 0.01]
    Y_std             = [0, 1.0, 1.0]
    f_etas            = [0, 1.0, 1.0, 0.01]
    b_etas            = [0, 0.001, 0.001]
    r_etas            = [0, 0.0, 0.0]
    output_burst_prob = 0.2
    min_Z             = 0.1
    u_range           = 2
    W_decay           = 0

if type(f_etas) in (int, float):
    f_etas = [0] + [ f_etas for i in range(1, n_layers) ]
if type(r_etas) in (int, float):
    r_etas = [0] + [ r_etas for i in range(1, n_layers-1) ]
if type(b_etas) in (int, float):
    b_etas = [0] + [ b_etas for i in range(1, n_layers-1) ]

# number of training & testing examples
if validation:
    n_examples      = 50000
    n_test_examples = 10000
else:
    n_examples      = 60000
    n_test_examples = 10000

# load MNIST data
x_set, t_set, x_test_set, t_test_set = utils.load_mnist_data(n_examples, n_test_examples, validation=validation)
x_set      = torch.from_numpy(x_set).type(dtype)
t_set      = torch.from_numpy(t_set).type(dtype)
x_test_set = torch.from_numpy(x_test_set).type(dtype)
t_test_set = torch.from_numpy(t_test_set).type(dtype)

def softplus(x, limit=30):
    return torch.nn.functional.softplus(Variable(x)).data

def softplus_deriv(x):
    return torch.nn.functional.sigmoid(Variable(x)).data
    # y = x.clone()
    # y[y > 0.5] = 1
    # y[y <= 0.5] = 0
    # return y

def relu(x, baseline=0):
    return torch.nn.functional.relu(Variable(x)).data

def relu_deriv(x, baseline=0):
    return torch.gt(x, baseline).type(dtype)

def forward(W, b, v, h, f_input):
    h[0] = f_input.unsqueeze(1)

    for i in range(1, n_layers):
        v[i] = W[i].mm(h[i-1]) + b[i]

        if i < n_layers-1:
            h[i] = softplus(v[i])
        else:
            h[i] = relu(v[i])

def backward(Y, Z, W, b, u, u_t, p, p_t, beta, beta_t, v, h, mean_c, t_input, ):
    cost   = [0] + [ 0 for i in range(1, n_layers) ]
    cost_Y = [0] + [ 0 for i in range(1, n_layers-1) ]
    cost_Z = [0] + [ 0 for i in range(1, n_layers-1) ]

    delta_W = [0] + [ 0 for i in range(1, n_layers) ]
    delta_b = [0] + [ 0 for i in range(1, n_layers) ]
    delta_Y = [0] + [ 0 for i in range(1, n_layers-1) ]
    delta_Z = [0] + [ 0 for i in range(1, n_layers-1) ]

    max_u = [0] + [ 0 for i in range(1, n_layers-1) ]

    delta_b_backprop = [0] + [ 0 for i in range(1, n_layers) ]

    for i in range(n_layers-1, 0, -1):
        if i == n_layers-1:
            beta[i]   = output_burst_prob*h[i]
            beta_t[i] = output_burst_prob*t_input.unsqueeze(1)

            cost[i]    = 0.5*torch.sum((beta_t[i] - beta[i])**2)
            e          = -(beta_t[i] - beta[i])*output_burst_prob*relu_deriv(h[i])
            delta_W[i] = e.mm(h[i-1].transpose(0, 1))
            delta_b[i] = e

            delta_b_backprop[i] = -(beta_t[i] - beta[i])*output_burst_prob*relu_deriv(h[i])
        else:
            c = Z[i].mm(h[i])

            mean_c[i] = 0.5*mean_c[i] + 0.5*c

            if i == n_layers-2:
                u[i]   = W[i+1].transpose(0, 1).mm(beta[i+1]*output_burst_prob*relu_deriv(beta[i+1]))
                u_t[i] = W[i+1].transpose(0, 1).mm(beta_t[i+1]*output_burst_prob*relu_deriv(beta_t[i+1]))
            else:
                u[i]   = W[i+1].transpose(0, 1).mm(p[i+1]*softplus_deriv(v[i+1]))
                u_t[i] = W[i+1].transpose(0, 1).mm(p_t[i+1]*softplus_deriv(v[i+1]))

            max_u[i] = torch.sum(torch.abs(Y[i]), dim=1).unsqueeze(1)/mean_c[i]

            p[i]   = u[i]
            p_t[i] = u_t[i]

            beta[i]   = p[i]*h[i]
            beta_t[i] = p_t[i]*h[i]

            cost[i]   = 0.5*torch.sum((beta_t[i] - beta[i])**2) + 0.5*torch.sum((W[i])**2)
            cost_Z[i] = 0.5*torch.sum((u[i])**2) + 0.5*torch.sum((min_Z - Z[i])**2)
            cost_Y[i] = 0.5*torch.sum((u_range - max_u[i])**2)

            e          = -(p_t[i] - p[i])*softplus_deriv(v[i])
            delta_W[i] = e.mm(h[i-1].transpose(0, 1))
            delta_b[i] = e

            delta_b_backprop[i] = W[i+1].transpose(0, 1).mm(delta_b_backprop[i+1])*softplus_deriv(v[i])

            e_Y = -(u_range - max_u[i])/mean_c[i]
            delta_Y[i] = torch.sign(Y[i]).transpose(0, 1).mm(e_Y).transpose(0, 1)

            delta_Z[i] = (-u[i])*(u[i]/c).mm(h[i].transpose(0, 1)) - (min_Z - Z[i])

    return cost, cost_Y, cost_Z, delta_W, delta_b, delta_Y, delta_Z, max_u, delta_b_backprop

def update_weights(W, b, Y, Z, delta_W, delta_b, delta_Y, delta_Z):
    for i in range(1, n_layers):
        W[i] -= f_etas[i]*(delta_W[i] + W_decay*W[i])
        b[i] -= f_etas[i]*(delta_b[i] + W_decay*b[i])

        if i < n_layers-1:
            Y[i] -= b_etas[i]*delta_Y[i]
            Z[i] -= r_etas[i]*delta_Z[i]
            Z[i][Z[i] < 0] = 0

def train(folder_prefix=None, continuing_folder=None):
    if folder_prefix is not None:
        n_units_string = " ".join([ str(i) for i in n_units[1:] ])
        f_etas_string  = " ".join([ str(i) for i in f_etas[1:] ])
        b_etas_string  = " ".join([ str(i) for i in b_etas[1:] ])
        r_etas_string  = " ".join([ str(i) for i in r_etas[1:] ])
        W_std_string   = " ".join([ str(i) for i in W_std[1:] ])
        Z_std_string   = " ".join([ str(i) for i in Z_std[1:] ])
        Y_std_string   = " ".join([ str(i) for i in Y_std[1:] ])

        folder = "{} - {} - {} - {} - {} - {} - {} - {} - {} - {} - {}".format(folder_prefix, n_units_string, f_etas_string, b_etas_string, r_etas_string, W_std_string, Z_std_string, Y_std_string, output_burst_prob, min_Z, u_range)
    else:
        folder = None

    if folder is not None and folder == continuing_folder:
        print("Error: If you're continuing a simulation, the new results need to be saved in a different directory.")
        raise

    if folder is not None:
        if not os.path.exists(folder):
            os.makedirs(folder)

        timestamp = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        with open(os.path.join(folder, "params.txt"), "w") as f:
            f.write("Simulation run @ {}\n".format(timestamp))
            if continuing_folder is not None:
                f.write("Continuing from \"{}\"\n".format(continuing_folder))
            f.write("Number of epochs: {}\n".format(n_epochs))
            f.write("Number of training examples: {}\n".format(n_examples))
            f.write("Number of testing examples: {}\n".format(n_test_examples))
            f.write("Using validation set: {}\n".format(validation))
            f.write("Number of units in each layer: {}\n".format(n_units))
            f.write("Feedforward learning rates: {}\n".format(f_etas))
            f.write("Feedback learning rates: {}\n".format(b_etas))
            f.write("Recurrent learning rates: {}\n".format(r_etas))
            f.write("W standard deviation: {}\n".format(W_std))
            f.write("Y standard deviation: {}\n".format(Y_std))
            f.write("Z standard deviation: {}\n".format(Z_std))
            f.write("Output layer burst probability: {}\n".format(output_burst_prob))
            f.write("Minimum Z value: {}\n".format(min_Z))
            f.write("Max apical potential: {}\n".format(u_range))

        filename = os.path.basename(__file__)
        if filename.endswith('pyc'):
            filename = filename[:-1]
        shutil.copyfile(filename, os.path.join(folder, filename))

        params_dict = {'timestamp'        : timestamp,
                       'continuing_folder': continuing_folder,
                       'n_epochs'         : n_epochs,
                       'n_examples'       : n_examples,
                       'n_test_examples'  : n_test_examples,
                       'validation'       : validation,
                       'n_units'          : n_units,
                       'W_std'            : W_std,
                       'Z_std'            : Z_std,
                       'Y_std'            : Y_std,
                       'f_etas'           : f_etas,
                       'r_etas'           : r_etas,
                       'b_etas'           : b_etas,
                       'output_burst_prob': output_burst_prob,
                       'min_Z'            : min_Z,
                       'u_range'          : u_range,
                       'W_decay'          : W_decay}

        with open(os.path.join(folder, "params.json"), 'w') as f:
            json.dump(params_dict, f)

    W, b, Y, Z, v, h, u, u_t, p, p_t, beta, beta_t, mean_c = create_dynamic_variables(symmetric_weights=False)
    if continuing_folder is not None:
        W, b, Y, Z, mean_c = load_dynamic_variables(continuing_folder)

    costs               = np.zeros((n_layers, n_epochs*n_examples))
    avg_costs           = np.zeros((n_layers, n_epochs*int(n_examples//1000)))
    test_costs          = np.zeros(n_epochs*int(n_examples//1000)+1)
    backprop_angles     = np.zeros((n_layers-2, n_epochs*n_examples))
    avg_backprop_angles = np.zeros((n_layers-2, n_epochs*int(n_examples//1000)))
    min_us              = np.zeros((n_layers-2, n_epochs*n_examples))
    max_us              = np.zeros((n_layers-2, n_epochs*n_examples))
    min_hs              = np.zeros((n_layers-1, n_epochs*n_examples))
    max_hs              = np.zeros((n_layers-1, n_epochs*n_examples))
    errors              = np.zeros(n_epochs*int(n_examples//1000)+1)

    if use_comet:
        hyper_params = {
            "output_burst_prob": output_burst_prob,
            "min_Z": min_Z,
            "u_range": u_range,
            "n_epochs": n_epochs,
            "n_examples": n_examples
        }
        for i in range(1, n_layers):
            hyper_params["n_units_{}".format(i)] = n_units[i]
            hyper_params["W_std_{}".format(i)]   = W_std[i]
            hyper_params["f_etas_{}".format(i)]  = f_etas[i]
            if i < n_layers-1:
                hyper_params["Z_std_{}".format(i)]   = Z_std[i]
                hyper_params["Y_std_{}".format(i)]   = Y_std[i]
                hyper_params["b_etas_{}".format(i)]  = b_etas[i]
                hyper_params["r_etas_{}".format(i)]  = r_etas[i]

        experiment = Experiment(api_key=misc.comet_api_key, project_name=comet_experiment_name)
        experiment.log_multiple_params(hyper_params)

    # make a list of training example indices
    example_indices = np.arange(n_examples)

    # calculate the initial test error as a percentage
    errors[0], test_costs[0] = test(W, b)
    print("Initial test error: {}%.".format(errors[0]))

    if use_comet:
        with experiment.validate():
            experiment.log_metric("accuracy", 100 - errors[0], step=0)
            experiment.log_metric("cost", test_costs[0], step=0)

    if folder is not None:
        save_dynamic_variables(folder, W, b, Y, Z, mean_c)
        save_results(folder, costs, backprop_angles, errors, test_costs)

    for epoch_num in range(n_epochs):
        start_time = time.time()

        if use_comet:
            experiment.log_current_epoch(epoch_num)

        # shuffle which examples to show
        np.random.shuffle(example_indices)

        train_error = 0

        for example_num in range(n_examples):
            example_index = example_indices[example_num]

            abs_ex_num = epoch_num*n_examples + example_num

            # get input and target for this example
            x = x_set[:, example_index]
            t = t_set[:, example_index]

            forward(W, b, v, h, f_input=x)

            # get the predicted & target class
            predicted_class = int(torch.max(h[-1], 0)[1])
            target_class    = int(torch.max(t, 0)[1])

            # update the train error
            if predicted_class != target_class:
                train_error += 1

            cost, cost_Y, cost_Z, delta_W, delta_b, delta_Y, delta_Z, max_u, delta_b_backprop = backward(Y, Z, W, b, u, u_t, p, p_t, beta, beta_t, v, h, mean_c, t_input=t)
            costs[:, epoch_num*n_examples + example_num] = cost

            backprop_angle = np.array([ (180/np.pi)*np.arccos(delta_b_backprop[i].squeeze().dot(delta_b[i].squeeze())/(1e-10 + torch.norm(delta_b_backprop[i])*torch.norm(delta_b[i]))) for i in range(1, n_layers-1) ])
            backprop_angles[:, epoch_num*n_examples + example_num] = backprop_angle
            min_us[:, epoch_num*n_examples + example_num] = np.array([ min(torch.min(u[i]), torch.min(u_t[i])) for i in range(1, n_layers-1) ])
            max_us[:, epoch_num*n_examples + example_num] = np.array([ max(torch.max(u[i]), torch.max(u_t[i])) for i in range(1, n_layers-1) ])
            min_hs[:, epoch_num*n_examples + example_num] = np.array([ torch.min(h[i]) for i in range(1, n_layers) ])
            max_hs[:, epoch_num*n_examples + example_num] = np.array([ torch.max(h[i]) for i in range(1, n_layers) ])

            if use_comet:
                with experiment.train():
                    experiment.log_metric("loss", float(cost[-1]), step=example_num+1)
                    experiment.log_metric("accuracy", 100.0*(1 - train_error/(example_num+1)), step=example_num+1)

                    for i in range(1, n_layers):
                        if i < n_layers-1:
                            experiment.log_metric("bp_angle_{}".format(i), backprop_angle[i-1], step=abs_ex_num+1)
                            experiment.log_metric("min_u_{}", min_us[i-1, abs_ex_num], step=abs_ex_num+1)
                            experiment.log_metric("max_u_{}", max_us[i-1, abs_ex_num], step=abs_ex_num+1)
                        experiment.log_metric("min_h_{}", min_hs[i-1, abs_ex_num], step=abs_ex_num+1)
                        experiment.log_metric("max_h_{}", max_hs[i-1, abs_ex_num], step=abs_ex_num+1)

            update_weights(W, b, Y, Z, delta_W, delta_b, delta_Y, delta_Z)

            if (example_num+1) % 1000 == 0:
                index = epoch_num*int(n_examples//1000) + int((example_num+1)//1000)

                errors[index], test_costs[index] = test(W, b)

                avg_backprop_angles[:, index-1] = np.mean(backprop_angles[:, abs_ex_num-999:abs_ex_num+1], axis=1)
                avg_costs[:, index-1]           = np.mean(costs[:, abs_ex_num-999:abs_ex_num+1], axis=1)

                # print test error
                print("Epoch {}, ex {}. Test Error: {}%. Test Cost: {}. Train Cost: {}.".format(epoch_num+1, example_num+1, errors[index], test_costs[index], avg_costs[-1, index-1]))

                for i in range(1, n_layers-1):
                    print("Layer {}. BPA: {:.1f}. u: {:.4f} to {:.4f}. h: {:.4f} to {:.4f}".format(i, avg_backprop_angles[i-1, index-1], np.mean(min_us[i-1, abs_ex_num-999:abs_ex_num+1]), np.mean(max_us[i-1, abs_ex_num-999:abs_ex_num+1]), np.mean(min_hs[i-1, abs_ex_num-999:abs_ex_num+1]), np.mean(max_hs[i-1, abs_ex_num-999:abs_ex_num+1])))
                print("Layer {}. h: {:.4f} to {:.4f}".format(n_layers-1, np.mean(min_hs[-1, abs_ex_num-999:abs_ex_num+1]), np.mean(max_hs[-1, abs_ex_num-999:abs_ex_num+1])))

                if use_comet:
                    with experiment.validate():
                        experiment.log_metric("accuracy", 100 - errors[index], step=abs_ex_num)
                        experiment.log_metric("cost", test_costs[index], step=abs_ex_num)

                if folder is not None:
                    save_dynamic_variables(folder, W, b, Y, Z, mean_c)
                    save_results(folder, avg_costs, avg_backprop_angles, errors, test_costs)

        end_time = time.time()

        print("Elapsed time: {} s.".format(end_time - start_time))

    return costs, backprop_angles, errors

def test(W, b):
    v = [0] + [ torch.from_numpy(np.zeros(n_units[i])).type(dtype) for i in range(1, n_layers) ]
    h = [0] + [ torch.from_numpy(np.zeros(n_units[i])).type(dtype) for i in range(1, n_layers) ]

    # initialize error
    error = 0
    cost  = 0
    for i in range(n_test_examples):
        # get input and target for this test example
        x = x_test_set[:, i]
        t = t_test_set[:, i]

        # do a forward pass
        forward(W, b, v, h, f_input=x)

        # compute cost
        beta   = output_burst_prob*h[-1]
        beta_t = output_burst_prob*t.unsqueeze(1)
        cost  += 0.5*torch.sum((beta_t - beta)**2)

        # get the predicted & target class
        predicted_class = int(torch.max(h[-1], 0)[1])
        target_class    = int(torch.max(t, 0)[1])

        # update the test error
        if predicted_class != target_class:
            error += 1

    cost /= n_test_examples

    return 100.0*error/n_test_examples, cost

def plot_backprop_angles(backprop_angles, filename=None):
    plt.figure()
    for i in range(1, n_layers-1):
        plt.plot(torch.mean(backprop_angles[:, i-1, :], axis=0))

    if filename is not None:
        plt.savefig(filename + ".png")
        plt.savefig(filename + ".svg")
    else:
        plt.show()

def gradient_check():
    W, b, Y, Z, v, h, u, u_t, p, p_t, beta, beta_t, mean_c = create_dynamic_variables(symmetric_weights=True)

    # get input and target
    x = x_set[:, 0]
    t = t_set[:, 0]

    # get the calculated delta values
    forward(W, b, v, h, f_input=x)
    cost, cost_Y, cost_Z, delta_W, delta_b, delta_Y, delta_Z, max_u, delta_b_backprop = backward(Y, Z, W, b, u, u_t, p, p_t, beta, beta_t, v, h, mean_c, t_input=t)

    # decrease and increase weights slightly, compare with delta values
    numerical_delta_W = [0] + [ torch.from_numpy(np.zeros(W[i].shape)).type(dtype) for i in range(1, n_layers) ]
    epsilon = 0.0001
    for i in range(1, n_layers):
        for j in range(W[i].shape[0]):
            for k in range(W[i].shape[1]):
                percent = 100*(j*W[i].shape[1] + k)/(W[i].shape[0]*W[i].shape[1])
                if percent % 5 == 0:
                    print("Computing numerical gradient for layer {}... {}% done.".format(i, percent))

                W[i][j, k] -= epsilon

                if i > 1:
                    Y[i-1][k, j] -= epsilon

                forward(W, b, v, h, f_input=x)
                beta   = output_burst_prob*h[-1]
                beta_t = output_burst_prob*t
                cost_minus = 0.5*torch.sum((beta_t - beta)**2)

                W[i][j, k] += 2*epsilon

                if i > 1:
                    Y[i-1][k, j] += 2*epsilon

                forward(W, b, v, h, f_input=x)
                beta   = output_burst_prob*h[-1]
                beta_t = output_burst_prob*t
                cost_plus = 0.5*torch.sum((beta_t - beta)**2)

                numerical_delta_W[i][j, k] = (cost_plus - cost_minus)/(2*epsilon)

                W[i][j, k] -= epsilon

                if i > 1:
                    Y[i-1][k, j] -= epsilon

    print([ torch.mean(torch.abs((delta_W[i] - numerical_delta_W[i]))) for i in range(1, n_layers) ])


def create_training_data():
    # load MNIST data
    x_set, t_set, x_test_set, t_test_set = utils.load_mnist_data(n_examples, n_test_examples, validation=validation)
    x_set      = torch.from_numpy(x_set).type(dtype)
    t_set      = torch.from_numpy(t_set).type(dtype)
    x_test_set = torch.from_numpy(x_test_set).type(dtype)
    t_test_set = torch.from_numpy(t_test_set).type(dtype)

def create_dynamic_variables(symmetric_weights=False):
    # create network variables
    W      = [0] + [ torch.from_numpy(np.random.normal(0, W_std[i], size=(n_units[i], n_units[i-1]))).type(dtype) for i in range(1, n_layers) ]
    W[-1] += 0.001
    b      = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers) ]
    if symmetric_weights:
        Y  = [0] + [ torch.from_numpy(W[i+1].T.copy()).type(dtype) for i in range(1, n_layers-1) ]
    else:
        Y  = [0] + [ torch.from_numpy(np.random.normal(0, Y_std[i], size=(n_units[i], n_units[i+1]))).type(dtype) for i in range(1, n_layers-1) ]
    Z      = [0] + [ torch.from_numpy(np.random.uniform(0, Z_std[i], size=(n_units[i], n_units[i]))).type(dtype) for i in range(1, n_layers-1) ]
    v      = [0] + [ torch.from_numpy(np.zeros(n_units[i])).type(dtype) for i in range(1, n_layers) ]
    h      = [0] + [ torch.from_numpy(np.zeros(n_units[i])).type(dtype) for i in range(1, n_layers) ]
    u      = [0] + [ torch.from_numpy(np.zeros(n_units[i])).type(dtype) for i in range(1, n_layers) ]
    u_t    = [0] + [ torch.from_numpy(np.zeros(n_units[i])).type(dtype) for i in range(1, n_layers) ]
    p      = [0] + [ torch.from_numpy(np.zeros(n_units[i])).type(dtype) for i in range(1, n_layers) ]
    p_t    = [0] + [ torch.from_numpy(np.zeros(n_units[i])).type(dtype) for i in range(1, n_layers) ]
    beta   = [0] + [ torch.from_numpy(np.zeros(n_units[i])).type(dtype) for i in range(1, n_layers) ]
    beta_t = [0] + [ torch.from_numpy(np.zeros(n_units[i])).type(dtype) for i in range(1, n_layers) ]
    mean_c = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers-1) ]

    return W, b, Y, Z, v, h, u, u_t, p, p_t, beta, beta_t, mean_c

def load_dynamic_variables(path):
    state = torch.load(os.path.join(path, "state.dat"))

    W      = state['W']
    b      = state['b']
    Y      = state['Y']
    Z      = state['Z']
    mean_c = state['mean_c']

    return W, b, Y, Z, mean_c

def save_dynamic_variables(path, W, b, Y, Z, mean_c):
    state = {
        'W': W,
        'b': b,
        'Y': Y,
        'Z': Z,
        'mean_c': mean_c
    }
    torch.save(state, os.path.join(path, "state.dat"))

def save_results(path, avg_costs, avg_backprop_angles, errors, test_costs):
    np.savetxt(os.path.join(path, "avg_costs.csv"), avg_costs, delimiter=",", fmt='%.5f')
    np.savetxt(os.path.join(path, "test_costs.csv"), test_costs, delimiter=",", fmt='%.5f')
    np.savetxt(os.path.join(path, "avg_backprop_angles.csv"), avg_backprop_angles, delimiter=",", fmt='%.5f')
    np.savetxt(os.path.join(path, "errors.csv"), errors, delimiter=",", fmt='%.2f')
