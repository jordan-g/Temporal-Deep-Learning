import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

import utils
import os
import datetime
import torch
from torch.autograd import Variable
import time
import json
import shutil
import pdb

# use CUDA if it is available
cuda = torch.cuda.is_available()
if cuda:
    dtype = torch.cuda.FloatTensor
    print("Using CUDA.")
else:
    dtype = torch.FloatTensor
    print("Not using CUDA.")

# set number of training & testing examples
validation = True
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

# -------------------------- ACTIVATION FUNCTIONS -------------------------- #

def softplus(x, limit=30):
    return torch.nn.functional.softplus(Variable(x)).data

def softplus_deriv(x):
    return torch.nn.functional.sigmoid(Variable(x)).data

def sigmoid(x):
    return torch.nn.functional.sigmoid(Variable(x)).data

def hard_deriv(x, mean, variance):
    y = torch.zeros_like(x)
    
    y[torch.le(x, mean+variance)*torch.ge(x, mean-variance)] = 1

    return y

def relu(x, baseline=0):
    return torch.nn.functional.relu(Variable(x)).data

def relu_deriv(x, baseline=0):
    return torch.gt(x, baseline).type(dtype)

# -------------------------------------------------------------------------- #

def forward(W, b, v, h, h_prev, f_input):
    h[0] = f_input.unsqueeze(1)

    for i in range(1, n_layers):
        v[i] = W[i].mm(h_prev[i-1]) + b[i]

        if i < n_layers-1:
            h[i] = sigmoid(v[i])
        else:
            h[i] = relu(v[i])

def forward_test(W, b, v, h, f_input):
    h[0] = f_input.unsqueeze(1)

    for i in range(1, n_layers):
        v[i] = W[i].mm(h[i-1]) + b[i]

        if i < n_layers-1:
            h[i] = sigmoid(v[i])
        else:
            h[i] = relu(v[i])

def backward(Y, Z, W, b, u, p, beta, beta_prev, v, h, h_prev, mean_c, c, t_input):
    for i in range(n_layers-1, 0, -1):
        if i == n_layers-1:
            # compute burst rate
            if t_input is not None:
                h[i] = t_input.unsqueeze(1)

            beta[i] = output_burst_prob*h[i]

            # compute cost
            cost[i]    = 0.5*torch.sum((beta[i] - beta_prev[i])**2)

            # compute feedforward weight update
            e          = -(beta[i] - beta_prev[i])*output_burst_prob*relu_deriv(h_prev[i])
            delta_W[i] = e.mm(h_prev[i-1].transpose(0, 1))
            delta_b[i] = e

            # compute backprop-prescribed weight update
            delta_b_backprop[i] = -(beta[i] - beta_prev[i])*output_burst_prob*relu_deriv(h_prev[i])
        else:
            c[i] = Z[i].mm(h_prev[i])

            if i == n_layers-2:
                u[i] = Y[i].mm(beta_prev[i+1]*output_burst_prob*relu_deriv(beta_prev[i+1])) - c[i]
            else:
                u[i] = Y[i].mm(beta_prev[i+1]*hard_deriv(beta_prev[i+1], mean=hard_m, variance=hard_v)) - c[i]

            # compute burst probability
            p[i]   = torch.sigmoid(u[i])

            # compute burst rate
            beta[i]   = p[i]*h[i]

            # compute costs
            cost[i]   = 0.5*torch.mean((beta[i] - beta_prev[i])**2)
            cost_Z[i] = 0.5*torch.sum((desired_u - u[i])**2)

            # compute recurrent weight update
            delta_Z[i] = (desired_u - u[i]).mm(h_prev[i].transpose(0, 1))
            
            # compute feedforward weight update
            e          = -(beta[i] - beta_prev[i])*hard_deriv(beta_prev[i], mean=hard_m, variance=hard_v)
            delta_W[i] = e.mm(h_prev[i-1].transpose(0, 1))
            delta_b[i] = e

            # compute backprop-prescribed weight update
            delta_b_backprop[i] = W[i+1].transpose(0, 1).mm(delta_b_backprop[i+1])*h_prev[i]*(1.0 - h_prev[i])

def update_weights(W, b, Y, Z, delta_W, delta_b, delta_Y, delta_Z, layer_num=None):
    if layer_num is None:
        for i in range(1, n_layers):
            # update feedforward weights
            W[i] -= f_etas[i]*delta_W[i]
            b[i] -= f_etas[i]*delta_b[i]

            # update recurrent & feedback weights
            if i < n_layers-1:
                Z[i] -= r_etas[i]*delta_Z[i]
                Y[i] -= b_etas[i]*delta_Y[i]
    else:
        i = layer_num

        # update feedforward weights
        W[i] -= f_etas[i]*delta_W[i]
        b[i] -= f_etas[i]*delta_b[i]

        # update recurrent & feedback weights
        if i < n_layers-1:
            Z[i] -= r_etas[i]*delta_Z[i]
            Y[i] -= b_etas[i]*delta_Y[i]

def train(folder_prefix=None, continuing_folder=None):
    if use_tensorboard:
        from tensorboardX import SummaryWriter

    if folder_prefix is not None:
        # generate a name for the folder where data will be stored
        n_units_string = " ".join([ str(i) for i in n_units[1:] ])
        f_etas_string  = " ".join([ str(i) for i in f_etas[1:] ])
        b_etas_string  = " ".join([ str(i) for i in b_etas[1:] ])
        r_etas_string  = " ".join([ str(i) for i in r_etas[1:] ])
        W_range_string = " ".join([ str(i) for i in W_range[1:] ])
        Z_range_string = " ".join([ str(i) for i in Z_range[1:] ])
        Y_range_string = " ".join([ str(i) for i in Y_range[1:] ])
        folder         = "Tensorboard/" + "{} - {} - {} - {} - {} - {} - {} - {} - {} - {} - {} - {}".format(folder_prefix, n_units_string, f_etas_string, b_etas_string, r_etas_string, W_range_string, Z_range_string, Y_range_string, output_burst_prob, desired_u, hard_m, hard_v) + " - {}".format(info)*(info != "")
    else:
        folder = None

    if folder is not None and folder == continuing_folder:
        print("Error: If you're continuing a simulation, the new results need to be saved in a different directory.")
        raise

    if folder is not None and folder == continuing_folder:
        print("Error: If you're continuing a simulation, the new results need to be saved in a different directory.")
        raise

    if folder is not None:
        # make the folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        # save a human-readable text file containing experiment details
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
            f.write("W initialization range: {}\n".format(W_range))
            f.write("Y initialization range: {}\n".format(Y_range))
            f.write("Z initialization range: {}\n".format(Z_range))
            f.write("Output layer burst probability: {}\n".format(output_burst_prob))
            f.write("Desired apical potential: {}\n".format(desired_u))
            f.write("Hard derivative mean: {}\n".format(hard_m))
            f.write("Hard derivative variance: {}\n".format(hard_v))
            f.write("Other info: {}\n".format(info))
        filename = os.path.basename(__file__)
        if filename.endswith('pyc'):
            filename = filename[:-1]
            shutil.copyfile(os.path.abspath(__file__)[:-1], os.path.join(folder, filename))
        shutil.copyfile(os.path.abspath(__file__), os.path.join(folder, filename))

        # save a JSON file containing experiment details
        params_dict = {'timestamp'        : timestamp,
                       'continuing_folder': continuing_folder,
                       'n_epochs'         : n_epochs,
                       'n_examples'       : n_examples,
                       'n_test_examples'  : n_test_examples,
                       'validation'       : validation,
                       'f_etas'           : f_etas,
                       'b_etas'           : b_etas,
                       'r_etas'           : r_etas,
                       'n_units'          : n_units,
                       'W_range'          : W_range,
                       'Y_range'          : Y_range,
                       'Z_range'          : Z_range,
                       'output_burst_prob': output_burst_prob,
                       'desired_u'        : desired_u,
                       'hard_m'           : hard_m,
                       'hard_v'           : hard_v,
                       'info'             : info}
        with open(os.path.join(folder, "params.json"), 'w') as f:
            json.dump(params_dict, f)

    # initialize dynamic variables
    W, b, Y, Z, v, h, h_prev, u, p, beta, beta_prev, mean_c, c = create_dynamic_variables(symmetric_weights=False)
    if continuing_folder is not None:
        W, b, Y, Z, mean_c = load_dynamic_variables(continuing_folder)

    # initialize recording variables
    global cost, cost_Y, cost_Z, delta_W, delta_b, delta_Y, delta_Z, max_u, delta_b_backprop
    cost                = [0] + [ 0 for i in range(1, n_layers) ]
    cost_Y              = [0] + [ 0 for i in range(1, n_layers-1) ]
    cost_Z              = [0] + [ 0 for i in range(1, n_layers-1) ]
    delta_W             = [0] + [ 0 for i in range(1, n_layers) ]
    delta_b             = [0] + [ 0 for i in range(1, n_layers) ]
    delta_Y             = [0] + [ 0 for i in range(1, n_layers-1) ]
    delta_Z             = [0] + [ 0 for i in range(1, n_layers-1) ]
    max_u               = [0] + [ 0 for i in range(1, n_layers-1) ]
    delta_b_backprop    = [0] + [ 0 for i in range(1, n_layers) ]
    avg_costs           = np.zeros(n_layers-1)
    avg_Y_costs         = np.zeros(n_layers-2)
    avg_Z_costs         = np.zeros(n_layers-2)
    test_costs          = 0
    avg_backprop_angles = np.zeros(n_layers-2)
    min_us              = np.zeros(n_layers-2)
    max_us              = np.zeros(n_layers-2)
    min_hs              = np.zeros(n_layers-1)
    max_hs              = np.zeros(n_layers-1)
    avg_cs              = np.zeros(n_layers-2)
    avg_std_cs          = np.zeros(n_layers-2)
    errors              = 0
    train_error         = 0
    avg_W_range         = np.zeros(n_layers-1)
    avg_W_mean          = np.zeros(n_layers-1)
    us                  = np.zeros(500)

    # make a list of training example indices
    example_indices = np.arange(n_examples)

    # calculate the initial test error as a percentage
    errors, test_costs = test(W, b)
    print("Initial test error: {}%.".format(errors))

    if use_tensorboard:
        # initialize a Tensorboard writer
        writer = SummaryWriter(log_dir=folder)

    if folder is not None:
        # save initial variables & test error
        save_dynamic_variables(folder, W, b, Y, Z, mean_c)
        save_results(folder, avg_costs, avg_backprop_angles, [errors], [test_costs], avg_Y_costs, avg_Z_costs, us)

    h_prev[0] = torch.from_numpy(np.zeros((x_set.shape[0], 1))).type(dtype)

    for epoch_num in range(n_epochs):
        start_time = time.time()

        # shuffle which examples to show
        np.random.shuffle(example_indices)

        for example_num in range(n_examples):
            example_index = example_indices[example_num]
            abs_ex_num    = epoch_num*n_examples + example_num

            # get input and target for this example
            x = x_set[:, example_index]
            t = t_set[:, example_index]

            no_target_length   = n_layers   # number of timesteps where no target can be presented (allows a full forward pass to occur)
            target_on_timestep = np.random.choice(np.arange(no_target_length, timesteps_per_example))

            for timestep in range(timesteps_per_example):
                # do a forward pass
                forward(W, b, v, h, h_prev, f_input=x)

                if timestep == target_on_timestep:
                    # get the predicted & target class
                    predicted_class = int(torch.max(h_prev[-1], 0)[1])
                    target_class    = int(torch.max(t, 0)[1])

                    # update the train error
                    if predicted_class != target_class:
                        train_error += 1

                if timestep == target_on_timestep:
                    target = t
                else:
                    target = None

                # do a backward pass
                backward(Y, Z, W, b, u, p, beta, beta_prev, v, h, h_prev, mean_c, c, t_input=target)

                h_prev    = [ h[i].clone() for i in range(len(h)) ]
                beta_prev = [0] + [ beta[i].clone() for i in range(1, len(beta)) ]

                # record variables
                avg_costs           += cost[1:]
                avg_Y_costs         += cost_Y[1:]
                avg_Z_costs         += cost_Z[1:]
                avg_backprop_angles += [(180/np.pi)*np.arccos(np.clip(delta_b_backprop[i].squeeze().dot(delta_b[i].squeeze())/(1e-10 + torch.norm(delta_b_backprop[i])*torch.norm(delta_b[i])),-1,1)) for i in range(1, n_layers-1)]
                min_us              += [ torch.min(u[i]) for i in range(1, n_layers-1) ]
                max_us              += [ torch.max(u[i]) for i in range(1, n_layers-1) ]
                min_hs              += [ torch.min(h[i]) for i in range(1, n_layers) ]
                max_hs              += [ torch.max(h[i]) for i in range(1, n_layers) ]
                avg_cs              += [ torch.mean(c[i]) for i in range(1, n_layers-1) ]
                avg_std_cs          += [ torch.std(c[i]) for i in range(1, n_layers-1) ]
                avg_W_range         += [ torch.std(W[i]) for i in range(1, n_layers) ]
                avg_W_mean          += [ torch.mean(W[i]) for i in range(1, n_layers) ]

                if example_num == 0 and epoch_num == 0 and folder is not None:
                    # save variables
                    us = u[-2]
                    save_results(folder, avg_costs, avg_backprop_angles, [errors], [test_costs], avg_Y_costs, avg_Z_costs, us.cpu().numpy())

                if target_on_timestep <= timestep < target_on_timestep + (n_layers - 1):
                    layer_num = max(0, n_layers - (timestep - target_on_timestep) - 1)
                    
                    # print("Example {}, t = {}. Updating weights for layer {}.".format(example_num, timestep, layer_num))
                    
                    # update weights
                    update_weights(W, b, Y, Z, delta_W, delta_b, delta_Y, delta_Z, layer_num=layer_num)

            if (example_num+1) % store == 0:
                # get test error
                errors, test_costs = test(W, b)

                avg_costs           = [avg_costs[i]/(timesteps_per_example*store) for i in range(n_layers-1)]
                avg_Y_costs         = [avg_Y_costs[i]/(timesteps_per_example*store) for i in range(n_layers-2)]
                avg_Z_costs         = [avg_Z_costs[i]/(timesteps_per_example*store) for i in range(n_layers-2)]
                avg_backprop_angles = [avg_backprop_angles[i]/(timesteps_per_example*store) for i in range(n_layers-2)]
                min_us              = [min_us[i]/(timesteps_per_example*store) for i in range(n_layers-2)]
                max_us              = [max_us[i]/(timesteps_per_example*store) for i in range(n_layers-2)]
                min_hs              = [min_hs[i]/(timesteps_per_example*store) for i in range(n_layers-1)]
                max_hs              = [max_hs[i]/(timesteps_per_example*store) for i in range(n_layers-1)]
                avg_cs              = [avg_cs[i]/(timesteps_per_example*store) for i in range(n_layers-2)]
                avg_std_cs          = [avg_std_cs[i]/(timesteps_per_example*store) for i in range(n_layers-2)]
                std_cs              = [torch.std(c[i]) for i in range(1, n_layers-1)]
                avg_W_range         = [avg_W_range[i]/(timesteps_per_example*store) for i in range(n_layers-1)]
                avg_W_mean          = [avg_W_mean[i]/(timesteps_per_example*store) for i in range(n_layers-1)]
                us                  = u[-2]
                
                step = (abs_ex_num+1)//store

                if use_tensorboard:
                    # write to Tensorboard
                    writer.add_scalar('1_errors', errors, step)
                    writer.add_scalar('2_train_error', train_error, step)
                    writer.add_scalar('3_test_costs', test_costs, step)
                    writer.add_scalar('4_avg_Y_costs', avg_Y_costs[-1], step)
                    writer.add_scalar('5_avg_Z_costs', avg_Z_costs[-1], step)
                    writer.add_scalar('6_avg_costs_o', avg_costs[-1], step)
                    writer.add_scalar('7_avg_costs_h', avg_costs[-2], step)
                    writer.add_scalar('8_avg_backprop_angles', avg_backprop_angles[-1], step)
                    writer.add_scalar('9_avg_W_range_o', avg_W_range[-1], step)
                    writer.add_scalar('10_avg_W_range_h', avg_W_range[-2], step)
                    writer.add_scalar('11_avg_W_mean_o', avg_W_mean[-1], step)
                    writer.add_scalar('12_avg_W_mean_h', avg_W_mean[-2], step)
                    writer.add_scalar('us_min_h', min_us[-1], step)
                    writer.add_scalar('us_max_h', max_us[-1], step)
                    writer.add_scalar('hs_min_o', min_hs[-1], step)
                    writer.add_scalar('hs_max_o', max_hs[-1], step)
                    writer.add_scalar('hs_min_h', min_hs[-2], step)
                    writer.add_scalar('hs_max_h', max_hs[-2], step)
                    writer.add_scalar('cs_mean_h', avg_cs[-1], step)
                    writer.add_scalar('cs_mean_std_h', avg_std_cs[-1], step)
                    writer.add_scalar('cs_std_h', std_cs[-1], step)
                    writer.add_scalar('us_h_0', us[0], step)
                    writer.add_scalar('us_h_1', us[1], step)

                # print test error
                print("Epoch {}, ex {}. Test Error: {}%. Test Cost: {}. Train Cost: {}.".format(epoch_num+1, example_num+1, errors, test_costs, avg_costs[-1]))

                # print some other variables
                for i in range(n_layers-1):
                    if i < n_layers-2:
                        print("Layer {}. BPA: {:.1f}. u: {:.4f} to {:.4f}. h: {:.4f} to {:.4f}. Y_loss_mean: {:.4f}. Z_loss_mean: {:.4f}.".format(i, avg_backprop_angles[i], min_us[i], max_us[i], min_hs[i], max_hs[i],avg_Y_costs[i],avg_Z_costs[i]))
                    else:
                        print("Layer {}. h: {:.4f} to {:.4f}".format(i, min_hs[i], max_hs[i]))

                if folder is not None:
                    # save network state & recording arrays
                    save_dynamic_variables(folder, W, b, Y, Z, mean_c)
                    save_results(folder, avg_costs, avg_backprop_angles, [errors], [test_costs], avg_Y_costs, avg_Z_costs, us.cpu().numpy())

                # reset recording arrays
                avg_costs           = np.zeros(n_layers-1)
                avg_Y_costs         = np.zeros(n_layers-2)
                avg_Z_costs         = np.zeros(n_layers-2)
                test_costs          = 0
                avg_backprop_angles = np.zeros(n_layers-2)
                min_us              = np.zeros(n_layers-2)
                max_us              = np.zeros(n_layers-2)
                min_hs              = np.zeros(n_layers-1)
                max_hs              = np.zeros(n_layers-1)
                errors              = 0
                train_error         = 0
                avg_cs              = np.zeros(n_layers-2)
                avg_std_cs          = np.zeros(n_layers-2)
                std_cs              = np.zeros(n_layers-2)
                avg_W_range         = np.zeros(n_layers-1)
                avg_W_mean          = np.zeros(n_layers-1)

        end_time = time.time()
        print("Elapsed time: {} s.".format(end_time - start_time))

    if use_tensorboard:
        # close Tensorboard writer
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()

    return avg_costs, avg_backprop_angles, errors

def test(W, b):
    v      = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers) ]
    h      = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers) ]
    
    # initialize error
    error = 0
    cost  = 0
    for i in range(n_test_examples):
        # get input and target for this test example
        x = x_test_set[:, i]
        t = t_test_set[:, i]

        # do a forward pass
        forward_test(W, b, v, h, f_input=x)

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
        
def create_training_data():
    # load MNIST data
    x_set, t_set, x_test_set, t_test_set = utils.load_mnist_data(n_examples, n_test_examples, validation=validation)
    x_set      = torch.from_numpy(x_set).type(dtype)
    t_set      = torch.from_numpy(t_set).type(dtype)
    x_test_set = torch.from_numpy(x_test_set).type(dtype)
    t_test_set = torch.from_numpy(t_test_set).type(dtype)

def create_dynamic_variables(symmetric_weights=False):
    # create network variables
    W         = [0] + [ torch.from_numpy(np.random.uniform(-W_range[i], W_range[i], size=(n_units[i], n_units[i-1]))).type(dtype) for i in range(1, n_layers) ]
    W[-1]    += 0.001
    b         = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers) ]
    if symmetric_weights:
        Y     = [0] + [ torch.from_numpy(W[i+1].T.copy()).type(dtype) for i in range(1, n_layers-1) ]
    else:
        Y     = [0] + [ torch.from_numpy(np.random.uniform(-Y_range[i], Y_range[i], size=(n_units[i], n_units[i+1]))).type(dtype) for i in range(1, n_layers-1) ]
    Z         = [0] + [ torch.from_numpy(np.random.uniform(0, Z_range[i], size=(n_units[i], n_units[i]))).type(dtype) for i in range(1, n_layers-1) ]
    v         = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers) ]
    h         = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers) ]
    h_prev    = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers) ]
    u         = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers) ]
    p         = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers) ]
    beta      = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers) ]
    beta_prev = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers) ]
    mean_c    = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers-1) ]
    c         = [0] + [ torch.from_numpy(np.zeros((n_units[i], 1))).type(dtype) for i in range(1, n_layers-1) ]

    return W, b, Y, Z, v, h, h_prev, u, p, beta, beta_prev, mean_c, c

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

def save_results(path, avg_costs, avg_backprop_angles, errors, test_costs, avg_Y_costs, avg_Z_costs, us):
    f=open(os.path.join(path, "avg_costs.csv"),'a')
    np.savetxt(f, [avg_costs], delimiter=" ", fmt='%.5f')
    f.close()
    f=open(os.path.join(path, "avg_Y_costs.csv"),'a')
    np.savetxt(f, [avg_Y_costs], delimiter=" ", fmt='%.5f')
    f.close()
    f=open(os.path.join(path,"avg_Z_costs.csv"),'a')
    np.savetxt(f, [avg_Z_costs], delimiter=" ", fmt='%.5f')
    f.close()
    f=open(os.path.join(path, "test_costs.csv"),'a')
    np.savetxt(f, [test_costs], delimiter=" ", fmt='%.5f')
    f.close()
    f=open(os.path.join(path, "avg_backprop_angles.csv"),'a')
    np.savetxt(f, [avg_backprop_angles], delimiter=" ", fmt='%.5f')
    f.close()
    f=open(os.path.join(path, "errors.csv"),'a')
    np.savetxt(f, [errors], delimiter=" ", fmt='%.1f')
    f.close()
    f=open(os.path.join(path, "us.csv"),'a')
    np.savetxt(f, us, delimiter=" ", fmt='%.5f')
    f.close()
