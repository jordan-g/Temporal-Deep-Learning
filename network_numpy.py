import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

import utils
from plotter import Plotter, SigmoidLimitsPlotter
import os
import datetime
import pickle

# training variables
folder       = "alt_test"
validation   = True
dynamic_plot = False
n_epochs     = 50

# hyperparameters
n_units           = [784, 500, 300, 10]
W_std             = [0, 0.01, 0.01, 0.1]
Z_std             = [0, 0.1, 0.1]
Y_std             = [0, 1.0, 1.0]
f_etas            = [0, 0.01, 0.01, 0.01]
b_etas            = [0, 0.01, 0.01]
r_etas            = [0, 0.01, 0.01]
output_burst_prob = 0.2
min_Z             = 0.01
u_range           = 2
W_decay           = 0
# relu_baseline     = 0.0001

# number of layers
n_layers = len(n_units)

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

def create_training_data():
    # load MNIST data
    x_set, t_set, x_test_set, t_test_set = utils.load_mnist_data(n_examples, n_test_examples, validation=validation)

def create_dynamic_variables(symmetric_weights=False):
    # create network variables
    W      = [0] + [ np.random.normal(0, W_std[i], size=(n_units[i], n_units[i-1])) for i in range(1, n_layers) ]
    b      = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
    if symmetric_weights:
        Y  = [0] + [ W[i+1].T.copy() for i in range(1, n_layers-1) ]
    else:
        Y  = [0] + [ np.random.normal(0, Y_std[i], size=(n_units[i], n_units[i+1])) for i in range(1, n_layers-1) ]
    Z      = [0] + [ np.random.uniform(0, Z_std[i], size=(n_units[i], n_units[i])) for i in range(1, n_layers-1) ]
    v      = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
    h      = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
    u      = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
    u_t    = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
    p      = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
    p_t    = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
    beta   = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
    beta_t = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
    mean_c = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers-1) ]

    return W, b, Y, Z, v, h, u, u_t, p, p_t, beta, beta_t, mean_c

def load_dynamic_variables(path):
    W      = pickle.load(open(os.path.join(path, "W.pkl"), 'rb'))
    b      = pickle.load(open(os.path.join(path, "b.pkl"), 'rb'))
    Y      = pickle.load(open(os.path.join(path, "Y.pkl"), 'rb'))
    Z      = pickle.load(open(os.path.join(path, "Z.pkl"), 'rb'))
    v      = pickle.load(open(os.path.join(path, "v.pkl"), 'rb'))
    h      = pickle.load(open(os.path.join(path, "h.pkl"), 'rb'))
    u      = pickle.load(open(os.path.join(path, "u.pkl"), 'rb'))
    u_t    = pickle.load(open(os.path.join(path, "u_t.pkl"), 'rb'))
    p      = pickle.load(open(os.path.join(path, "p.pkl"), 'rb'))
    p_t    = pickle.load(open(os.path.join(path, "p_t.pkl"), 'rb'))
    beta   = pickle.load(open(os.path.join(path, "beta.pkl"), 'rb'))
    beta_t = pickle.load(open(os.path.join(path, "beta_t.pkl"), 'rb'))
    mean_c = pickle.load(open(os.path.join(path, "mean_c.pkl"), 'rb'))

    return W, b, Y, Z, v, h, u, u_t, p, p_t, beta, beta_t, mean_c

def save_dynamic_variables(path, W, b, Y, Z, v, h, u, u_t, p, p_t, beta, beta_t, mean_c):
    pickle.dump(W, open(os.path.join(path, "W.pkl"), 'wb'))
    pickle.dump(b, open(os.path.join(path, "b.pkl"), 'wb'))
    pickle.dump(Y, open(os.path.join(path, "Y.pkl"), 'wb'))
    pickle.dump(Z, open(os.path.join(path, "Z.pkl"), 'wb'))
    pickle.dump(v, open(os.path.join(path, "v.pkl"), 'wb'))
    pickle.dump(h, open(os.path.join(path, "h.pkl"), 'wb'))
    pickle.dump(u, open(os.path.join(path, "u.pkl"), 'wb'))
    pickle.dump(u_t, open(os.path.join(path, "u_t.pkl"), 'wb'))
    pickle.dump(p, open(os.path.join(path, "p.pkl"), 'wb'))
    pickle.dump(p_t, open(os.path.join(path, "p_t.pkl"), 'wb'))
    pickle.dump(beta, open(os.path.join(path, "beta.pkl"), 'wb'))
    pickle.dump(beta_t, open(os.path.join(path, "beta_t.pkl"), 'wb'))
    pickle.dump(mean_c, open(os.path.join(path, "mean_c.pkl"), 'wb'))

def save_results(path, costs, backprop_angles, errors):
    pickle.dump(costs, open(os.path.join(path, "costs.pkl"), 'wb'))
    pickle.dump(backprop_angles, open(os.path.join(path, "backprop_angles.pkl"), 'wb'))
    pickle.dump(errors, open(os.path.join(path, "errors.pkl"), 'wb'))

def gradient_check():
    W, b, Y, Z, v, h, u, u_t, p, p_t, beta, beta_t, mean_c = create_dynamic_variables(symmetric_weights=True)

    # get input and target
    x = x_set[:, 0]
    t = t_set[:, 0]

    # get the calculated delta values
    forward(W, b, v, h, f_input=x)
    cost, cost_Y, cost_Z, delta_W, delta_b, delta_Y, delta_Z, max_u, delta_b_backprop = backward(Y, Z, W, b, u, u_t, p, p_t, beta, beta_t, v, h, mean_c, t_input=t)

    # decrease and increase weights slightly, compare with delta values
    numerical_delta_W = [0] + [ np.zeros(W[i].shape) for i in range(1, n_layers) ]
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
                cost_minus = 0.5*np.sum((beta_t - beta)**2)

                W[i][j, k] += 2*epsilon

                if i > 1:
                    Y[i-1][k, j] += 2*epsilon

                forward(W, b, v, h, f_input=x)
                beta   = output_burst_prob*h[-1]
                beta_t = output_burst_prob*t
                cost_plus = 0.5*np.sum((beta_t - beta)**2)

                numerical_delta_W[i][j, k] = (cost_plus - cost_minus)/(2*epsilon)

                W[i][j, k] -= epsilon

                if i > 1:
                    Y[i-1][k, j] -= epsilon

    print([ np.mean(np.abs((delta_W[i] - numerical_delta_W[i]))) for i in range(1, n_layers) ])

def test(W, b):
    v = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
    h = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]

    # initialize error
    error = 0

    for i in range(n_test_examples):
        print("Test example {}".format(i))
        # get input and target for this test example
        x = x_test_set[:, i]
        t = t_test_set[:, i]

        # do a forward pass
        forward(W, b, v, h, f_input=x)

        # get the predicted & target class
        predicted_class = np.argmax(h[-1])
        target_class    = np.argmax(t)

        # update the test error
        if predicted_class != target_class:
            error += 1

    return 100.0*error/n_test_examples

def forward(W, b, v, h, f_input):
    h[0] = f_input

    for i in range(1, n_layers):
        v[i] = np.dot(W[i], h[i-1]) + b[i]

        if i < n_layers-1:
            h[i] = softplus(v[i])
        else:
            h[i] = relu(v[i]) + np.random.uniform(0, 0.001, size=h[i].shape)

def backward(Y, Z, W, b, u, u_t, p, p_t, beta, beta_t, v, h, mean_c, t_input):
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
            beta_t[i] = output_burst_prob*t_input

            cost[i] = 0.5*np.sum((beta_t[i] - beta[i])**2)

            e          = -(beta_t[i] - beta[i])*output_burst_prob*output_burst_prob*relu_deriv(v[i])
            delta_W[i] = np.outer(e, h[i-1])
            delta_b[i] = e

            delta_b_backprop[i] = -(beta_t[i] - beta[i])*output_burst_prob*relu_deriv(v[i])
        else:
            c = np.dot(Z[i], h[i])

            mean_c[i] = 0.5*mean_c[i] + 0.5*c

            if i == n_layers-2:
                u[i]   = np.dot(Y[i], beta[i+1]*output_burst_prob*relu_deriv(beta[i+1]))/c
                u_t[i] = np.dot(Y[i], beta_t[i+1]*output_burst_prob*relu_deriv(beta[i+1]))/c
            else:
                u[i]   = np.dot(Y[i], beta[i+1])/c
                u_t[i] = np.dot(Y[i], beta_t[i+1])/c

            max_u[i] = np.sum(np.abs(Y[i]), axis=1)/mean_c[i]

            p[i]   = expit(u[i])
            p_t[i] = expit(u_t[i])

            beta[i]   = p[i]*h[i]
            beta_t[i] = p_t[i]*h[i]

            cost[i]   = 0.5*np.sum((beta_t[i] - beta[i])**2) + 0.5*np.sum((W[i])**2)
            cost_Z[i] = 0.5*np.sum((u[i])**2) + 0.5*np.sum((min_Z - Z[i])**2)
            cost_Y[i] = 0.5*np.sum((u_range - max_u[i])**2)

            e          = -(beta_t[i] - beta[i])
            delta_W[i] = np.outer(e, h[i-1])
            delta_b[i] = e

            delta_b_backprop[i] = np.dot(W[i+1].T, delta_b_backprop[i+1])*softplus_deriv(v[i])

            e_Y = -(u_range - max_u[i])/mean_c[i]
            delta_Y[i] = (np.sign(Y[i]).T * e_Y).T

            delta_Z[i] = np.outer((-u[i])*(u[i]/c), h[i]) - (min_Z - Z[i])

    return cost, cost_Y, cost_Z, delta_W, delta_b, delta_Y, delta_Z, max_u, delta_b_backprop

def train(path=None, continuing_path=None):
    W, b, Y, Z, v, h, u, u_t, p, p_t, beta, beta_t, mean_c = create_dynamic_variables(symmetric_weights=False)

    # # calculate the initial test error as a percentage
    # v = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
    # h = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]

    # # initialize error
    error = 0

    # x = x_test_set[:, 0]
    # t = t_test_set[:, 0]

    for i in range(n_test_examples):
        # print("Test example {}".format(i))
        # get input and target for this test example
        # x = x_test_set[:, i]
        # t = t_test_set[:, i]

        # do a forward pass
        # h[0] = x

        # for i in range(1, n_layers):
            # v[i] = np.dot(W[i], h[i-1]) + b[i]

            # if i < n_layers-1:
            #     h[i] = softplus(v[i])
            # else:
            #     h[i] = relu(v[i]) + np.random.uniform(0, 0.001, size=h[i].shape)

        # get the predicted & target class
        # predicted_class = np.argmax(h[-1])
        # target_class    = np.argmax(t)

        predicted_class = 0
        target_class    = 0

        # update the test error
        if predicted_class != target_class:
            error += 1

    error = 100.0*error/n_test_examples

    # error = test(W, b)
    print("Initial test error: {}%.".format(error))

    return error

def do_nothing():
    print("Doing nothing.")
    return 0, 0, 10

def plot_backprop_angles(backprop_angles, filename=None):
    plt.figure()
    for i in range(1, n_layers-1):
        plt.plot(np.mean(backprop_angles[:, i-1, :], axis=0))

    if filename is not None:
        plt.savefig(filename + ".png")
        plt.savefig(filename + ".svg")
    else:
        plt.show()

def update_weights(W, b, Y, Z, delta_W, delta_b, delta_Y, delta_Z):
    for i in range(1, n_layers):
        W[i] -= f_etas[i]*(delta_W[i] + W_decay*W[i])
        b[i] -= f_etas[i]*(delta_b[i] + W_decay*b[i])

        if i < n_layers-1:
            Y[i] -= b_etas[i]*delta_Y[i]
            Z[i] -= r_etas[i]*delta_Z[i]
            Z[i][Z[i] < 0] = 0

def softplus(x, limit=30):
    y = x.copy()
    y[x <= limit] = np.log(1.0 + np.exp(x[x <= limit]))
    
    return y

def softplus_deriv(x):
    return expit(x)

def relu(x, baseline=0):
    y = x.copy()
    y[x < baseline] = baseline

    return y

def relu_deriv(x, baseline=0):
    return (x > baseline).astype(float)