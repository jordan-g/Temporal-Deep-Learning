import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit

import utils
from plotter import Plotter, SigmoidLimitsPlotter

# training variables
folder       = "alt_test"
validation   = True
dynamic_plot = False
n_units      = [784, 500, 10]
n_epochs     = 50
n_trials     = 1

# hyperparameters
W_std             = [0, 0.1, 0.1]
Z_std             = [0, 0.1]
Y_std             = [0, 1.0]
f_etas            = [0, 0.01, 0.01]
r_etas            = [0, 0.01]
b_etas            = [0, 0.01]
relu_baseline     = 0.0001
output_burst_prob = 0.2
min_Z             = 0.1
u_range           = 2

# f_etas = 0
# b_etas = 0
# r_etas = 0

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

def gradient_check():
    # create network variables
    W      = [0] + [ np.random.uniform(0, W_std[i], size=(n_units[i], n_units[i-1])) for i in range(1, n_layers) ]
    b      = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
    Y      = [0] + [ np.random.normal(0, Y_std[i], size=(n_units[i], n_units[i+1])) for i in range(1, n_layers-1) ]
    Y      = [0] + [ W[i+1].T.copy() for i in range(1, n_layers-1) ]
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

    # get input and target
    x = x_set[:, 0]
    t = t_set[:, 0]

    # get the calculated delta values
    forward(W, b, v, h, f_input=x)
    cost, cost_Y, cost_Z, delta_W, delta_b, delta_Y, delta_Z, max_u, delta_b_backprop = backward(Y, Z, u, u_t, p, p_t, beta, beta_t, v, h, mean_c, t_input=t)

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
            h[i] = softplus(v[i])

def backward(Y, Z, u, u_t, p, p_t, beta, beta_t, v, h, mean_c, t_input):
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

            e          = -(beta_t[i] - beta[i])*output_burst_prob
            delta_W[i] = np.outer(e, h[i-1])
            delta_b[i] = e

            delta_b_backprop[i] = -(beta_t[i] - beta[i])*output_burst_prob*softplus_deriv(v[i])
        else:
            c = np.dot(Z[i], h[i])

            mean_c[i] = 0.5*mean_c[i] + 0.5*c

            if i == n_layers-2:
                u[i]   = np.dot(Y[i], beta[i+1]*output_burst_prob*((beta[i+1] > output_burst_prob*relu_baseline).astype(int)))/c
                u_t[i] = np.dot(Y[i], beta_t[i+1]*output_burst_prob*((beta_t[i+1] > output_burst_prob*relu_baseline).astype(int)))/c
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
            delta_W[i] = np.outer(e, h[i-1]) + 0.001*W[i]
            delta_b[i] = e

            delta_b_backprop[i] = np.dot(W[i+1].T, delta_b_backprop[i+1])*softplus_deriv(v[i])

            e_Y = -(u_range - max_u[i])/mean_c[i]
            delta_Y[i] = (np.sign(Y[i]).T * e_Y).T

            delta_Z[i] = np.outer((-u[i])*(u[i]/c), h[i]) - (min_Z - Z[i])

    return cost, cost_Y, cost_Z, delta_W, delta_b, delta_Y, delta_Z, max_u, delta_b_backprop

def update_weights(W, b, Y, Z, delta_W, delta_b, delta_Y, delta_Z):
    for i in range(1, n_layers):
        W[i] -= f_etas[i]*delta_W[i]
        b[i] -= f_etas[i]*delta_b[i]

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

if __name__ == "__main__":
    if dynamic_plot:
        # cost_plotter           = Plotter(title="Cost Function")
        # mean_W_plotter         = Plotter(title="Mean W")
        mean_delta_W_plotter   = Plotter(title="Mean Delta W")
        # mean_Y_plotter         = Plotter(title="Mean Y")
        # mean_Z_plotter         = Plotter(title="Mean Z")
        # max_r_plotter          = Plotter(title="Max event rate")
        # max_u_plotter          = Plotter(title="Maximum u")
        # sigmoid_limits_plotter = SigmoidLimitsPlotter(title="Sigmoid Limits")

    costs           = np.zeros((n_trials, n_layers, n_epochs*n_examples))
    backprop_angles = np.zeros((n_trials, n_layers-2, n_epochs*n_examples))
    min_us          = np.zeros((n_trials, n_layers-2, n_epochs*n_examples))
    max_us          = np.zeros((n_trials, n_layers-2, n_epochs*n_examples))
    min_hs          = np.zeros((n_trials, n_layers-1, n_epochs*n_examples))
    max_hs          = np.zeros((n_trials, n_layers-1, n_epochs*n_examples))
    errors          = np.zeros((n_trials, n_epochs+1))

    for trial_num in range(n_trials):
        print("Trial {:>2d}/{:>2d}. --------------------".format(trial_num+1, n_trials))

        # create network variables
        W      = [0] + [ np.random.normal(0, W_std[i], size=(n_units[i], n_units[i-1])) for i in range(1, n_layers) ]
        b      = [0] + [ np.zeros(n_units[i]) for i in range(1, n_layers) ]
        Y      = [0] + [ np.random.normal(0, Y_std[i], size=(n_units[i], n_units[i+1])) for i in range(1, n_layers-1) ]
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

        # make a list of training example indices
        example_indices = np.arange(n_examples)

        # calculate the initial test error as a percentage
        errors[trial_num, 0] = test(W, b)
        print("Initial test error: {}%.".format(errors[trial_num, 0]))

        for epoch_num in range(n_epochs):
            # shuffle which examples to show
            np.random.shuffle(example_indices)

            for example_num in range(n_examples):
                example_index = example_indices[example_num]

                # get input and target for this example
                x = x_set[:, example_index]
                t = t_set[:, example_index]

                # print(np.amax(x))

                forward(W, b, v, h, f_input=x)
                cost, cost_Y, cost_Z, delta_W, delta_b, delta_Y, delta_Z, max_u, delta_b_backprop = backward(Y, Z, u, u_t, p, p_t, beta, beta_t, v, h, mean_c, t_input=t)
                costs[trial_num, :, epoch_num*n_examples + example_num] = cost

                backprop_angle = np.array([ (180/np.pi)*np.arccos(np.dot(delta_b_backprop[i], delta_b[i])/(np.linalg.norm(delta_b_backprop[i])*np.linalg.norm(delta_b[i]))) for i in range(1, n_layers-1) ])
                backprop_angles[trial_num, :, epoch_num*n_examples + example_num] = backprop_angle
                min_us[trial_num, :, epoch_num*n_examples + example_num] = np.array([ np.amin(u[i]) for i in range(1, n_layers-1) ])
                max_us[trial_num, :, epoch_num*n_examples + example_num] = np.array([ np.amax(u[i]) for i in range(1, n_layers-1) ])
                min_hs[trial_num, :, epoch_num*n_examples + example_num] = np.array([ np.amin(h[i]) for i in range(1, n_layers) ])
                max_hs[trial_num, :, epoch_num*n_examples + example_num] = np.array([ np.amax(h[i]) for i in range(1, n_layers) ])

                update_weights(W, b, Y, Z, delta_W, delta_b, delta_Y, delta_Z)

                if dynamic_plot:
                    # cost_plotter.plot([cost[i] for i in range(1, n_layers)], labels=["Layer {}".format(i) for i in range(1, n_layers)])
                    # mean_W_plotter.plot([np.mean(W[i]) for i in range(1, n_layers)], labels=["Layer {}".format(i) for i in range(1, n_layers)])
                    mean_delta_W_plotter.plot([np.mean(delta_W[i]) for i in range(1, n_layers)], labels=["Layer {}".format(i) for i in range(1, n_layers)])
                    # mean_Y_plotter.plot([np.mean(Y[i]) for i in range(1, n_layers-1)], labels=["Layer {}".format(i) for i in range(1, n_layers-1)])
                    # mean_Z_plotter.plot([np.mean(Z[i]) for i in range(1, n_layers-1)], labels=["Layer {}".format(i) for i in range(1, n_layers-1)])
                    # max_r_plotter.plot([np.amax(h[i]) for i in range(1, n_layers)], labels=["Layer {}".format(i) for i in range(1, n_layers)])
                    # max_u_plotter.plot([np.mean(max_u[i]) for i in range(1, n_layers-1)], labels=["Layer {}".format(i) for i in range(1, n_layers-1)])
                    # sigmoid_limits_plotter.plot([np.amax(max_u[i]) for i in range(1, n_layers-1)], [-np.amax(max_u[i]) for i in range(1, n_layers-1)], [max(np.amax(u[i]), np.amax(u_t[i])) for i in range(1, n_layers-1)], [min(np.amin(u[i]), np.amin(u_t[i])) for i in range(1, n_layers-1)], labels=["Layer {}".format(i) for i in range(1, n_layers-1)])

                if (example_num+1) % 1000 == 0:
                    # print("Example {}.".format(example_num+1))
                    error = test(W, b)

                    # print test error
                    print("Epoch {}, ex {}. TE: {}%.".format(epoch_num+1, example_num+1, error))

                    abs_ex_num = epoch_num*n_examples + example_num+1

                    for i in range(1, n_layers-1):
                        print("Layer {}. BPA: {:.1f}. u: {:.4f} to {:.4f}. h: {:.4f} to {:.4f}".format(i, np.mean(backprop_angles[trial_num, i-1, abs_ex_num-1000:abs_ex_num]), np.mean(min_us[trial_num, i-1, abs_ex_num-1000:abs_ex_num]), np.mean(max_us[trial_num, i-1, abs_ex_num-1000:abs_ex_num]), np.mean(min_hs[trial_num, i-1, abs_ex_num-1000:abs_ex_num]), np.mean(max_hs[trial_num, i-1, abs_ex_num-1000:abs_ex_num])))
                    print("Layer {}. h: {:.4f} to {:.4f}".format(n_layers-1, np.mean(min_hs[trial_num, -1, abs_ex_num-1000:abs_ex_num]), np.mean(max_hs[trial_num, -1, abs_ex_num-1000:abs_ex_num])))


            errors[trial_num, epoch_num+1] = test(W, b)
            print("Epoch {} test error: {}.".format(epoch_num+1, errors[trial_num, epoch_num+1]))

            plt.figure()
            for i in range(1, n_layers-1):
                plt.plot(backprop_angles[trial_num, i-1, :(epoch_num+1)*n_examples])
            plt.savefig("backprop_angles.png")

