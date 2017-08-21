import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------
"""                 Simulation parameters                     """
# ---------------------------------------------------------------

sequence_length    = 2000 # length of the input sequence to be repeated
n_spikes_per_burst = 10   # number of spikes in each burst
teach_prob         = 0.05 # probability of a teaching signal being provided

use_sparse_feedback  = False # zero out a proportion of the feedback weights
sparse_feedback_prop = 0.5   # proportion of feedback weights to set to 0

# uniform distribution ranges for initial weights
W_range = 0.1
Y_range = 10

# ---------------------------------------------------------------
"""                     Functions                             """
# ---------------------------------------------------------------

def create_data(n_in, n_out):
    '''
    Generate input & target data using sine & cosine functions.
    '''

    x_set = np.zeros((n_in, sequence_length)).astype(np.float32)
    for i in range(n_in):
        x = np.arange(sequence_length)
        x_set[i] = 0.2 + 0.5*(np.sin(np.random.uniform(0.01, 0.04)*x + np.random.uniform(0, 20)) + 1)*np.random.uniform(0.1, 0.3)
        x_set[i] += 0.5*(np.cos(np.random.uniform(0.01, 0.05)*x + np.random.uniform(0, 20)) + 1)*np.random.uniform(0.1, 0.3)

    t_set = np.zeros((n_out, sequence_length)).astype(np.float32)
    for i in range(n_out):
        x = np.arange(sequence_length)
        t_set[i] = 0.2 + 0.5*(np.sin(np.random.uniform(0.01, 0.04)*x + np.random.uniform(0, 20)) + 1)*np.random.uniform(0.1, 0.3)
        t_set[i] += 0.5*(np.cos(np.random.uniform(0.01, 0.05)*x + np.random.uniform(0, 20)) + 1)*np.random.uniform(0.1, 0.3)

    np.save("x_set.npy", x_set)
    np.save("t_set.npy", t_set)

    return torch.from_numpy(x_set), torch.from_numpy(t_set)

def load_data():
    x_set = np.load("x_set.npy")
    t_set = np.load("t_set.npy")

    return torch.from_numpy(x_set), torch.from_numpy(t_set)

def get_x(n):
    return x_set[:, n].unsqueeze_(1)

def get_t(n):
    return t_set[:, n].unsqueeze_(1)

# ---------------------------------------------------------------
"""                     Network class                         """
# ---------------------------------------------------------------

class Network:
    def __init__(self, n):
        '''
        Initialize the network.

        Arguments:
            n (tuple) : Number of units in each layer of the network, eg. (500, 100, 10),
                        including the input layer.
        '''

        self.n = n               # layer sizes - eg. (500, 100, 10)
        self.M = len(self.n) - 1 # number of layers, not including the input layer

        # create input & target data
        try:
            self.x_set, self.t_set = load_data()
        except:
            self.x_set, self.t_set = create_data(self.n[0], self.n[-1])

        print("Creating a network with {} layers.".format(self.M))
        print("---------------------------------")

        self.init_layers()

    def init_layers(self):
        '''
        Create the layers of the network.
        '''

        # initialize layers list
        self.l = []

        # create all layers
        if self.M == 1:
            self.l.append(finalLayer(net=self, m=0, f_input_size=self.n[0]))
        else:
            for m in range(1, self.M):
                self.l.append(hiddenLayer(net=self, m=m-1, f_input_size=self.n[m-1], b_input_size=self.n[-1]))
            self.l.append(finalLayer(net=self, m=self.M-1, f_input_size=self.n[-2]))

    def out(self, x, t, prev_t, time, generate_activity=False, update_b_weights=False, update_f_weights=False):
        '''
        Simulate the network's activity over one timestep.

        Arguments:
            x (tensor)                : The activity of the input layer for this time step.
            t (tensor/None)           : The target activity for the output layer for this time step.
            prev_t (tensor/None)      : The target activity for the output layer for the previous time step.
            time (int)                : The current time step.
            generate_activity (bool) : Whether to use feedback from the output layer to generate hidden layer activity (generative mode).
            update_b_weights (bool)   : Whether to update feedback weights.
            update_f_weights (bool)   : Whether to update feedforward weights.
        '''

        if self.M == 1:
            if time >= 1:
                self.l[0].update_activity(x, t)

            if t is not None and update_f_weights:
                self.l[0].burst(self.f_etas[0])
        else:
            if time >= self.M:
                self.l[-1].update_activity(self.l[-2].event_rate, t)

            for m in range(self.M-2, -1, -1):
                if time >= m+1:
                    if m == 0:
                        self.l[0].update_f_input(x)
                    else:
                        self.l[m].update_f_input(self.l[m-1].event_rate)

            for m in range(self.M-1):
                if time > self.M+1:
                    self.l[m].update_b_input(self.l[-1].event_rate, self.b_etas[m], generate_activity=generate_activity, update_b_weights=update_b_weights)

                    if prev_t is not None and update_f_weights:
                        self.l[m].burst(self.f_etas[m])

            if time >= self.M and t is not None and update_f_weights:
                self.l[-1].burst(self.f_etas[-1])

    def train(self, f_etas, b_etas, n_epochs, plot_activity=False, weight_decay=0, update_b_weights=False, generate_activity=False):
        '''
        Train the network.

        Arguments:
            f_etas (list/tuple/int)  : The learning rates for the feedforward weights.
                                       If an int is provided, each layer will have the same learning rate.
            b_etas (list/tuple/int)  : The learning rates for the feedback weights.
                                       If an int is provided, each layer will have the same learning rate.
            n_epochs (int)           : The number of epochs of training.
            plot_activity (bool)     : Whether to create a plot that compares the output & targets for the network.
            weight_decay (int)       : Weight decay constant.
            update_b_weights (bool)  : Whether to update feedback weights.
            generate_activity (bool) : Whether to internally generate activity during the second half of the last epoch.
        '''

        if not update_b_weights:
            b_etas = 0

        if type(f_etas) is int:
            f_etas = [f_etas]*self.M

        if type(b_etas) is int:
            b_etas = [b_etas]*(self.M-1)

        if len(f_etas) != self.M:
            raise Exception("Mismatch between the number of feedforward learning rates provided and the number of layers.")

        if len(b_etas) != self.M-1:
            raise Exception("Mismatch between the number of feedback learning rates provided and the number of hidden layers.")

        print("Starting training.\n")

        self.weight_decay      = weight_decay
        self.update_b_weights  = update_b_weights
        self.generate_activity = generate_activity

        # set learning rate instance variables
        self.f_etas = f_etas
        self.b_etas = b_etas

        if plot_activity:
            # set colors for plotting the output & target output of the network during training
            colors = ["red", "blue", "green", "purple", "brown", "cyan", "orange", "magenta"]

            if len(colors) < self.n[-1]:
                raise Exception("Number of output neurons exceeds the number of defined colors for plotting.")

            # create the figure
            self.figure = plt.figure(figsize=(15, 8), facecolor='white')

            # create top axis
            self.animation_axis_top = plt.Axes(self.figure, [0.07, 0.57, 0.86, 0.36])
            self.figure.add_axes(self.animation_axis_top)
            self.target_lines_top = [ self.animation_axis_top.plot([], [], color=colors[i], lw=1, label='Unit {} Target'.format(i+1))[0] for i in range(self.n[-1]) ]
            self.output_lines_top = [ self.animation_axis_top.plot([], [], color=colors[i], lw=1, linestyle='--', alpha=0.5, label='Unit {} Activity'.format(i+1))[0] for i in range(self.n[-1]) ]
            self.animation_axis_top.set_title("Start of Training")
            self.animation_axis_top.legend()

            # create bottom axis
            self.animation_axis_bottom = plt.Axes(self.figure, [0.07, 0.07, 0.86, 0.36])
            self.figure.add_axes(self.animation_axis_bottom)
            self.target_lines_bottom = [ self.animation_axis_bottom.plot([], [], color=colors[i], lw=1, label='Unit {} Target'.format(i+1))[0] for i in range(self.n[-1]) ]
            self.output_lines_bottom = [ self.animation_axis_bottom.plot([], [], color=colors[i], lw=1, linestyle='--', alpha=0.5, label='Unit {} Activity'.format(i+1))[0] for i in range(self.n[-1]) ]
            self.animation_axis_bottom.set_title("After Training")
            self.animation_axis_bottom.set_xlabel("Timestep")

        # initialize counter for number of time steps at which no target is present
        no_t_count = 0

        # initialize array to hold average loss over each 100 time steps
        # and a counter to keep track of where we are in the avg_losses array
        # note: in the last epoch we don't present a target.
        avg_losses = np.zeros((self.M, int((n_epochs-int(plot_activity))*sequence_length/100.0)))
        counter = 0

        # initialize arrays to hold targets and outputs over time
        self.targets = np.zeros((sequence_length*n_epochs, self.n[-1]))
        self.outputs = np.zeros((sequence_length*n_epochs, self.n[-1]))
        self.target_times = []

        # initialize target
        t = None

        for k in range(n_epochs):
            for time in range(sequence_length):
                # set previous target
                if t is not None:
                    prev_t = t.clone()
                else:
                    prev_t = None

                # get input & target for this time step
                self.x = self.x_set[:, time].unsqueeze_(1)
                self.t = self.t_set[:, time].unsqueeze_(1)

                # targets are not shown in the last epoch
                if (k < n_epochs-int(plot_activity) and np.random.uniform(0, 1) >= 1 - teach_prob):
                    no_t = False
                    t    = self.t
                    self.target_times.append(k*sequence_length + time)
                else:
                    no_t = True
                    t    = None
                    no_t_count += 1

                # don't update feedback weights during the last epoch
                update_b_weights = self.update_b_weights and k < n_epochs-int(plot_activity)

                # don't update feedforward weights during the last epoch
                update_f_weights = k < n_epochs-int(plot_activity)

                # internally generate activity during the second half of the last epoch
                generate_activity = self.generate_activity and k == n_epochs-int(plot_activity) and time > sequence_length/2

                # simulate network activity for this time step
                self.out(self.x, t, prev_t, time, generate_activity=generate_activity, update_b_weights=update_b_weights, update_f_weights=update_f_weights)

                # add the loss to average loss, only if a target was present
                if not no_t:
                    for m in range(self.M):
                        avg_losses[m, counter] += float(self.l[m].loss)

                # record targets & outputs for this time step
                self.targets[k*sequence_length + time] = self.t.numpy()[:, 0]
                self.outputs[k*sequence_length + time] = self.l[-1].event_rate.numpy()[:, 0]

                if (time+1) % 100 == 0 and k < n_epochs-int(plot_activity):
                    # compute average loss over the last 100 time steps
                    # minus those where a target wasn't present
                    if 100 - no_t_count > 0:
                        avg_losses[:, counter] /= (100 - no_t_count)
                        if self.M > 1:
                            print("Epoch {:>3d}, t={:>4d}. Avg. output loss: {:.10f}. Avg. last hidden loss: {:.10f}.".format(k+1, time+1, avg_losses[-1, counter], avg_losses[-2, counter]))
                        else:
                            print("Epoch {:>3d}, t={:>4d}. Avg. output loss: {:.10f}.".format(k+1, time+1, avg_losses[-1, counter]))
                    else:
                        if self.M > 1:
                            print("Epoch {:>3d}, t={:>4d}. Avg. output loss: {}. Avg. last hidden loss: {}.".format(k+1, time+1, "_"*12, "_"*12))
                        else:
                            print("Epoch {:>3d}, t={:>4d}. Avg. output loss: {}.".format(k+1, time+1, "_"*12))
                        
                    no_t_count = 0
                    counter   += 1

        if plot_activity:
            # plot activity vs. target
            for l in range(self.n[-1]):
                x_range = np.arange(sequence_length)
                self.target_lines_top[l].set_data(x_range, self.targets[:sequence_length, l])
                self.output_lines_top[l].set_data(x_range, self.outputs[:sequence_length, l])

                self.target_lines_bottom[l].set_data(x_range, self.targets[(n_epochs - 1)*sequence_length:, l])
                self.output_lines_bottom[l].set_data(x_range, self.outputs[(n_epochs - 1)*sequence_length:, l])

                # add scatter points to show when target was present
                for target_time in self.target_times:
                    if target_time < sequence_length:
                        self.animation_axis_top.scatter(target_time+1, self.outputs[target_time+1, l], c=colors[l], s=10)
                    elif (n_epochs - 1)*sequence_length <= target_time:
                        self.animation_axis_bottom.scatter(target_time+1, self.outputs[target_time+1, l], c=colors[l], s=10)

                self.animation_axis_top.relim()
                self.animation_axis_top.autoscale_view(scalex=True, scaley=True)

                self.animation_axis_bottom.relim()
                self.animation_axis_bottom.autoscale_view(scalex=True, scaley=True)

            plt.draw()
            plt.savefig("outputs.svg")
            plt.savefig("outputs.png")
            plt.pause(100000)

        return avg_losses

    def save_weights(self, path, prefix=""):
        '''
        Save the network's current weights to .npy files.

        Arguments:
            path (string)   : The path of the folder in which to save the network's weights.
            prefix (string) : A prefix to append to the filenames of the saved weights.
        '''

        for m in range(self.M):
            np.save(os.path.join(path, prefix + "W_{}.npy".format(m)), self.l[m].W)
            np.save(os.path.join(path, prefix + "b_{}.npy".format(m)), self.l[m].b)
            if m != self.M-1:
                np.save(os.path.join(path, prefix + "Y_{}.npy".format(m)), self.l[m].Y)

    def load_weights(self, path, prefix=""):
        '''
        Load weights from .npy files and set them to the network's weights.

        Arguments:
            path (string)   : The path of the folder from which to load the weights.
            prefix (string) : Prefix appended to the filenames of the saved weights.
        '''

        print("Loading weights from \"{}\" with prefix \"{}\".".format(path, prefix))
        print("--------------------------------")

        for m in range(self.M):
            self.l[m].W = np.load(os.path.join(path, prefix + "W_{}.npy".format(m)))
            self.l[m].b = np.load(os.path.join(path, prefix + "b_{}.npy".format(m)))
            if m != self.M-1:
                self.l[m].Y = np.load(os.path.join(path, prefix + "Y_{}.npy".format(m)))

        # print network weights
        self.print_weights()

        print("--------------------------------")

    def set_weights(self, W_list, b_list, Y_list):
        '''
        Set the weights of the network.

        Arguments:
            W_list (list) : The feedforward weights of the network.
            b_list (list) : The feedforward biases of the network.
            Y_list (list) : The feedback weights of the network.
        '''

        if len(W_list) != self.M or len(b_list) != self.M or len(Y_list) != self.M-1:
            raise Exception("There is a mismatch between provided weight lists and the number of layers in the network.")
        
        for m in range(self.M):
            self.l[m].W = W_list[m].copy()
            self.l[m].b = b_list[m].copy()

            if m < self.M-1:
                self.l[m].Y = Y_list[m].copy()

# ---------------------------------------------------------------
"""                     Layer classes                         """
# ---------------------------------------------------------------

class Layer:
    def __init__(self, net, m, f_input_size):
        '''
        Create the layer.

        Arguments:
            net (Network)      : The network this layer belongs to.
            m (int)            : The index of this layer (indexing starts at the first hidden layer).
            f_input_size (int) : The size of the feedforward input. This is equivalent to the size
                                 of the previous layer.
        '''

        # create all of the layer variables
        self.net      = net
        self.m        = m
        self.size     = self.net.n[m+1]

        # feedforward input
        self.f_input       = torch.from_numpy(np.zeros((f_input_size, 1)).astype(np.float32))
        self.f_input_prev  = torch.from_numpy(np.zeros((f_input_size, 1)).astype(np.float32))

        # somatic voltage
        self.h  = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        
        # spike rate, burst probability, burst rate, event rate
        self.spike_rate       = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        self.burst_prob       = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        self.burst_prob_prev  = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        self.burst_rate       = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        self.burst_rate_prev  = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        self.event_rate       = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        self.event_rate_prev  = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))

        # loss
        self.loss = 0

        # feedforward weights & biases
        self.W = torch.from_numpy(W_range*np.random.uniform(-1, 1, size=(self.size, f_input_size)).astype(np.float32))
        self.b = torch.from_numpy(0.1*np.ones((self.size, 1)).astype(np.float32))

    def update_f_input(self, f_input):
        # update previous feedforward input
        self.f_input_prev = self.f_input.clone()

        # apply exponential smoothing to feedforward input
        self.f_input = (self.f_input + f_input.clone())/2.0

        # calculate somatic voltage
        self.h = self.W.mm(self.f_input) + self.b

        # update previous event rate
        self.event_rate_prev = self.event_rate.clone()

        # update event rate
        self.event_rate = torch.sigmoid(self.h)

        # update spike rate
        self.spike_rate = (1.0 - self.burst_rate)*self.event_rate + self.burst_rate*self.event_rate*n_spikes_per_burst

    def update_W(self, f_eta, E):
        # update feedforward weights & biases
        self.delta_W = E.mm(self.f_input.t())
        self.W      += -f_eta*self.delta_W

        self.delta_b = E
        self.b      += -f_eta*self.delta_b

    def decay_W(self):
        self.W      -= self.net.weight_decay*self.W

class hiddenLayer(Layer):
    def __init__(self, net, m, f_input_size, b_input_size):
        '''
        Create the hidden layer.

        Arguments:
            net (Network)      : The network this layer belongs to.
            m (int)            : The index of this layer (indexing starts at the first hidden layer).
            f_input_size (int) : The size of the feedforward input. This is equivalent to the size
                                 of the previous layer.
            b_input_size (int) : The size of the feedback input. This is equivalent to the size
                                 of the next layer.
        '''

        Layer.__init__(self, net, m, f_input_size)

        # feedback input
        self.b_input      = torch.from_numpy(np.zeros((b_input_size, 1)).astype(np.float32))
        self.b_input_prev = torch.from_numpy(np.zeros((b_input_size, 1)).astype(np.float32))

        # feedback weights
        self.Y = Y_range*np.random.uniform(-1, 1, size=(self.size, self.net.n[-1])).astype(np.float32)
        if use_sparse_feedback:
            # zero out a proportion of the feedback weights
            self.Y_dropout_indices = np.random.choice(len(self.Y.ravel()), int(sparse_feedback_prop*len(self.Y.ravel())), False)
            self.Y.ravel()[self.Y_dropout_indices] = 0
        self.Y = torch.from_numpy(self.Y)

        # apical voltage
        self.g = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))

    def update_b_input(self, b_input, b_eta, generate_activity=False, update_b_weights=False):
        # update previous feedback input
        self.b_input_prev = self.b_input.clone()

        # apply exponential smoothing to feedback input
        self.b_input = (self.b_input + b_input.clone())/2.0

        # calculate apical voltage
        self.g = self.Y.mm(self.b_input)

        # update previous burst probability
        self.burst_prob_prev = self.burst_prob.clone()

        # update burst probability
        self.burst_prob = torch.sigmoid(self.g)

        # update previous burst rate
        self.burst_rate_prev = self.burst_rate.clone()

        # update burst rate
        self.burst_rate = self.burst_prob*self.event_rate

        if update_b_weights:
            # calculate feedback loss
            self.backward_loss = torch.mean((self.event_rate - self.burst_rate)**2)

            # calculate error term
            E = self.event_rate*(self.event_rate - self.burst_rate)*-self.burst_prob*(1.0 - self.burst_prob)

            # update feedback weights
            self.update_Y(b_eta, E)
            self.decay_Y()

        if generate_activity:
            # generate activity using feedback, by setting the event rate to be equal to the burst rate (which is determined by feedback)
            self.event_rate = self.burst_rate.clone()

    def burst(self, f_eta):
        # calculate feedforward loss
        self.loss = torch.mean((self.event_rate_prev + self.burst_rate - self.burst_rate_prev - self.event_rate_prev)**2)

        # calculate error term
        E = (self.burst_rate - self.burst_rate_prev)*-self.event_rate_prev*(1.0 - self.event_rate_prev)

        # update feedforward weights
        self.update_W(f_eta, E)
        self.decay_W()

    def update_Y(self, b_eta, E):
        # update feedback weights
        self.delta_Y = E.mm(self.b_input_prev.t())
        self.Y      += -b_eta*self.delta_Y

        if use_sparse_feedback:
            # zero out a proportion of the feedback weights
            self.Y = self.Y.numpy()

            self.Y.ravel()[self.Y_dropout_indices] = 0

            self.Y = torch.from_numpy(self.Y)

    def decay_Y(self):
        self.Y -= self.net.weight_decay*self.Y

class finalLayer(Layer):
    def __init__(self, net, m, f_input_size):
        Layer.__init__(self, net, m, f_input_size)

    def burst(self, f_eta):
        # calculate loss
        self.loss = torch.mean((self.target - self.event_rate_prev)**2)

        # calculate error term
        E = (self.event_rate - self.event_rate_prev)*-self.event_rate_prev*(1.0 - self.event_rate_prev)

        # update feedforward weights
        self.update_W(f_eta, E)
        self.decay_W()

    def update_activity(self, f_input, target=None):
        Layer.update_f_input(self, f_input)

        if target is not None:
            # set the target activity
            self.target = target.clone()

            # push the event rate towards the target
            self.event_rate = (self.event_rate + self.target.clone())/2.0

        # update spike rate
        self.spike_rate = self.event_rate