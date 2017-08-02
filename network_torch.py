from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import copy
import datetime
import os
import pdb
import sys
import time
import shutil
import json
from scipy.special import expit
import pdb

if sys.version_info >= (3,):
    xrange = range

# ---------------------------------------------------------------
"""                 Simulation parameters                     """
# ---------------------------------------------------------------

default_simulations_folder = 'Simulations/' # folder in which to save simulations (edit accordingly)

sequence_length    = 2000 # length of the input sequence to be repeated
n_spikes_per_burst = 10   # number of spikes in each burst
teach_prob         = 0.05 # probability of a teaching signal being provided

visualize_while_training = False # show a plot of the output & target output of the network during training
use_sparse_feedback      = False # zero out a proportion of the feedback weights
sparse_feedback_prop     = 0.5   # proportion of feedback weights to set to 0

# uniform distribution ranges for initial weights
W_range = 0.1
Y_range = 50

# ---------------------------------------------------------------
"""                     Functions                             """
# ---------------------------------------------------------------

def create_data():
    '''
    Generate input & target data using sine & cosine functions.
    '''

    x_set = np.zeros((n_in, sequence_length)).astype(np.float32)
    for i in range(n_in):
        # x = 2*(np.arange(sequence_length)/sequence_length - 0.5)*5
        # np.random.shuffle(x)
        # x_set[i] = x

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

def sigmoid(x):
    return expit(x)

# ---------------------------------------------------------------
"""                     Network class                         """
# ---------------------------------------------------------------

# create input & target data
x_set, t_set = load_data()
# x_set, t_set = create_data()

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
            for m in xrange(1, self.M):
                self.l.append(hiddenLayer(net=self, m=m-1, f_input_size=self.n[m-1], b_input_size=self.n[-1]))
            self.l.append(finalLayer(net=self, m=self.M-1, f_input_size=self.n[-2]))

    def out(self, x, t, prev_t, time):
        '''
        Simulate the network's activity over one timestep.

        Arguments:
            x (tensor)           : The activity of the input layer for this time step.
            t (tensor/None)      : The target activity for the output layer for this time step.
            prev_t (tensor/None) : The target activity for the output layer for the previous time step.
            time (int)           : The current time step.
        '''

        if self.M == 1:
            if time >= 1:
                self.l[0].update_activity(x, t)

            if t is not None:
                self.l[0].burst(self.f_etas[0])
        else:
            if time >= self.M:
                self.l[-1].update_activity(self.l[-2].event_rate, t)

            for m in xrange(self.M-2, -1, -1):
                if time >= m+1:
                    if m == 0:
                        self.l[0].update_f_input(x)
                    else:
                        self.l[m].update_f_input(self.l[m-1].event_rate)

            for m in xrange(self.M-1):
                if time > self.M+1:
                    self.l[m].update_b_input(self.l[-1].event_rate)

                    if prev_t is not None:
                        self.l[m].burst(self.f_etas[m], self.b_etas[m])

            if t is not None:
                self.l[-1].burst(self.f_etas[-1])

    def train(self, f_etas, b_etas, n_epochs, plot_activity=False, weight_decay=0):
        '''
        Train the network.

        Arguments:
            f_etas (list/tuple/int) : The learning rates for the feedforward weights.
                                      If an int is provided, each layer will have the same learning rate.
            b_etas (list/tuple/int) : The learning rates for the feedback weights.
                                      If an int is provided, each layer will have the same learning rate.
            n_epochs (int)          : The number of epochs of training.
            plot_activity (bool)    : Whether to create a plot that compares the output & targets for the network.
            weight_decay (int)      : Weight decay constant.
        '''

        print("Starting training.\n")

        self.weight_decay = weight_decay

        # set learning rate instance variables
        self.f_etas = f_etas
        self.b_etas = b_etas

        if plot_activity:
            # set colors for plotting the output & target output of the network during training
            colors = ["red", "blue", "green", "purple", "brown", "cyan", "orange", "magenta"]

            if len(colors) < self.n[-1]:
                raise Exception("Number of output neurons exceeds the number of defined colors for plotting.")

            # create the figure
            self.figure = plt.figure(figsize=(15, 6), facecolor='white')
            self.animation_axis = plt.Axes(self.figure, [0.07, 0.07, 0.86, 0.36])
            self.figure.add_axes(self.animation_axis)
            self.target_lines = [ self.animation_axis.plot([], [], color=colors[i], lw=1)[0] for i in range(self.n[-1]) ]
            self.output_lines = [ self.animation_axis.plot([], [], color=colors[i], lw=1, linestyle='--', alpha=0.5)[0] for i in range(self.n[-1]) ]

            self.animation_axis_2 = plt.Axes(self.figure, [0.07, 0.57, 0.86, 0.36])
            self.figure.add_axes(self.animation_axis_2)
            self.loss_line = self.animation_axis_2.plot([], [], color='red', lw=1)[0]

            # show the plot
            plt.ion()
            plt.show()

            # initialize list of scatter points
            scatter_points = []

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

        # initialize target
        t = None

        for k in xrange(n_epochs):
            # clear targets & outputs arrays
            self.targets *= 0
            self.outputs *= 0

            for time in xrange(sequence_length):
                # set previous target
                if t is not None:
                    prev_t = t.clone()
                else:
                    prev_t = None

                # get input & target for this time step
                self.x = get_x(time)
                self.t = get_t(time)

                if (k < n_epochs-int(plot_activity) and np.random.uniform(0, 1) >= 1 - teach_prob):
                    no_t = False
                    t    = self.t
                else:
                    no_t = True
                    t    = None
                    no_t_count += 1

                # simulate network activity for this time step
                self.out(self.x, t, prev_t, time)

                # add the loss to average loss, only if a target was present
                if not no_t:
                    for m in xrange(self.M):
                        avg_losses[m, counter] += float(self.l[m].loss)

                # record targets & outputs for this time step
                self.targets[k*sequence_length + time] = self.t.numpy()[:, 0]
                self.outputs[k*sequence_length + time] = self.l[-1].event_rates_prev[-1].numpy()[:, 0]

                if visualize_while_training and (not no_t) and plot:
                    # add scatter points
                    for l in xrange(self.n[-1]):
                        scatter_point = self.animation_axis.scatter(k*sequence_length + time, self.l[-1].event_rate.numpy()[l, 0], c=colors[l], s=10)
                        scatter_points.append(scatter_point)

                if (time+1) % 100 == 0 and k < n_epochs-int(plot_activity):
                    # compute average loss over the last 100 time steps
                    # minus those where a target wasn't present
                    if 100 - no_t_count > 0:
                        avg_losses[:, counter] /= (100 - no_t_count)
                    no_t_count = 0
                    counter += 1

                if (time+1) % 100 == 0 and k < n_epochs-int(plot_activity):
                    print("Epoch {:>3d}, t={:>4d}. Average loss: {:.10f}.".format(k+1, time+1, avg_losses[-1, counter-1]))

                    no_t_count = 0

            if visualize_while_training and plot_activity:
                # update plot
                for l in xrange(self.n[-1]):
                    min_x = max(0, (k+1)*sequence_length-2000)
                    max_x = (k+1)*sequence_length
                    self.target_lines[l].set_data(np.arange(min_x, max_x), self.targets[min_x:max_x, l])
                    self.output_lines[l].set_data(np.arange(min_x, max_x), self.outputs[min_x:max_x, l])

                    self.animation_axis.relim()
                    self.animation_axis.autoscale_view(scalex=True, scaley=True)

                    self.loss_line.set_data(range(counter), avg_losses[-1, :counter])
                    self.animation_axis_2.relim()
                    self.animation_axis_2.autoscale_view(scalex=True, scaley=True)

                plt.draw()
                plt.pause(1)

                for i in range(len(scatter_points)):
                    scatter_points[i].remove()
                scatter_points = []

        if not visualize_while_training and plot_activity:
            # create plot
            for l in xrange(self.n[-1]):
                min_x = max(0, n_epochs*sequence_length-2000)
                max_x = n_epochs*sequence_length
                self.target_lines[l].set_data(np.arange(min_x, max_x), self.targets[min_x:max_x, l])
                self.output_lines[l].set_data(np.arange(min_x, max_x), self.outputs[min_x:max_x, l])

                self.animation_axis.relim()
                self.animation_axis.autoscale_view(scalex=True, scaley=True)

                self.loss_line.set_data(range(len(avg_losses[-1])), avg_losses[-1])
                self.animation_axis_2.relim()
                self.animation_axis_2.autoscale_view(scalex=True, scaley=True)

            plt.draw()
            plt.pause(100000)

        return avg_losses

    def save_weights(self, path, prefix=""):
        '''
        Save the network's current weights to .npy files.

        Arguments:
            path (string)   : The path of the folder in which to save the network's weights.
            prefix (string) : A prefix to append to the filenames of the saved weights.
        '''

        for m in xrange(self.M):
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

        for m in xrange(self.M):
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
        
        for m in xrange(self.M):
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

        self.f_input       = torch.from_numpy(np.zeros((f_input_size, 1)).astype(np.float32))
        self.f_inputs_prev = [ torch.from_numpy(np.zeros((f_input_size, 1)).astype(np.float32)) for i in range(10) ]

        self.y_pre  = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        
        self.spike_rate       = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        self.burst_rate       = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        self.event_rate       = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        self.event_rates_prev = [ torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32)) for i in range(10) ]

        self.E = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))

        self.loss = 0

        self.W = torch.from_numpy(W_range*np.random.uniform(-1, 1, size=(self.size, f_input_size)).astype(np.float32))
        self.b = torch.from_numpy(0.1*np.ones((self.size, 1)).astype(np.float32))

        self.delta_W = torch.from_numpy(np.zeros(self.W.size()).astype(np.float32))
        self.delta_b = torch.from_numpy(np.zeros(self.b.size()).astype(np.float32))

    def update_f_input(self, f_input):
        # update previous feedforward input
        del self.f_inputs_prev[0]
        self.f_inputs_prev.append(self.f_input.clone())

        # update feedforward input
        self.f_input = (self.f_input + f_input.clone())/2.0

        # calculate pre-nonlinearity activity
        self.y_pre = self.W.mm(self.f_input) + self.b

        # update previous event rate
        del self.event_rates_prev[0]
        self.event_rates_prev.append(self.event_rate.clone())

        # update event rate
        self.event_rate = torch.sigmoid(self.y_pre)

    def update_W(self, f_eta):
        self.delta_W = self.E.mm(self.f_input.t())
        self.W      += -f_eta*self.delta_W

        self.delta_b = self.E
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

        self.b_input  = torch.from_numpy(np.zeros((b_input_size, 1)).astype(np.float32))

        # create feedback weights
        self.Y = Y_range*np.random.uniform(-1, 1, size=(self.size, self.net.n[-1])).astype(np.float32)

        if use_sparse_feedback:
            # zero out a proportion of the feedback weights
            self.Y_dropout_indices = np.random.choice(len(self.Y.ravel()), int(sparse_feedback_prop*len(self.Y.ravel())), False)
            self.Y.ravel()[self.Y_dropout_indices] = 0

        self.Y = torch.from_numpy(self.Y)

        self.g  = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        self.g_prev  = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))

    def update_b_input(self, b_input):
        self.b_input = (self.b_input + b_input.clone())/2.0

        self.g_prev = self.g.clone()

        self.g = self.Y.mm(self.b_input)

    def update_f_input(self, f_input):
        Layer.update_f_input(self, f_input)

        self.spike_rate = (1.0 - self.burst_rate)*self.event_rate + self.burst_rate*self.event_rate*n_spikes_per_burst

    def burst(self, f_eta, b_eta):
        self.burst_rate = torch.sigmoid(self.g - self.g_prev)

        self.E = (2*self.burst_rate - 1)*-self.event_rates_prev[-1]*(1.0 - self.event_rates_prev[-1])

        self.E_inv = (self.g - self.g_prev)*-1

        self.loss = torch.mean((2*self.burst_rate - 1)**2)

        self.update_W(f_eta)

        self.decay_W()

        self.update_Y(b_eta)

        self.decay_Y()

    def update_Y(self, b_eta):
        self.delta_Y = self.E_inv.mm(self.b_input.t())
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

        self.b_input  = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))

    def burst(self, f_eta):
        self.burst_rate = torch.sigmoid(self.event_rate - self.event_rates_prev[-1])

        self.E = (self.event_rate - self.event_rates_prev[-1])*-self.event_rates_prev[-1]*(1.0 - self.event_rates_prev[-1])

        self.loss = torch.mean((self.b_input - self.event_rates_prev[-1])**2)

        self.update_W(f_eta)

        self.decay_W()

    def update_activity(self, f_input, b_input=None):
        Layer.update_f_input(self, f_input)

        if b_input is not None:
            self.b_input = b_input.clone()

            self.event_rate = self.b_input.clone()

        self.spike_rate = self.event_rate