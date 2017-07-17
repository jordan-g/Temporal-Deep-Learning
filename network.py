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
weight_cmap                = 'bone'         # color map to use for weight plotting

sequence_length = 2000
n_in            = 500
n_out           = 3
visualize_while_training = False
use_skip_connections = False
use_sparse_feedback = False

teach_prob = 0.2
W_range = [0.5, 0.5, 0.5, 0.5, 0.5]
W_2_range = [0.5, 0.5]
b_range = [0.1, 0.1, 0.1, 0.1, 0.1]
Y_range = [1, 1, 1, 1]
Y_2_range = [1, 1, 1]
c_range = [0.1, 0.1, 0.1, 0.1]

x_set = np.zeros((n_in, sequence_length)).astype(np.float32)
for i in range(n_in):
    x = 2*(np.arange(sequence_length)/sequence_length - 0.5)*5
    np.random.shuffle(x)
    x_set[i] = x

t_set = np.zeros((n_out, sequence_length)).astype(np.float32)
for i in range(n_out):
    x = np.arange(sequence_length)
    t_set[i] = 0.2 + 0.5*(np.sin(np.random.uniform(0.01, 0.1)*x + np.random.uniform(0, 20)) + 1)*np.random.uniform(0.1, 0.3)
    t_set[i] += 0.5*(np.cos(np.random.uniform(0.01, 0.05)*x + np.random.uniform(0, 20)) + 1)*np.random.uniform(0.1, 0.3)

# np.save("x_set.npy", x_set)
# np.save("t_set.npy", t_set)

x_set = np.load("x_set.npy")
t_set = np.load("t_set.npy")

colors = ["red", "blue", "green", "yellow", "brown", "purple", "white", "cyan", "orange", "magenta"][:n_out]

# ---------------------------------------------------------------
"""                     Functions                             """
# ---------------------------------------------------------------

def get_x(n):
    return x_set[:, n][:, np.newaxis]

def get_t(n):
    return t_set[:, n][:, np.newaxis]

def sigmoid(x):
    return expit(x)

# ---------------------------------------------------------------
"""                     Network class                         """
# ---------------------------------------------------------------

class Network:
    def __init__(self, n, use_h=False):
        '''
        Initialize the network. Note: This also loads the MNIST dataset.

        Arguments:
            n (tuple) : Number of units in each layer of the network, eg. (500, 100, 10).
        '''

        if type(n) == int:
            n = (n,)

        self.n = n           # layer sizes - eg. (500, 100, 10)
        self.M = len(self.n) # number of layers

        self.n_in  = get_x(0).shape[0] # input size
        self.n_out = self.n[-1]      # output size

        self.use_h = use_h

        self.current_epoch = None # current epoch of simulation

        print("Creating network with {} layers.".format(self.M))
        print("--------------------------------")

        self.init_layers(use_h=use_h)

    def init_layers(self, use_h=False):
        '''
        Create the layers of the network.
        '''

        # initialize layers list
        self.l = []

        # create all layers
        if self.M == 1:
            self.l.append(finalLayer(net=self, m=-1, f_input_size=self.n_in, use_h=use_h))
        else:
            self.l.append(hiddenLayer(net=self, m=0, f_input_size=self.n_in, b_input_size=self.n[1], use_h=use_h))
            for m in xrange(1, self.M-1):
                self.l.append(hiddenLayer(net=self, m=m, f_input_size=self.n[m-1], b_input_size=self.n[m+1], use_h=use_h))
            self.l.append(finalLayer(net=self, m=self.M-1, f_input_size=self.n[-2], use_h=use_h))

    def out(self, x, t):
        '''
        Perform a pass through the network and update weights.
        '''

        if self.M == 1:
            self.l[0].out(x, t)
        else:
            self.l[0].out(x, self.l[1].y_backward)

            for m in xrange(1, self.M-1):
                self.l[m].out(self.l[m-1].y_forward, self.l[m+1].y_backward)

            self.l[-1].out(self.l[-2].y_forward, t)

            for m in xrange(self.M-2, 0, -1):
                self.l[m].out(self.l[m-1].y_forward, self.l[m+1].y_backward)

            self.l[0].out(x, self.l[1].y_backward)

        if t is not None:
            for m in xrange(self.M-1, -1, -1): # for the hidden layers:
                # update weights
                self.l[m].calc_error()

            # update feedforward weights for the final layer
            self.l[-1].update_W(self.f_etas[-1])

            for m in xrange(self.M-2, -1, -1): # for the hidden layers:
                # update weights
                self.l[m].update_W(self.f_etas[m])
        # else:
            for m in xrange(self.M-2, -1, -1): # for the hidden layers:
                self.l[m].update_Y(self.b_etas[m])

    def train(self, f_etas, b_etas, n_epochs, save_simulation, simulations_folder=default_simulations_folder, folder_name="", overwrite=False, simulation_notes=None, current_epoch=None, plot=False):
        print("Starting training.\n")

        if current_epoch != None:
            self.current_epoch == current_epoch
        elif self.current_epoch == None:
            # set current epoch
            self.current_epoch = 0

        if self.current_epoch == 0:
            continuing = False
        else:
            continuing = True

        # get current date/time and create simulation directory
        if save_simulation:
            sim_start_time = datetime.datetime.now()

            if folder_name == "":
                self.simulation_path = os.path.join(simulations_folder, "{}.{}.{}-{}.{}".format(sim_start_time.year,
                                                                                 sim_start_time.month,
                                                                                 sim_start_time.day,
                                                                                 sim_start_time.hour,
                                                                                 sim_start_time.minute))
            else:
                self.simulation_path = os.path.join(simulations_folder, folder_name)

            # make simulation directory
            if not os.path.exists(self.simulation_path):
                os.makedirs(self.simulation_path)
            elif not continuing:
                if overwrite == False:
                    print("Error: Simulation directory \"{}\" already exists.".format(self.simulation_path))
                    return
                else:
                    shutil.rmtree(self.simulation_path, ignore_errors=True)
                    os.makedirs(self.simulation_path)

            # copy current script to simulation directory
            filename = os.path.basename(__file__)
            if filename.endswith('pyc'):
                filename = filename[:-1]
            shutil.copyfile(filename, os.path.join(self.simulation_path, filename))

            params = {
                'n'                      : self.n,
                'f_etas'                 : f_etas,
                'b_etas'                 : b_etas,
                'sequence_length'        : sequence_length,
                'n_epochs'               : n_epochs
            }

            # save simulation params
            if not continuing:
                with open(os.path.join(self.simulation_path, 'simulation.txt'), 'w') as simulation_file:
                    print("Simulation done on {}.{}.{}-{}.{}.".format(sim_start_time.year,
                                                                     sim_start_time.month,
                                                                     sim_start_time.day,
                                                                     sim_start_time.hour,
                                                                     sim_start_time.minute), file=simulation_file)
                    if simulation_notes:
                        print(simulation_notes, file=simulation_file)
                    print("Start time: {}".format(sim_start_time), file=simulation_file)
                    print("-----------------------------", file=simulation_file)
                    for key, value in sorted(params.items()):
                        line = '{}: {}'.format(key, value)
                        print(line, file=simulation_file)

                with open(os.path.join(self.simulation_path, 'simulation.json'), 'w') as simulation_file:
                    simulation_file.write(json.dumps(params))

        # set learning rate instance variables
        self.f_etas = f_etas
        self.b_etas = b_etas

        if save_simulation and not continuing:
            # save initial weights
            self.save_weights(self.simulation_path, prefix='initial_')

        print("Start of epoch {}.\n".format(self.current_epoch + 1))

        # start time used for timing how long each 1000 examples take
        start_time = None

        avg_loss = 0
        no_t_count = 0

        if plot:
            # Create the figure
            self.figure = plt.figure(figsize=(15, 6), facecolor='white')
            self.animation_axis = plt.Axes(self.figure, [0.07, 0.07, 0.86, 0.36])
            self.figure.add_axes(self.animation_axis)
            self.target_lines = [ self.animation_axis.plot([], [], color=colors[i], lw=1)[0] for i in range(self.n_out) ]
            self.output_lines = [ self.animation_axis.plot([], [], color=colors[i], lw=1, linestyle='--', alpha=0.5)[0] for i in range(self.n_out) ]

            self.animation_axis_2 = plt.Axes(self.figure, [0.07, 0.57, 0.86, 0.36])
            self.figure.add_axes(self.animation_axis_2)
            self.loss_line = self.animation_axis_2.plot([], [], color='red', lw=1)[0]

            # Show the plot
            plt.ion()
            plt.show()

        avg_losses = np.zeros((self.M, int(n_epochs*sequence_length/100.0)))
        counter = 0
        scatter_points = []

        self.targets = np.zeros((sequence_length*n_epochs, self.n_out))
        self.outputs = np.zeros((sequence_length*n_epochs, self.n_out))

        show_target = False

        for k in xrange(n_epochs):
            self.targets *= 0
            self.outputs *= 0

            for n in xrange(sequence_length):
                # set start time
                if start_time == None:
                    start_time = time.time()

                # print every 100 examples
                # if (n+1) % 100 == 0:
                #     sys.stdout.write("\x1b[2K\rEpoch {0}, example {1}/{2}. ".format(self.current_epoch + 1, n+1, sequence_length))
                #     sys.stdout.flush()

                self.x = get_x(n)
                self.t = get_t(n)

                # self.x, self.t = next(iter(train_loader))

                # print(self.t)

                if (k < n_epochs-1 and np.random.uniform(0, 1) >= 1 - teach_prob):
                    no_t = False
                    t    = self.t
                    show_target = False
                else:
                    no_t = True
                    t    = None
                    no_t_count += 1

                # if k == n_epochs - 1 and n > 0:
                #     self.x *= 0

                # do a pass through the network & update weights
                self.out(self.x, t)

                # t_2 = self.t

                if not no_t:
                    for m in xrange(self.M):
                        avg_losses[m, counter] += self.l[m].loss

                    # if self.l[-1].loss > 0.002:
                    #     show_target = True

                # avg_loss += self.l[-1].loss

                self.targets[k*sequence_length + n] = self.t[:, 0]
                self.outputs[k*sequence_length + n] = self.l[-1].y_forward[:, 0]

                if visualize_while_training and (not no_t) and plot:
                    for l in xrange(self.n_out):
                        scatter_point = self.animation_axis.scatter(n, self.l[-1].y_forward[l], c=colors[l], s=10)
                        scatter_points.append(scatter_point)

                if (n+1) % 100 == 0:
                    if 100 - no_t_count > 0:
                        avg_losses[:, counter] /= (100 - no_t_count)
                    no_t_count = 0
                    counter += 1

                if (n+1) % 1000 == 0:
                    # if n != sequence_length - 1:
                    #     sys.stdout.write("\x1b[2K\rEpoch {0}, example {1}/{2}. ".format(self.current_epoch + 1, n+1, sequence_length))

                    print("Epoch {}, t={}. Average loss: {}. ".format(self.current_epoch + 1, n+1, avg_losses[-1, counter-1]))

                    no_t_count = 0

                    # get end time & reset start time
                    end_time = time.time()
                    time_elapsed = end_time - start_time
                    print("T: {0:.3f}s.\n".format(time_elapsed))
                    start_time = None

            if visualize_while_training and plot:
                for l in xrange(self.n_out):
                    min_x = max(0, k*sequence_length-1000)
                    max_x = k*n_epochs
                    self.target_lines[l].set_data(np.arange(min_x, max_x), self.targets[min_x:max_x, l])
                    self.output_lines[l].set_data(np.arange(min_x, max_x), self.outputs[min_x:max_x, l])

                    self.animation_axis.relim()
                    self.animation_axis.autoscale_view(scalex=True, scaley=True)

                    self.loss_line.set_data(range(len(avg_losses)), avg_losses)
                    self.animation_axis_2.relim()
                    self.animation_axis_2.autoscale_view(scalex=True, scaley=True)

                plt.draw()
                plt.pause(1)

                for i in range(len(scatter_points)):
                    scatter_points[i].remove()
                scatter_points = []

            # update latest epoch counter
            self.current_epoch += 1

        if not visualize_while_training and plot:
            for l in xrange(self.n_out):
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

        # record end time of training
        if save_simulation:
            with open(os.path.join(self.simulation_path, 'simulation.txt'), 'a') as simulation_file:
                sim_end_time = datetime.datetime.now()
                print("-----------------------------", file=simulation_file)
                print("End time: {}".format(sim_end_time), file=simulation_file)

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

        # print network weights
        self.print_weights()

        print("--------------------------------")

    def set_weights(self, W_list, b_list, Y_list, c_list):
        for m in xrange(self.M):
            self.l[m].W = W_list[m].copy()
            self.l[m].b = b_list[m].copy()

            if m < self.M-1:
                self.l[m].Y = Y_list[m].copy()
                self.l[m].c = c_list[m].copy()

                if use_sparse_feedback:
                    self.l[m].Y_dropout_indices = np.random.choice(len(self.l[m].Y.ravel()), int(0.5*len(self.l[m].Y.ravel())), False)
                    self.l[m].Y.ravel()[self.l[m].Y_dropout_indices] = 0

# ---------------------------------------------------------------
"""                     Layer classes                         """
# ---------------------------------------------------------------

class Layer:
    def __init__(self, net, m, f_input_size, use_h=False):
        self.net      = net
        self.m        = m
        self.size     = self.net.n[m]
        self.y_forward  = np.zeros((self.size, 1))
        self.y_backward = np.zeros((self.size, 1))
        self.E = np.zeros((self.size, 1))
        self.y_prev   = np.zeros((self.size, 1))
        self.t        = np.zeros((self.size, 1))
        self.t_prev   = np.zeros((self.size, 1))
        self.f_input  = np.zeros((f_input_size, 1))
        self.b_input  = np.zeros((self.size, 1))
        self.b_input_old = np.zeros((self.size, 1))

        self.W = W_range[self.m]*np.random.uniform(-1, 1, size=(self.size, f_input_size))
        self.b = b_range[self.m]*np.ones((self.size, 1))

    def out(self, f_input):
        self.f_input = f_input

        self.y_forward = sigmoid(np.dot(self.W, self.f_input) + self.b)

        self.y_prev = self.y_forward.copy()

    def update_W(self, f_eta):
        self.delta_W = np.dot(self.E, self.f_input.T)
        self.W      += -f_eta*self.delta_W

        self.delta_b = self.E
        self.b      += -f_eta*self.delta_b

class hiddenLayer(Layer):
    def __init__(self, net, m, f_input_size, b_input_size, use_h=False):
        Layer.__init__(self, net, m, f_input_size, use_h=use_h)

        self.a_backward = np.zeros((self.size, 1)).astype(np.float32)
        self.a_forward  = np.zeros((self.size, 1)).astype(np.float32)
        self.Y = Y_range[self.m]*np.random.uniform(-1, 1, size=(self.size, b_input_size)).astype(np.float32)

        self.Y_2 = Y_2_range[self.m]*np.random.uniform(-1, 1, size=(self.size, self.net.n[-1])).astype(np.float32)

        if use_sparse_feedback:
            self.Y_dropout_indices = np.random.choice(len(self.Y.ravel()), int(0.5*len(self.Y.ravel())), False)
            self.Y.ravel()[self.Y_dropout_indices] = 0

        # self.Y_2_dropout_indices = np.random.choice(len(self.Y_2.ravel()), int(0.5*len(self.Y_2.ravel())), False)
        # self.Y_2.ravel()[self.Y_2_dropout_indices] = 0

        # remove_percent = 0.8
        # self.Y_dropout_indices = torch.LongTensor([np.random.choice(self.size, int(remove_percent*self.size), False).tolist(), np.random.choice(b_input_size, int(remove_percent*self.size), True).tolist()])
        # self.Y_dropout_values = torch.FloatTensor([1]*int(remove_percent*self.size))
        # self.Y_mask = torch.sparse.FloatTensor(self.Y_dropout_indices, self.Y_dropout_values, torch.Size([self.size, b_input_size]))
        # for i in range(self.size):
        #     for j in range(b_input_size):
        #         if np.random.uniform(0, 1) > 0.2:
        #             self.Y_mask[i, j] = 0
        # self.Y_mask = torch.from_numpy(self.Y_mask)
        # self.Y.sparse_mask(self.Y_mask)
        # self.Y *= 5

        self.c = c_range[self.m]*np.ones((self.size, 1)).astype(np.float32)

    def calc_error(self):
        # self.a_forward  = self.net.l[self.m+1].W.t().mm(self.net.l[self.m+1].a_forward)
        # self.a_backward = self.y_forward + torch.sigmoid(self.Y.mm(self.net.l[self.m+1].a_backward)) - torch.sigmoid(self.Y.mm(self.net.l[self.m+1].y_forward))

        self.E = ((self.y_backward - self.y_forward))*-self.y_forward*(1.0 - self.y_forward)

        self.loss = np.mean((self.y_backward - self.y_forward)**2)

    def out(self, f_input, b_input):
        Layer.out(self, f_input)

        # self.b_input = b_input

        # self.a_backward = torch.sigmoid(self.net.l[self.m+1].W.t().mm(self.net.l[self.m+1].a_backward))
        if use_skip_connections:
            self.y_backward = self.y_forward + np.tanh(np.dot(self.Y, (self.net.l[self.m+1].y_backward - self.net.l[self.m+1].y_forward))) + np.tanh(np.dot(self.Y_2, (self.net.l[-1].y_backward - self.net.l[-1].y_forward)))
        else:
            # self.y_backward = self.y_forward + np.tanh(np.dot(self.Y, (self.net.l[self.m+1].y_backward - self.net.l[self.m+1].y_forward)))
            self.y_backward = self.y_forward + np.tanh(np.dot(self.Y_2, (self.net.l[-1].y_backward - self.net.l[-1].y_forward)))

    def update_Y(self, b_eta):
        pass
        # a = torch.sigmoid(self.Y.mm(self.net.l[self.m+1].y_forward))
        # self.E_inv = torch.sigmoid(self.Y.mm(self.net.l[self.m+1].y_backward) - self.Y.mm(self.net.l[self.m+1].y_forward))*(1.0 - torch.sigmoid(self.Y.mm(self.net.l[self.m+1].y_backward) - self.Y.mm(self.net.l[self.m+1].y_forward)))
        # self.delta_Y = self.E_inv.mm((self.net.l[self.m+1].y_backward - self.net.l[self.m+1].y_forward).t())
        # self.Y += np.random.normal(0, 0.001, size=self.Y.shape)
        # self.Y_2 += np.random.normal(0, 0.001, size=self.Y_2.shape)

        # self.Y[self.Y_dropout_indices_1, self.Y_dropout_indices_2] = 0
        # self.Y.sparse_mask(self.Y_mask)
        # self.Y *= 5

        # self.delta_c = self.E_inv
        # self.c      += -b_eta*self.delta_c

class finalLayer(Layer):
    def __init__(self, net, m, f_input_size, use_h=False):
        Layer.__init__(self, net, m, f_input_size, use_h=use_h)

    def calc_error(self):
        self.E = (self.y_backward - self.y_forward)*-self.y_forward*(1.0 - self.y_forward)

        self.loss = np.mean((self.y_backward - self.y_forward)**2)

    def out(self, f_input, b_input):
        Layer.out(self, f_input)

        # self.a_forward = self.y_forward

        # self.b_input = b_input


        if b_input is not None:
            self.y_backward = b_input
            # self.a_backward = b_input

# ---------------------------------------------------------------
"""                     Helper functions                      """
# ---------------------------------------------------------------

def load_simulation(latest_epoch, folder_name, simulations_folder=default_simulations_folder):
    simulation_path = os.path.join(simulations_folder, folder_name)

    print("Loading simulation from \"{}\" @ epoch {}.\n".format(simulation_path, latest_epoch))

    if not os.path.exists(simulation_path):
        print("Error: Could not find simulation folder – path does not exist.")
        return None

    # load parameters
    with open(os.path.join(simulation_path, 'simulation.json'), 'r') as simulation_file:
        params = json.loads(simulation_file.read())

    global sequence_length

    n                       = params['n']
    f_etas                  = params['f_etas']
    b_etas                  = params['b_etas']
    sequence_length         = params['sequence_length']

    # create network and load weights
    net = Network(n=n)
    net.load_weights(simulation_path, prefix="epoch_{}_".format(latest_epoch))
    net.current_epoch = latest_epoch + 1

    return net, f_etas, b_etas