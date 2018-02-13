import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy import interpolate
from scipy.special import expit
import pdb

class Network:
    def __init__(self, n, n_in, W_ranges, Y_ranges, Z_ranges):
        self.n        = n           # layer sizes - eg. (500, 100, 10)
        self.n_layers = len(self.n) # number of layers

        # initialize layers list
        self.layers = []

        if self.n_layers == 1:
            self.layers.append(outputLayer(self, 0, size=self.n[0], f_input_size=n_in, W_range=W_ranges[0]))
        else:
            for layer_num in range(self.n_layers-1):
                if layer_num == 0:
                    self.layers.append(hiddenLayer(self, layer_num, size=self.n[layer_num], f_input_size=n_in, b_input_size=self.n[1], W_range=W_ranges[0], Y_range=Y_ranges[0], Z_range=Z_ranges[0]))
                else:
                    self.layers.append(hiddenLayer(self, layer_num, size=self.n[layer_num], f_input_size=self.n[layer_num-1], b_input_size=self.n[layer_num+1], W_range=W_ranges[layer_num], Y_range=Y_ranges[layer_num], Z_range=Z_ranges[layer_num]))
            self.layers.append(outputLayer(self, self.n_layers-1, size=self.n[-1], f_input_size=self.n[-2], W_range=W_ranges[-1]))

    def forward(self, x):
        if self.n_layers == 1:
            self.layers[0].forward(x)
        else:
            for layer_num in range(self.n_layers-1):
                if layer_num == 0:
                    self.layers[0].forward(x)
                else:
                    self.layers[layer_num].forward(self.layers[layer_num-1].event_rate)
            self.layers[-1].forward(self.layers[-2].event_rate)

    def backward(self, x, t, f_etas, r_etas):
        losses = np.zeros(self.n_layers)
        if self.n_layers == 1:
            losses[0] = self.layers[0].backward(t, f_etas[0])
        else:
            losses[-1] = self.layers[-1].backward(t, f_etas[-1])
            for layer_num in range(self.n_layers-2, -1, -1):
                losses[layer_num] = self.layers[layer_num].backward(self.layers[layer_num+1].burst_rate, self.layers[layer_num+1].burst_rate_t, f_etas[layer_num], r_etas[layer_num])

        return losses

class hiddenLayer:
    def __init__(self, net, layer_num, size, f_input_size, b_input_size, W_range, Y_range, Z_range):
        self.net          = net
        self.layer_num    = layer_num
        self.size         = size
        self.f_input_size = f_input_size
        self.b_input_size = b_input_size

        # initialize feedforward weights & biases
        self.W = W_range*np.random.normal(0, 1, size=(self.size, self.f_input_size))
        self.b = np.zeros(self.size)

        # initialize feedback weights
        self.Y = Y_range*np.random.normal(0, 1, size=(self.size, self.b_input_size))

        # self.Q = W_range*np.ones((self.size, self.size))/np.sqrt(self.size)
        self.Z = Z_range*np.random.normal(0, 1, size=(self.size, self.size)) + 0.1

        # print(np.mean(self.Z))
        # self.d = np.zeros(self.size)

        self.f_input         = np.zeros(self.f_input_size)
        self.b_input         = np.zeros(self.b_input_size)
        self.event_rate      = np.zeros(self.size)
        self.burst_prob      = np.zeros(self.size)
        self.burst_prob_prev = np.zeros(self.size)
        self.burst_rate      = np.zeros(self.size) + 0.2
        self.burst_rate_prev = np.zeros(self.size)
        self.u               = np.zeros(self.size)

    def forward(self, f_input):
        self.f_input = f_input

        self.s = np.dot(self.W, self.f_input) + self.b

        self.event_rate = expit(self.s)

    def backward(self, b_input, target_input, f_eta, r_eta):
        c = np.dot(self.Z, self.event_rate)

        self.u   = np.dot(self.Y, b_input)/c
        self.u_t = np.dot(self.Y, target_input)/c

        # update burst probability
        self.burst_prob   = expit(self.u)
        self.burst_prob_t = expit(self.u_t)

        self.burst_rate   = self.burst_prob*self.event_rate
        self.burst_rate_t = self.burst_prob_t*self.event_rate

        E_Z = (-self.u)*(self.u/c)

        self.Z -= r_eta*(np.outer(E_Z, self.event_rate) - (0.01 - self.Z))

        # calculate loss
        loss = 0.5*np.sum((self.u)**2) + 0.5*np.sum((0.01 - self.Z)**2)

        # E = (self.burst_rate_t - self.burst_rate)*-self.event_rate*(1.0 - self.event_rate)

        # update feedforward weights & biases
        # self.W -= f_eta*np.outer(E, self.f_input)
        # self.b -= f_eta*E

        return loss

class outputLayer:
    def __init__(self, net, layer_num, size, f_input_size, W_range, cuda=False):
        self.net          = net
        self.layer_num    = layer_num
        self.size         = size
        self.f_input_size = f_input_size

        # initialize feedforward weights & biases
        self.W = W_range*np.random.normal(0, 1, size=(self.size, self.f_input_size))
        self.b = np.zeros(self.size)

        self.f_input    = np.zeros(self.f_input_size)
        self.event_rate = np.zeros(self.size)
        self.burst_prob = np.zeros(self.size)
        self.burst_rate = np.zeros(self.size)
        self.u          = np.zeros(self.size)

        self.baseline_event_rate = 0.0001

    def forward(self, f_input):
        self.f_input = f_input

        self.event_rate = np.dot(self.W, self.f_input) + self.b
        self.event_rate[self.event_rate < self.baseline_event_rate] = self.baseline_event_rate

    def backward(self, t, f_eta):
        # calculate loss
        self.burst_prob = 0.2

        self.burst_rate   = self.burst_prob*self.event_rate
        self.burst_rate_t = self.burst_prob*(0.98*t + 0.01)

        loss = np.mean(((self.burst_rate_t - self.burst_rate))**2)

        # E = (self.burst_rate_t - self.burst_rate)*-self.burst_prob*(self.event_rate > self.baseline_event_rate).astype(int)
        # E = (self.burst_rate_t - self.burst_rate)*-self.burst_prob

        # update feedforward weights & biases
        # self.W -= f_eta*np.outer(E, self.f_input)
        # self.b -= f_eta*E

        return loss