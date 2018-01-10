import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy import interpolate
import pdb
from scipy.special import expit

troubleshooting = False

class Network:
    def __init__(self, n, n_in, W_std, Y_std):
        self.n        = n           # layer sizes - eg. (500, 100, 10)
        self.n_layers = len(self.n) # number of layers

        if type(W_std) is float:
            W_std = [W_std]*len(n)

        if type(Y_std) is float:
            Y_std = [Y_std]*(len(n)-1)

        # initialize layers list
        self.layers = []

        if self.n_layers == 1:
            self.layers.append(outputLayer(self, 0, size=self.n[0], f_input_size=n_in, W_std=W_std[0]))
        else:
            for layer_num in range(self.n_layers-1):
                if layer_num == 0:
                    self.layers.append(hiddenLayer(self, layer_num, size=self.n[layer_num], f_input_size=n_in, b_input_size=self.n[layer_num+1], W_std=W_std[layer_num], Y_std=Y_std[layer_num]))
                else:
                    self.layers.append(hiddenLayer(self, layer_num, size=self.n[layer_num], f_input_size=self.n[layer_num-1], b_input_size=self.n[layer_num+1], W_std=W_std[layer_num], Y_std=Y_std[layer_num]))
            self.layers.append(outputLayer(self, self.n_layers-1, size=self.n[-1], f_input_size=self.n[-2], W_std=W_std[-1]))

    def forward(self, x_prev):
        if self.n_layers == 1:
            self.layers[0].forward(x_prev)
        else:
            for layer_num in range(self.n_layers-1):
                if layer_num == 0:
                    self.layers[0].forward(x_prev)
                else:
                    self.layers[layer_num].forward(self.layers[layer_num-1].event_rate_prev)
            self.layers[-1].forward(self.layers[-2].event_rate_prev)

    def backward(self, t, f_etas, update_final_weights=False, update_hidden_weights=False):
        losses = np.zeros(self.n_layers)
        if self.n_layers == 1:
            losses[0] = self.layers[0].backward(t, f_etas[0], update_weights=update_final_weights)
        else:
            losses[-1] = self.layers[-1].backward(t, f_etas[-1], update_weights=update_final_weights)
            for layer_num in range(self.n_layers-2, -1, -1):
                losses[layer_num] = self.layers[layer_num].backward(self.layers[layer_num+1].burst_rate_prev, f_etas[layer_num], update_weights=update_hidden_weights)

        return losses

class hiddenLayer:
    def __init__(self, net, layer_num, size, f_input_size, b_input_size, W_std, Y_std):
        self.net          = net
        self.layer_num    = layer_num
        self.size         = size
        self.f_input_size = f_input_size
        self.b_input_size = b_input_size

        # initialize feedforward weights & biases
        self.W = np.random.normal(0, W_std, size=(self.size, self.f_input_size))
        self.b = np.ones(self.size)

        # initialize feedback weights
        self.Y = Y_std*np.random.uniform(-1, 1, size=(self.size, self.b_input_size))

        self.Q = W_std*np.random.uniform(-1, 1, size=(self.size, self.size))
        self.Z = np.random.uniform(-1, 1, size=(self.size, self.size))

        self.f_input         = np.zeros(self.f_input_size)
        self.b_input         = np.zeros(self.b_input_size)
        self.event_rate      = np.zeros(self.size)
        self.burst_prob      = np.zeros(self.size)
        self.burst_prob_prev = np.zeros(self.size)
        self.burst_rate      = np.zeros(self.size)
        self.burst_rate_prev = np.zeros(self.size)
        self.u               = np.zeros(self.size)

    def forward(self, f_input):
        # self.f_input = 0.8*f_input + 0.2*self.f_input
        self.f_input = f_input

        self.s = np.dot(self.W, self.f_input) + self.b

        self.event_rate_prev = self.event_rate
        self.event_rate = self.s

    def backward(self, b_input, f_eta, update_weights=False):
        self.b_input = b_input

        # self.q = self.Z.mv(self.r)

        # update burst probability
        self.u = np.dot(self.Y, self.b_input)
        
        self.burst_prob_prev = self.burst_prob
        self.burst_prob      = expit(self.u)

        self.burst_rate_prev = self.burst_rate
        self.burst_rate = self.burst_prob*self.event_rate

        if update_weights:
            # calculate loss
            loss = np.mean((self.burst_prob - self.burst_prob_prev)**2)

            E = (self.burst_rate - self.burst_rate_prev)*-1

            # update feedforward weights & biases
            self.W -= f_eta*np.outer(E, self.f_input)
            self.b -= f_eta*E

            if troubleshooting:
                pdb.set_trace()
        else:
            loss = 0

            # self.Z -= f_eta*((0.5 - self.u)-(self.u/self.q).ger(self.r))

        return loss

class outputLayer:
    def __init__(self, net, layer_num, size, f_input_size, W_std):
        self.net          = net
        self.layer_num    = layer_num
        self.size         = size
        self.f_input_size = f_input_size

        # initialize feedforward weights & biases
        self.W = np.random.normal(0, W_std, size=(self.size, self.f_input_size))
        self.b = np.ones(self.size)

        self.f_input    = np.zeros(self.f_input_size)
        self.event_rate = np.zeros(self.size)
        self.burst_prob = np.zeros(self.size)
        self.burst_rate = np.zeros(self.size)
        self.u          = np.zeros(self.size)

        self.baseline_event_rate = 0.001

    def forward(self, f_input):
        self.f_input = f_input

        self.event_rate = np.dot(self.W, self.f_input) + self.b

        self.event_rate[self.event_rate < self.baseline_event_rate] = self.baseline_event_rate

    def backward(self, t, f_eta, update_weights=False):
        if t is not None:
            self.u = 4*t.copy() - 2

            
            # self.event_rate = 0.8*t + 0.2*self.event_rate_prev
        else:
            self.u = np.ones(self.size)

        self.burst_prob_prev = self.burst_prob.copy()
        self.burst_prob = expit(self.u)

        self.burst_rate_prev = self.burst_rate.copy()
        self.burst_rate = self.burst_prob*self.event_rate

        if update_weights:
            # calculate loss
            loss = np.mean((self.burst_rate - self.burst_rate_prev)**2)

            E = (self.burst_rate - self.event_rate)*-(self.event_rate > self.baseline_event_rate).astype(int)

            # update feedforward weights & biases
            self.W -= f_eta*np.outer(E, self.f_input)
            self.b -= f_eta*E

            if troubleshooting:
                pdb.set_trace()
        else:
            loss = 0

        return loss