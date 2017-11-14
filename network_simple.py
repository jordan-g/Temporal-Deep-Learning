import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy import interpolate

class Network:
    def __init__(self, n, n_in, W_range, Y_range):
        self.n        = n           # layer sizes - eg. (500, 100, 10)
        self.n_layers = len(self.n) # number of layers

        # initialize layers list
        self.layers = []

        if self.n_layers == 1:
            self.layers.append(outputLayer(self, 0, size=self.n[0], f_input_size=n_in, W_range=W_range))
        else:
            for layer_num in range(self.n_layers-1):
                if layer_num == 0:
                    self.layers.append(hiddenLayer(self, layer_num, size=self.n[layer_num], f_input_size=n_in, b_input_size=self.n[-1], W_range=W_range, Y_range=Y_range))
                else:
                    self.layers.append(hiddenLayer(self, layer_num, size=self.n[layer_num], f_input_size=self.n[layer_num-1], b_input_size=self.n[-1], W_range=W_range, Y_range=Y_range))
            self.layers.append(outputLayer(self, self.n_layers-1, size=self.n[-1], f_input_size=self.n[-2], W_range=W_range))

    def forward(self, x):
        for layer_num in range(self.n_layers-1):
            if layer_num == 0:
                self.layers[0].forward(x)
            else:
                self.layers[layer_num].forward(self.layers[layer_num-1].event_rate)
        self.layers[-1].forward(self.layers[-2].event_rate)

    def backward(self, x, t, f_etas):
        losses = np.zeros(self.n_layers)
        if self.n_layers == 1:
            losses[0] = self.layers[0].backward(t, f_etas[0])
        else:
            losses[-1] = self.layers[-1].backward(t, f_etas[-1])
            for layer_num in range(self.n_layers-2, -1, -1):
                losses[layer_num] = self.layers[layer_num].backward(self.layers[-1].event_rate, t, f_etas[layer_num])

        return losses

class hiddenLayer:
    def __init__(self, net, layer_num, size, f_input_size, b_input_size, W_range, Y_range):
        self.net          = net
        self.layer_num    = layer_num
        self.size         = size
        self.f_input_size = f_input_size
        self.b_input_size = b_input_size

        # initialize feedforward weights & biases
        self.W = torch.from_numpy(W_range*np.random.uniform(-1, 1, size=(self.size, self.f_input_size)).astype(np.float32))
        self.b = torch.from_numpy(np.zeros(self.size).astype(np.float32))

        # initialize feedback weights
        self.Y = torch.from_numpy(Y_range*np.random.uniform(-1, 1, size=(self.size, self.b_input_size)).astype(np.float32))

    def forward(self, f_input):
        self.f_input = f_input

        self.event_rate = torch.sigmoid(self.W.mv(self.f_input) + self.b)

    def backward(self, b_input, target_input, f_eta):
        # update burst probability
        self.burst_prob   = torch.sigmoid(self.Y.mv(b_input))
        self.burst_prob_t = torch.sigmoid(self.Y.mv(target_input))

        # calculate loss
        loss = torch.mean((self.burst_prob_t - self.burst_prob)**2)

        E = (self.burst_prob_t - self.burst_prob)*-self.event_rate*(1.0 - self.event_rate)

        # update feedforward weights & biases
        self.W -= f_eta*E.ger(self.f_input)
        self.b -= f_eta*E

        return loss

class outputLayer:
    def __init__(self, net, layer_num, size, f_input_size, W_range):
        self.net          = net
        self.layer_num    = layer_num
        self.size         = size
        self.f_input_size = f_input_size

        # initialize feedforward weights & biases
        self.W = torch.from_numpy(W_range*np.random.uniform(-1, 1, size=(self.size, self.f_input_size)).astype(np.float32))
        self.b = torch.from_numpy(np.zeros(self.size).astype(np.float32))

    def forward(self, f_input):
        self.f_input = f_input

        self.event_rate = torch.sigmoid(self.W.mv(self.f_input) + self.b)

    def backward(self, t, f_eta):
        # calculate loss
        loss = torch.mean((t - self.event_rate)**2)

        E = (t - self.event_rate)*-self.event_rate*(1.0 - self.event_rate)

        # update feedforward weights & biases
        self.W -= f_eta*E.ger(self.f_input)
        self.b -= f_eta*E

        return loss