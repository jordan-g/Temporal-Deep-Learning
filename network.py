import torch
from torch.autograd import Variable
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy import interpolate

class Network:
    def __init__(self, n, n_in, W_range, Y_range, cuda=False):
        self.n        = n           # layer sizes - eg. (500, 100, 10)
        self.n_layers = len(self.n) # number of layers

        # initialize layers list
        self.layers = []

        if self.n_layers == 1:
            self.layers.append(outputLayer(self, 0, size=self.n[0], f_input_size=n_in, W_range=W_range, cuda=cuda))
        else:
            for layer_num in range(self.n_layers-1):
                if layer_num == 0:
                    self.layers.append(hiddenLayer(self, layer_num, size=self.n[layer_num], f_input_size=n_in, b_input_size=self.n[-1], W_range=W_range, Y_range=Y_range, cuda=cuda))
                else:
                    self.layers.append(hiddenLayer(self, layer_num, size=self.n[layer_num], f_input_size=self.n[layer_num-1], b_input_size=self.n[-1], W_range=W_range, Y_range=Y_range, cuda=cuda))
            self.layers.append(outputLayer(self, self.n_layers-1, size=self.n[-1], f_input_size=self.n[-2], W_range=W_range, cuda=cuda))

    def forward(self, x_prev, t):
        if self.n_layers == 1:
            self.layers[0].forward(x_prev, t)
        else:
            for layer_num in range(self.n_layers-1):
                if layer_num == 0:
                    self.layers[0].forward(x_prev)
                else:
                    self.layers[layer_num].forward(self.layers[layer_num-1].event_rate_prev)
            self.layers[-1].forward(self.layers[-2].event_rate_prev, t)

    def backward(self, t, f_etas, update_final_weights=False, update_hidden_weights=False):
        losses = np.zeros(self.n_layers)
        if self.n_layers == 1:
            losses[0] = self.layers[0].backward(t, f_etas[0], update_weights=update_final_weights)
        else:
            losses[-1] = self.layers[-1].backward(t, f_etas[-1], update_weights=update_final_weights)
            for layer_num in range(self.n_layers-2, -1, -1):
                losses[layer_num] = self.layers[layer_num].backward(self.layers[-1].event_rate_prev, f_etas[layer_num], update_weights=update_hidden_weights)

        return losses

class hiddenLayer:
    def __init__(self, net, layer_num, size, f_input_size, b_input_size, W_range, Y_range, cuda=False):
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

        self.f_input    = torch.from_numpy(np.zeros(self.f_input_size).astype(np.float32))
        self.b_input    = torch.from_numpy(np.zeros(self.b_input_size).astype(np.float32))
        self.event_rate = torch.from_numpy(np.zeros(self.size).astype(np.float32))
        self.burst_prob = torch.from_numpy(np.zeros(self.size).astype(np.float32))

        if cuda:
            self.W          = self.W.cuda()
            self.b          = self.b.cuda()
            self.Y          = self.Y.cuda()
            self.f_input    = self.f_input.cuda()
            self.b_input    = self.b_input.cuda()
            self.event_rate = self.event_rate.cuda()
            self.burst_prob = self.burst_prob.cuda()

    def forward(self, f_input):
        # self.f_input = 0.8*f_input + 0.2*self.f_input
        self.f_input = f_input

        self.event_rate_prev = self.event_rate
        self.event_rate = torch.sigmoid(self.W.mv(self.f_input) + self.b)

    def backward(self, b_input, f_eta, update_weights=False):
        # self.b_input = 0.8*b_input + 0.2*self.b_input
        self.b_input = b_input

        # update burst probability
        self.burst_prob_prev = self.burst_prob
        self.burst_prob      = torch.sigmoid(self.Y.mv(self.b_input))

        if update_weights:
            # calculate loss
            loss = torch.mean((self.burst_prob - self.burst_prob_prev)**2)

            E = (self.burst_prob - self.burst_prob_prev)*-self.event_rate*(1.0 - self.event_rate)

            # update feedforward weights & biases
            self.W -= f_eta*E.ger(self.f_input)
            self.b -= f_eta*E
        else:
            loss = 0

        return loss

class outputLayer:
    def __init__(self, net, layer_num, size, f_input_size, W_range, cuda=False):
        self.net          = net
        self.layer_num    = layer_num
        self.size         = size
        self.f_input_size = f_input_size

        # initialize feedforward weights & biases
        self.W = torch.from_numpy(W_range*np.random.uniform(-1, 1, size=(self.size, self.f_input_size)).astype(np.float32))
        self.b = torch.from_numpy(np.zeros(self.size).astype(np.float32))

        self.f_input    = torch.from_numpy(np.zeros(self.f_input_size).astype(np.float32))
        self.event_rate = torch.from_numpy(np.zeros(self.size).astype(np.float32))

        if cuda:
            self.W          = self.W.cuda()
            self.b          = self.b.cuda()
            self.f_input    = self.f_input.cuda()
            self.event_rate = self.event_rate.cuda()

    def forward(self, f_input, t):
        # self.f_input = 0.8*f_input + 0.2*self.f_input
        self.f_input = f_input

        self.event_rate_prev = self.event_rate

        if t is not None:
            self.event_rate = 0.8*t + 0.2*self.event_rate_prev
        else:
            self.event_rate = torch.sigmoid(self.W.mv(self.f_input) + self.b)

    def backward(self, t, f_eta, update_weights=False):
        if update_weights:
            # calculate loss
            loss = torch.mean((t - self.event_rate_prev)**2)

            E = (self.event_rate - self.event_rate_prev)*-self.event_rate_prev*(1.0 - self.event_rate_prev)

            # update feedforward weights & biases
            self.W -= f_eta*E.ger(self.f_input)
            self.b -= f_eta*E
        else:
            loss = 0

        return loss