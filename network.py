import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy import interpolate

# ---------------------------------------------------------------
"""                 Simulation parameters                     """
# ---------------------------------------------------------------

sequence_length       = 2000 # length of the input sequence to be repeated
n_spikes_per_burst    = 10   # number of spikes in each burst
teach_prob            = 0.05 # probability of a teaching signal being provided
n_classes             = 10   # number of training classes to train the network on
n_sequences_per_class = 10   # number of sequences per training class

n_sequences           = n_classes*n_sequences_per_class # total number of sequences per epoch

use_sparse_feedback  = False # zero out a proportion of the feedback weights
sparse_feedback_prop = 0.5   # proportion of feedback weights to set to 0

# uniform distribution ranges for initial weights
W_range = 0.1
Y_range = 1.0

# ---------------------------------------------------------------
"""                     Functions                             """
# ---------------------------------------------------------------

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / N 

def create_data(n_in, n_out):
    '''
    Generate input & target data using sine & cosine functions.
    '''

    x_set = np.zeros((n_sequences + n_classes, n_in, sequence_length)).astype(np.float32)
    t_set = np.zeros((n_sequences + n_classes, n_out, sequence_length)).astype(np.float32)

    x_class_set = np.zeros((n_classes, n_in, sequence_length)).astype(np.float32)
    t_class_set = np.zeros((n_classes, n_out, sequence_length)).astype(np.float32)

    x_class_amplitudes  = np.zeros((n_classes, n_in))
    x_class_frequencies = np.zeros((n_classes, n_in))
    x_class_phases      = np.zeros((n_classes, n_in))

    t_xor_x_pairs = np.zeros((n_classes, n_out, 2)).astype(int)

    x = np.arange(sequence_length)

    for k in range(n_classes):
        for i in range(n_in):
            x_class_frequencies[k, i] = np.random.uniform(0.005, 0.03)
            x_class_phases[k, i]      = np.random.uniform(0, 1)
            x_class_amplitudes[k, i]  = np.random.uniform(0.2, 0.4)

            x_class_set[k, i] = x_class_amplitudes[k, i]*np.cos(x_class_frequencies[k, i]*x + x_class_phases[k, i]) + 0.5

        for i in range(n_out):
            a = np.random.randint(0, n_in)

            b = np.random.randint(0, n_in)
            while a == b:
                b = np.random.randint(0, n_in)

            t_class_set[k, i] = np.zeros(x.shape)
            t_class_set[k, i, np.logical_or(x_class_set[k, a] > 0.5, x_class_set[k, b] > 0.5)] = 0.8
            t_class_set[k, i, np.logical_and(x_class_set[k, a] > 0.5, x_class_set[k, b] > 0.5)] = 0.2
            t_class_set[k, i, np.logical_and(x_class_set[k, a] <= 0.5, x_class_set[k, b] <= 0.5)] = 0.2

            ind = np.arange(0, sequence_length, 5)
            tck = interpolate.splrep(ind, t_class_set[k, i, ind], s=10)
            ynew = interpolate.splev(np.arange(sequence_length), tck, der=0)
            t_class_set[k, i] = ynew

            t_xor_x_pairs[k, i] = (a, b)

        for m in range(n_sequences_per_class):
            index = int(k*n_sequences_per_class + m)
            for i in range(n_in):
                amplitude_pertubation = np.random.uniform(0.05, 0.2)*np.cos(np.random.uniform(0.005, 0.05)*x + np.random.uniform(0, 1)) + 1
                shift_pertubation = np.random.uniform(0.01, 0.05)*np.cos(np.random.uniform(0.005, 0.05)*x + np.random.uniform(0, 1))
                x_set[index, i] = amplitude_pertubation*x_class_set[k, i] + shift_pertubation

            for i in range(n_out):
                amplitude_pertubation = np.random.uniform(0.05, 0.2)*np.cos(np.random.uniform(0.005, 0.05)*x + np.random.uniform(0, 1)) + 1
                shift_pertubation = np.random.uniform(0.01, 0.05)*np.cos(np.random.uniform(0.005, 0.05)*x + np.random.uniform(0, 1))
                t_set[index, i] = amplitude_pertubation*t_class_set[k, i] + shift_pertubation

        index = int(n_classes*n_sequences_per_class + k)
        for i in range(n_in):
            amplitude_pertubation = np.random.uniform(0.05, 0.2)*np.cos(np.random.uniform(0.005, 0.05)*x + np.random.uniform(0, 1)) + 1
            shift_pertubation = np.random.uniform(0.01, 0.05)*np.cos(np.random.uniform(0.005, 0.05)*x + np.random.uniform(0, 1))
            x_set[index, i] = amplitude_pertubation*x_class_set[k, i] + shift_pertubation

        for i in range(n_out):
            amplitude_pertubation = np.random.uniform(0.05, 0.2)*np.cos(np.random.uniform(0.005, 0.05)*x + np.random.uniform(0, 1)) + 1
            shift_pertubation = np.random.uniform(0.01, 0.05)*np.cos(np.random.uniform(0.005, 0.05)*x + np.random.uniform(0, 1))
            t_set[index, i] = amplitude_pertubation*t_class_set[k, i] + shift_pertubation

    np.save("x_class_set.npy", x_class_set)
    np.save("t_class_set.npy", t_class_set)
    np.save("t_xor_x_pairs.npy", t_xor_x_pairs)
    np.save("x_set.npy", x_set)
    np.save("t_set.npy", t_set)

    for k in range(n_classes):
        plt.figure(figsize=(15, 6))
        plt.plot(np.arange(sequence_length), x_class_set[k, 0], "#57E964", lw=2, label="Input Unit 0")
        plt.plot(np.arange(sequence_length), x_class_set[k, 1], "#4FF2FF", lw=2, label="Input Unit 1")
        plt.plot(np.arange(sequence_length), t_class_set[k, 0], "#FF87E4", lw=2, label="Output Unit 0")
        plt.legend()
        plt.title("Class 0 Sequence Curves")
        plt.xlabel("Timestep")
        plt.ylabel("Activity")
        plt.savefig("inputs_targets_{}.png".format(k))
        plt.savefig("inputs_targets_{}.svg".format(k))
        plt.close()

    colors = ["#FF6666", "#41BFFF", "#57E964", "#FF87E4", "#4FF2FF", "#FFD061", "orange", "purple", "gray", "brown"]

    plt.figure(figsize=(15, 6))
    for k in range(3):
        plt.plot(np.arange(sequence_length), x_class_set[k, 0], lw=2, c=colors[k], label="Class {}".format(k))
        for i in range(n_sequences_per_class):
            plt.plot(np.arange(sequence_length), x_set[k*n_sequences_per_class + i, 0], lw=1, c=colors[k], linestyle='--', alpha=0.5)
    plt.legend()
    plt.title("Input 0 Training Sequences")
    plt.xlabel("Timestep")
    plt.ylabel("Activity")
    plt.savefig("input_unit_0_classes.png")
    plt.savefig("input_unit_0_classes.svg")

    plt.figure(figsize=(15, 6))
    for k in range(3):
        plt.plot(np.arange(sequence_length), t_class_set[k, 0], lw=2, c=colors[k+3], label="Class {}".format(k))
        for i in range(n_sequences_per_class):
            plt.plot(np.arange(sequence_length), t_set[k*n_sequences_per_class + i, 0], lw=1, c=colors[k+3], linestyle='--', alpha=0.5)
    plt.legend()
    plt.title("Output 0 Training Sequences")
    plt.xlabel("Timestep")
    plt.ylabel("Activity")
    plt.savefig("output_unit_0_classes.png")
    plt.savefig("output_unit_0_classes.svg")

    plt.figure(figsize=(15, 6))
    t_xor_x_pair = t_xor_x_pairs[0, 0]
    plt.plot(np.arange(sequence_length), x_class_set[0, t_xor_x_pair[0]], lw=2, c=colors[1], label="Class {}, Input {}".format(0, t_xor_x_pair[0]))
    for i in range(n_sequences_per_class):
        plt.plot(np.arange(sequence_length), x_set[i, t_xor_x_pair[0]], lw=1, c=colors[1], linestyle='--', alpha=0.5)
    plt.plot(np.arange(sequence_length), x_class_set[0, t_xor_x_pair[1]], lw=2, c=colors[2], label="Class {}, Input {}".format(0, t_xor_x_pair[1]))
    for i in range(n_sequences_per_class):
        plt.plot(np.arange(sequence_length), x_set[i, t_xor_x_pair[1]], lw=1, c=colors[2], linestyle='--', alpha=0.5)
    plt.plot(np.arange(sequence_length), t_class_set[0, 0], lw=2, c=colors[3], label="Class {}, Output {}".format(0, 0))
    for i in range(n_sequences_per_class):
        plt.plot(np.arange(sequence_length), t_set[i, 0], lw=1, c=colors[3], linestyle='--', alpha=0.5)

    plt.legend()
    plt.title("Class 0 Training Sequences")
    plt.xlabel("Timestep")
    plt.ylabel("Activity")
    plt.savefig("class_0_seqs.png")
    plt.savefig("class_0_seqs.svg")

    return torch.from_numpy(x_set), torch.from_numpy(t_set)

def load_data():
    x_set = np.load("x_set.npy")
    t_set = np.load("t_set.npy")

    return torch.from_numpy(x_set), torch.from_numpy(t_set)

def get_x(k, n):
    return x_set[k, :, n].unsqueeze_(1)

def get_t(k, n):
    return t_set[k, :, n].unsqueeze_(1)

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
            raise
            self.x_set, self.t_set = load_data()
            print("Loaded training data.")
        except:
            print("Creating training data.")
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

        if self.M == 1:
            update_b_weights = 0
            generate_activity = False

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
            colors = ["red", "blue", "green", "cyan", "orange", "purple", "brown", "magenta"]

            if len(colors) < self.n[-1]:
                raise Exception("Number of output neurons exceeds the number of defined colors for plotting.")

            # create the figure
            self.figure = plt.figure(figsize=(15, 8), facecolor='white')

            # create top axis
            self.animation_axis_top = plt.Axes(self.figure, [0.07, 0.57, 0.86, 0.36])
            self.figure.add_axes(self.animation_axis_top)
            self.target_lines_top = [ self.animation_axis_top.plot([], [], color=colors[i], lw=1, label='Unit {} Target'.format(i+1))[0] for i in range(self.n[-1]) ]
            # self.input_lines_top = [ self.animation_axis_top.plot([], [], color=colors[i+3], lw=2, alpha=0.2, label='Input {}'.format(i+1))[0] for i in range(2) ]
            self.output_lines_top = [ self.animation_axis_top.plot([], [], color=colors[i], lw=1, linestyle='--', alpha=0.5, label='Unit {} Activity'.format(i+1))[0] for i in range(self.n[-1]) ]
            self.animation_axis_top.set_title("Start of Training")
            self.animation_axis_top.legend()

            # create bottom axis
            self.animation_axis_bottom = plt.Axes(self.figure, [0.07, 0.07, 0.86, 0.36])
            self.figure.add_axes(self.animation_axis_bottom)
            self.target_lines_bottom = [ self.animation_axis_bottom.plot([], [], color=colors[i], lw=1, label='Unit {} Target'.format(i+1))[0] for i in range(self.n[-1]) ]
            # self.input_lines_bottom = [ self.animation_axis_bottom.plot([], [], color=colors[i+3], lw=2, alpha=0.2, label='Input {}'.format(i+1))[0] for i in range(2) ]
            self.output_lines_bottom = [ self.animation_axis_bottom.plot([], [], color=colors[i], lw=1, linestyle='--', alpha=0.5, label='Unit {} Activity'.format(i+1))[0] for i in range(self.n[-1]) ]
            self.animation_axis_bottom.set_title("After Training")
            self.animation_axis_bottom.set_xlabel("Timestep")

        # initialize counter for number of time steps at which no target is present
        no_t_count = 0

        # initialize array to hold average loss over each 100 time steps
        # and a counter to keep track of where we are in the avg_training_losses array
        avg_training_losses = np.zeros((self.M, int(n_epochs*sequence_length*n_sequences/100.0)))
        counter = 0

        test_errors = np.zeros(n_classes)

        # initialize arrays to hold targets and outputs over time
        self.targets = np.zeros((n_epochs*sequence_length*n_sequences, self.n[-1]))
        self.outputs = np.zeros((n_epochs*sequence_length*n_sequences, self.n[-1]))
        self.target_times = []

        # initialize target
        t = None

        seq_nums = np.arange(n_sequences)

        # train the network
        for l in range(n_epochs):
            np.random.shuffle(seq_nums)
            for k in range(n_sequences):
                seq_num = seq_nums[k]
                for time in range(sequence_length):
                    # set previous target
                    if t is not None:
                        prev_t = t.clone()
                    else:
                        prev_t = None

                    # get input & target for this time step
                    self.x = self.x_set[seq_num, :, time].unsqueeze_(1)
                    self.t = self.t_set[seq_num, :, time].unsqueeze_(1)

                    if np.random.uniform(0, 1) >= 1 - teach_prob:
                        no_t = False
                        t    = self.t
                        self.target_times.append((l*n_sequences + k)*sequence_length + time)
                    else:
                        no_t = True
                        t    = None
                        no_t_count += 1

                    update_b_weights  = self.update_b_weights
                    update_f_weights  = True
                    generate_activity = False

                    # simulate network activity for this time step
                    self.out(self.x, t, prev_t, time, generate_activity=generate_activity, update_b_weights=update_b_weights, update_f_weights=update_f_weights)

                    # add the loss to average loss, only if a target was present
                    if not no_t:
                        for m in range(self.M):
                            avg_training_losses[m, counter] += float(self.l[m].loss)

                    # record targets & outputs for this time step
                    self.targets[(l*n_sequences + k)*sequence_length + time] = self.t.numpy()[:, 0]
                    self.outputs[(l*n_sequences + k)*sequence_length + time] = self.l[-1].event_rate.numpy()[:, 0]

                    if (time+1) % 100 == 0:
                        # compute average loss over the last 100 time steps
                        # minus those where a target wasn't present
                        if 100 - no_t_count > 0:
                            avg_training_losses[:, counter] /= (100 - no_t_count)
                            if self.M > 1:
                                print("Epoch {:>3d}, example {:>3d}, t={:>4d}. Example {}. Avg. output loss: {:.10f}. Avg. last hidden loss: {:.10f}.".format(l+1, k+1, time+1, seq_num, avg_training_losses[-1, counter], avg_training_losses[-2, counter]))
                            else:
                                print("Epoch {:>3d}, example {:>3d}, t={:>4d}. Example {}. Avg. output loss: {:.10f}.".format(l+1, k+1, time+1, seq_num, avg_training_losses[-1, counter]))
                        else:
                            if self.M > 1:
                                print("Epoch {:>3d}, example {:>3d}, t={:>4d}. Example {}. Avg. output loss: {}. Avg. last hidden loss: {}.".format(l+1, k+1, time+1, seq_num, "_"*12, "_"*12))
                            else:
                                print("Epoch {:>3d}, example {:>3d}, t={:>4d}. Example {}. Avg. output loss: {}.".format(l+1, k+1, time+1, seq_num, "_"*12))
                            
                        no_t_count = 0
                        counter   += 1

        if plot_activity:
            x_range = np.arange(sequence_length)

            for l in range(self.n[-1]):
                self.target_lines_top[l].set_data(x_range, self.targets[:sequence_length, l])
                self.output_lines_top[l].set_data(x_range, self.outputs[:sequence_length, l])

                self.target_lines_bottom[l].set_data(x_range, self.targets[(n_epochs*n_sequences - 1)*sequence_length:, l])
                self.output_lines_bottom[l].set_data(x_range, self.outputs[(n_epochs*n_sequences - 1)*sequence_length:, l])

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

        # test the network on the test set
        seq_nums = np.arange(n_classes) + n_sequences
        np.random.shuffle(seq_nums)

        for k in range(n_classes):
            seq_num = seq_nums[k]
            for time in range(sequence_length):
                # get input & target for this time step
                self.x = self.x_set[seq_num, :, time].unsqueeze_(1)
                self.t = self.t_set[seq_num, :, time].unsqueeze_(1)

                update_b_weights  = False
                update_f_weights  = False
                generate_activity = False

                # simulate network activity for this time step
                self.out(self.x, None, None, time, generate_activity=generate_activity, update_b_weights=update_b_weights, update_f_weights=update_f_weights)

                # add the loss to average loss, only if a target was present
                if not no_t:
                    for m in range(self.M):
                        avg_training_losses[m, counter] += float(self.l[m].loss)

                test_errors[k] += np.mean(np.abs(self.t.numpy()[:, 0] - self.l[-1].event_rate.numpy()[:, 0]))

            test_errors[k] /= sequence_length

        if generate_activity:
            diff = np.zeros((n_classes, int(sequence_length/2.0)))

            seq_nums = np.arange(n_classes) + n_sequences
            np.random.shuffle(seq_nums)

            targets = np.zeros((n_classes, sequence_length, self.n[-1]))
            generated_outputs = np.zeros((n_classes, sequence_length, self.n[-1]))

            for k in range(n_classes):
                seq_num = seq_nums[k]
                for time in range(sequence_length):
                    # get input & target for this time step
                    self.x = self.x_set[seq_num, :, time].unsqueeze_(1)
                    self.t = self.t_set[seq_num, :, time].unsqueeze_(1)

                    update_b_weights  = False
                    update_f_weights  = False

                    generate_activity = time >= int(sequence_length/2)

                    # simulate network activity for this time step
                    self.out(self.x, None, None, time, generate_activity=generate_activity, update_b_weights=update_b_weights, update_f_weights=update_f_weights)

                    # record targets & outputs for this time step
                    targets[k, time] = self.t.numpy()[:, 0]
                    generated_outputs[k, time] = self.l[-1].event_rate.numpy()[:, 0]

                    diff[k, time - int(sequence_length/2)] = np.mean(np.abs(targets[k, time] - generated_outputs[k, time]))

        if not self.generate_activity:
            return avg_training_losses, test_errors
        else:
            return avg_training_losses, test_errors, diff, generated_outputs[:, int(sequence_length/2):], targets[:, int(sequence_length/2):]

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
        # self.f_input = f_input.clone()

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
        self.g      = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))
        self.g_prev = torch.from_numpy(np.zeros((self.size, 1)).astype(np.float32))

    def update_b_input(self, b_input, b_eta, generate_activity=False, update_b_weights=False):
        # update previous feedback input
        self.b_input_prev = self.b_input.clone()

        # apply exponential smoothing to feedback input
        self.b_input = (self.b_input + b_input.clone())/2.0
        # self.b_input = b_input.clone()

        self.g_prev = self.g.clone()

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
            E = (self.event_rate - self.burst_rate)*-self.burst_prob*(1.0 - self.burst_prob)

            # update feedback weights
            self.update_Y(b_eta, E)
            self.decay_Y()

        if generate_activity:
            # generate activity using feedback, by setting the event rate to be equal to the burst rate (which is determined by feedback)
            self.event_rate = self.burst_rate.clone()

    def burst(self, f_eta):
        # calculate feedforward loss
        self.loss = torch.mean((self.event_rate_prev + self.burst_prob - self.burst_prob_prev - self.event_rate_prev)**2)

        # calculate error term
        E = (self.burst_prob - self.burst_prob_prev)*-self.event_rate_prev*(1.0 - self.event_rate_prev)
        # E = (self.g - self.g_prev)*-self.event_rate_prev*(1.0 - self.event_rate_prev)

        # update feedforward weights
        self.update_W(f_eta, E)
        self.decay_W()

    def update_Y(self, b_eta, E):
        # update feedback weights
        self.delta_Y = E.mm(self.b_input.t())
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
            # self.event_rate = (self.event_rate + self.target.clone())/2.0
            self.event_rate = self.target.clone()

        # update spike rate
        self.spike_rate = self.event_rate