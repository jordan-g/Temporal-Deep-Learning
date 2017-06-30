import network_torch

# set training parameters
f_etas = (0.5, 0.1, 0.05)
f_etas = [ f_etas[i] for i in range(len(f_etas)) ]
b_etas = None
n_epochs = 21

# create the network -- this will also load the MNIST dataset files
net = network_torch.Network(n=(200, 100, network_torch.n_out))

# train the network
net.train(f_etas, b_etas, n_epochs, save_simulation=True, simulations_folder="Simulations", folder_name="Example Simulation", overwrite=True)