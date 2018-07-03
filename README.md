# Temporal Deep Learning (Burstprop)

### Note:
In order for this code to run, there must exist a directory named `mnist` containing the [original MNIST files](http://yann.lecun.com/exdb/mnist/).

### Files:
- **`train.py` & `network.py`:** A simplified version of the network without any time dynamics, instead using a simple forward pass followed by a backward pass for each training example. Uses online training.
- **`train_time.py` & `network_time.py`:** A network with time dynamics, where each training example is presented for a certain number of timesteps, and a target is randomly presented. Uses online training.
- **`mnist.py` & `utils.py`:** Helper files containing functions that load the MNIST data.
- **`train.sh` & train_multi.sh`:** Bash scripts used on [Cedar](https://docs.computecanada.ca/wiki/Cedar) to run either individual jobs or many jobs at once to perform a grid search.

### To-do:
- [ ] Make a version of the time dynamics network that works with batch or mini-batch training.
- [ ] Try to get training to work with weights that are continuously updating.
- [ ] Show a significant performance improvement with two hidden layers vs. one hidden layer.
- [ ] Get training working with three hidden layers.
