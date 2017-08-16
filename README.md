# Temporal Deep Learning

![Network.png](Network.png)

## Dependencies

- `torch`
- `numpy`
- `matplotlib`

## Description

The network here is trained on learning a temporal sequence of outputs given a temporal sequence of inputs. Both the input to the network and the target output are randomly generated sinusoidal curves. During one epoch, the sinusoidal input is presented to the network over time, while the target output is presented to the network only a fraction of the time (currently set to 5%):

![InputTarget.png](InputTarget.png)

Each time step, an exponential smoothing kernel is applied to the input as it arrives at the somatic compartment of a hidden layer neuron, where it is multiplied by the feedforward weights <img alt="$W_0$" src="svgs/e0ab31cb9c791e75fc086f61bfb584f8.png?invert_in_darkmode" align=middle width="22.077825pt" height="22.46574pt"/> and added to a bias term <img alt="$b$" src="svgs/4bdc8d9bcfb35e1c9bfb51fc69687dfc.png?invert_in_darkmode" align=middle width="7.054855500000005pt" height="22.831379999999992pt"/>. This is the somatic voltage of the neuron, <img alt="$y$" src="svgs/deceeaf6940a8c7a5a02373728002b0f.png?invert_in_darkmode" align=middle width="8.649300000000004pt" height="14.155350000000013pt"/>. Then, the event rate of a neuron (the number of spiking events per timestep -- either a single spike or a burst of spikes) is given by the standard sigmoid function applied to the 