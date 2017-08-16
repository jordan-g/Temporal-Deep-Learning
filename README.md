# Temporal Deep Learning

![Network.png](Network.png)

## Dependencies

- `torch`
- `numpy`
- `matplotlib`

## Description

The network here is trained on learning a temporal sequence of outputs given a temporal sequence of inputs. Both the input to the network and the target output are randomly generated sinusoidal curves. During one epoch, the sinusoidal input is presented to the network over time, while the target output is presented to the network only a fraction of the time (currently set to 5%):

![InputTarget.png](InputTarget.png)

Each time step, an exponential smoothing kernel is applied to the input as it arrives at the somatic compartment of a hidden layer neuron <img alt="$i$" src="svgs/77a3b857d53fb44e33b53e4c8b68351a.png?invert_in_darkmode" align=middle width="5.663295000000005pt" height="21.683310000000006pt"/>, where it is multiplied by the feedforward weights <img alt="$W^0_i$" src="svgs/c429624422db8204fe8d584d5e9f09a1.png?invert_in_darkmode" align=middle width="24.360930000000003pt" height="26.76201000000001pt"/> and added to a bias term <img alt="$b^0_i$" src="svgs/a4d801a7bcc5e60b7625a6f958bfeb0c.png?invert_in_darkmode" align=middle width="13.607385000000003pt" height="26.76201000000001pt"/>. This is the somatic voltage of the neuron, <img alt="$y^0_i$" src="svgs/f2af248053c53d5e4aded4609f4db6d0.png?invert_in_darkmode" align=middle width="15.201780000000003pt" height="26.76201000000001pt"/>. Then, the event rate of a neuron (the number of spiking events per timestep – either a single spike or a burst of spikes) <img alt="$\lambda^0_i$" src="svgs/70c1a65cc59e7170bd9e167cbb90d4b0.png?invert_in_darkmode" align=middle width="16.141620000000003pt" height="26.76201000000001pt"/> is given by the standard sigmoid function applied to the somatic voltage:

<p align="center"><img alt="$$&#10;\lambda^0_i = \sigma(y^0_i) = \frac{1}{1 + e^{-y^0_i}}&#10;$$" src="svgs/9e8679762c596281d50c1a51b8db4ac2.png?invert_in_darkmode" align=middle width="162.1191pt" height="36.055305pt"/></p>

All hidden layer neurons are fully connected to all output layer neurons. An output layer neuron <img alt="$j$" src="svgs/36b5afebdba34564d884d347484ac0c7.png?invert_in_darkmode" align=middle width="7.710483000000004pt" height="21.683310000000006pt"/> calculates its somatic voltage by multiplying the exponentially smoothed presynaptic event rates by the feedforward weights <img alt="$W^1_j$" src="svgs/b30274615096949c0ec323c945eabb8d.png?invert_in_darkmode" align=middle width="24.360930000000003pt" height="26.76201000000001pt"/> and adding a bias term <img alt="$b^1_j$" src="svgs/8dc83b5c86c749277660452c65266788.png?invert_in_darkmode" align=middle width="13.607385000000003pt" height="26.76201000000001pt"/>. The event rate is calculated using the sigmoid function as above. The event rates of the output neurons are considered to be the output of the network.

All output layer neurons are also fully connected to the apical compartments of the hidden layer neurons – these are the feedback connections (although learning still works with only partial feedback connections). For hidden layer neuron <img alt="$i$" src="svgs/77a3b857d53fb44e33b53e4c8b68351a.png?invert_in_darkmode" align=middle width="5.663295000000005pt" height="21.683310000000006pt"/>, the exponentially smoothed event rates of the output layer neurons are multiplied by the feedback weights <img alt="$Y_i$" src="svgs/c503cd3cc90b9dc8646daa73c42365ae.png?invert_in_darkmode" align=middle width="14.194290000000004pt" height="22.46574pt"/> – this becomes the apical voltage <img alt="$g^0_i$" src="svgs/bdc5fb8d47ccffcdde9d3ec1922a5517.png?invert_in_darkmode" align=middle width="14.982990000000003pt" height="26.76201000000001pt"/>. The burst probability (the probability of an event being a burst), <img alt="$\rho^0_i$" src="svgs/a88d749bd44de165525f4786ee7f9fce.png?invert_in_darkmode" align=middle width="15.051465000000004pt" height="26.76201000000001pt"/>, is calculated using the apical voltage, again by applying a standard sigmoid function:

<p align="center"><img alt="$$&#10;\rho^0_i = \sigma(g^0_i) = \frac{1}{1 + e^{-g^0_i}}&#10;$$" src="svgs/f723ab0a26fb7b223972de73b5522f43.png?invert_in_darkmode" align=middle width="160.55638499999998pt" height="36.055305pt"/></p>

The burst rate (the number of burst events per timestep), <img alt="$\psi^0_i$" src="svgs/a0e116627e1b7e56d2a311dbb6b99fba.png?invert_in_darkmode" align=middle width="17.850195000000003pt" height="26.76201000000001pt"/>, is then simply the product of the burst probability and the event rate:

<p align="center"><img alt="$$&#10;\psi^0_i = \rho^0_i \lambda^0_i&#10;$$" src="svgs/1f14ed810fe5e9d137662a53deaa6ff9.png?invert_in_darkmode" align=middle width="72.60462pt" height="18.266655pt"/></p>