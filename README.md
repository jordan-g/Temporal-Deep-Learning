# Temporal Deep Learning

![Network.png](Network.png)

## Dependencies

- `torch`
- `numpy`
- `matplotlib`

## Description

The network here is trained on learning a temporal sequence of outputs given a temporal sequence of inputs. Both the input to the network and the target output are randomly generated sinusoidal curves. During one epoch, the sinusoidal input is presented to the network over time, while the target output is presented to the network only a fraction of the time (currently set to 5%):

![InputTarget.png](InputTarget.png)

Each time step, a hidden layer neuron <img alt="$i$" src="svgs/77a3b857d53fb44e33b53e4c8b68351a.png?invert_in_darkmode" align=middle width="5.663295000000005pt" height="21.683310000000006pt"/> calculates its somatic voltage, <img alt="$y^0_i$" src="svgs/f2af248053c53d5e4aded4609f4db6d0.png?invert_in_darkmode" align=middle width="15.201780000000003pt" height="26.76201000000001pt"/>, by multiplying the exponentially smoothed presynaptic input, <img alt="$\boldsymbol{\tilde{x}}$" src="svgs/8418098471dcae07d32cc65cafe2e1aa.png?invert_in_darkmode" align=middle width="10.833405000000004pt" height="22.831379999999992pt"/>, with the feedforward weights <img alt="$\boldsymbol{W}^0_i$" src="svgs/45b3ffde5017b95c559024d9fc5a6335.png?invert_in_darkmode" align=middle width="27.14613pt" height="29.260439999999992pt"/> and adding a bias term <img alt="$b^0_i$" src="svgs/a4d801a7bcc5e60b7625a6f958bfeb0c.png?invert_in_darkmode" align=middle width="13.607385000000003pt" height="26.76201000000001pt"/>:

<p align="center"><img alt="$$&#10;y^0_i = \sum_{k} W^0_{ik} \tilde{x}_k + b^0_i&#10;$$" src="svgs/8bb4e39f2f2cb5761a4ee960f3565bfd.png?invert_in_darkmode" align=middle width="143.87076pt" height="37.032104999999994pt"/></p>

Then, the event rate of a neuron (the number of spiking events per timestep – either a single spike or a burst of spikes), <img alt="$\lambda^0_i$" src="svgs/70c1a65cc59e7170bd9e167cbb90d4b0.png?invert_in_darkmode" align=middle width="16.141620000000003pt" height="26.76201000000001pt"/>, is given by the standard sigmoid function applied to the somatic voltage:

<p align="center"><img alt="$$&#10;\lambda^0_i = \sigma(y^0_i) = \frac{1}{1 + e^{-y^0_i}}&#10;$$" src="svgs/9e8679762c596281d50c1a51b8db4ac2.png?invert_in_darkmode" align=middle width="162.1191pt" height="36.055305pt"/></p>

All hidden layer neurons are fully connected to all output layer neurons. An output layer neuron <img alt="$j$" src="svgs/36b5afebdba34564d884d347484ac0c7.png?invert_in_darkmode" align=middle width="7.710483000000004pt" height="21.683310000000006pt"/> calculates its somatic voltage, <img alt="$y^1_i$" src="svgs/ec65b9fbdafb51436b6061d185b389f2.png?invert_in_darkmode" align=middle width="15.201780000000003pt" height="26.76201000000001pt"/>, by multiplying the exponentially smoothed presynaptic event rates, <img alt="$\boldsymbol{\tilde{\lambda}}^0$" src="svgs/8c995a169d3b0be4226118410c7225b7.png?invert_in_darkmode" align=middle width="17.579925000000003pt" height="37.75365pt"/>, with the feedforward weights <img alt="$\boldsymbol{W}^1_j$" src="svgs/6622353aecca39cf1d6a3f9dc415ee96.png?invert_in_darkmode" align=middle width="27.14613pt" height="29.260439999999992pt"/> and adding a bias term <img alt="$b^1_j$" src="svgs/8dc83b5c86c749277660452c65266788.png?invert_in_darkmode" align=middle width="13.607385000000003pt" height="26.76201000000001pt"/>:

<p align="center"><img alt="$$&#10;y^1_j = \sum_{i} W^1_{ji} \tilde{\lambda}^0_i + b^1_j&#10;$$" src="svgs/2e5b7b704399af4a2169110150b1eb11.png?invert_in_darkmode" align=middle width="142.18974pt" height="36.655409999999996pt"/></p>

The event rate is calculated using the sigmoid function as above. The event rates of the output neurons are considered to be the output of the network.

All output layer neurons are also fully connected to the apical compartments of the hidden layer neurons – these are the feedback connections (although learning still works with only partial feedback connections). For hidden layer neuron <img alt="$i$" src="svgs/77a3b857d53fb44e33b53e4c8b68351a.png?invert_in_darkmode" align=middle width="5.663295000000005pt" height="21.683310000000006pt"/>, the exponentially smoothed event rates of the output layer neurons are multiplied by the feedback weights <img alt="$Y_i$" src="svgs/c503cd3cc90b9dc8646daa73c42365ae.png?invert_in_darkmode" align=middle width="14.194290000000004pt" height="22.46574pt"/> – this becomes the apical voltage <img alt="$g^0_i$" src="svgs/bdc5fb8d47ccffcdde9d3ec1922a5517.png?invert_in_darkmode" align=middle width="14.982990000000003pt" height="26.76201000000001pt"/>. The burst probability (the probability of an event being a burst), <img alt="$\rho^0_i$" src="svgs/a88d749bd44de165525f4786ee7f9fce.png?invert_in_darkmode" align=middle width="15.051465000000004pt" height="26.76201000000001pt"/>, is calculated using the apical voltage, again by applying a standard sigmoid function:

<p align="center"><img alt="$$&#10;\rho^0_i = \sigma(g^0_i) = \frac{1}{1 + e^{-g^0_i}}&#10;$$" src="svgs/f723ab0a26fb7b223972de73b5522f43.png?invert_in_darkmode" align=middle width="160.55638499999998pt" height="36.055305pt"/></p>

The burst rate (the number of burst events per timestep), <img alt="$\psi^0_i$" src="svgs/a0e116627e1b7e56d2a311dbb6b99fba.png?invert_in_darkmode" align=middle width="17.850195000000003pt" height="26.76201000000001pt"/>, is then simply the product of the burst probability and the event rate:

<p align="center"><img alt="$$&#10;\psi^0_i = \rho^0_i \lambda^0_i&#10;$$" src="svgs/1f14ed810fe5e9d137662a53deaa6ff9.png?invert_in_darkmode" align=middle width="72.60462pt" height="18.266655pt"/></p>