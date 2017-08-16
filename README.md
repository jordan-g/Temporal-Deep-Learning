# Temporal Deep Learning

![Network.png](Network.png)

## Dependencies

- `torch`
- `numpy`
- `matplotlib`

## Description

The network here is trained on learning a temporal sequence of outputs given a temporal sequence of inputs. Both the input to the network and the target output are randomly generated sinusoidal curves. During one epoch, the sinusoidal input is presented to the network over time, while the target output is presented to the network only a fraction of the time (currently set to 5%):

![InputTarget.png](InputTarget.png)

Each time step, an exponential smoothing kernel is applied to the input as it arrives at the somatic compartment of a hidden layer neuron <img alt="$i$" src="svgs/77a3b857d53fb44e33b53e4c8b68351a.png?invert_in_darkmode" align=middle width="5.663295000000005pt" height="21.683310000000006pt"/>, where it is multiplied by the feedforward weights <img alt="$W^0_i$" src="svgs/c429624422db8204fe8d584d5e9f09a1.png?invert_in_darkmode" align=middle width="24.360930000000003pt" height="26.76201000000001pt"/> and added to a bias term <img alt="$b_i$" src="svgs/d3aa71141bc89a24937c86ec1d350a7c.png?invert_in_darkmode" align=middle width="11.705760000000003pt" height="22.831379999999992pt"/>. This is the somatic voltage of the neuron, <img alt="$y^0_i$" src="svgs/f2af248053c53d5e4aded4609f4db6d0.png?invert_in_darkmode" align=middle width="15.201780000000003pt" height="26.76201000000001pt"/>. Then, the event rate of a neuron (the number of spiking events per timestep -- either a single spike or a burst of spikes) <img alt="$r^0_i$" src="svgs/6295deef22b17c4718f6aee74d9921f7.png?invert_in_darkmode" align=middle width="14.425620000000004pt" height="26.76201000000001pt"/> is given by the standard sigmoid function applied to the somatic voltage:

<p align="center"><img alt="$$&#10;r^0_i = \sigma(y^0_i) = \frac{1}{1 + e^{-y^0_i}}&#10;$$" src="svgs/63dfa67a9bb65bf016a57042a8e02765.png?invert_in_darkmode" align=middle width="160.40293499999999pt" height="36.055305pt"/></p>