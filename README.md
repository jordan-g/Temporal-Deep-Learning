# Temporal Deep Learning

![Network.png](Network.png)

## Dependencies

- `torch`
- `numpy`
- `matplotlib`

## Description

The network here is trained on learning a temporal sequence of outputs given a temporal sequence of inputs. Both the input to the network and the target output are randomly generated sinusoidal curves. During one epoch, the sinusoidal input is presented to the network over time, and the target output is presented to the network only a fraction of the time (currently set to 5%):

![InputTarget.png](InputTarget.png)