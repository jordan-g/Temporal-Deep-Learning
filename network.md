The network here is trained on learning a set of temporal sequences of outputs given a set of temporal sequences of inputs. The length of a sequence is 2000 timesteps. The network has 500 input units and 3 output units.

## Training Set

The training set consists of 10 classes of input-output sequences – each class is based on a distinct set of input and target sequences which are perturbed slightly in order to generate different examples from that class. For a given class, we first randomly generate the canonical input & target output sequences which examples from that class will be based on. The canonical sequence for input unit $i$ is given by a randomly generated sinusoidal curve:

\begin{equation}
x_i(t) = A_i \text{cos}(B_i t + C_i) + 0.5
\end{equation}

where the amplitude factor $A_i$, the frequency $B_i$ and the phase $C_i$ are drawn from uniform distributions:

\begin{align}
\begin{split}
A_i &\sim \text{U}(0.2, 0.4) \\
B_i &\sim \text{U}(0.005, 0.03) \\
C_i &\sim \text{U}(0, 1)
\end{split}
\end{align}

The canonical target sequences for the 3 output units are generated in order to introduce some complexity in the functions that need to be learned – specifically, XOR functions. This is done because an XOR function cannot be properly learned without a hidden layer of neurons.

For each output unit $j$, we define the target sequence $\hat{y}^1_j$ as

\begin{align}
\begin{split}
\hat{y}^1_j(t) &= \left\{ \begin{array}{lr}
       0.8 \text{,} \quad \text{ if } x_a(t) > \gamma \text{ or } x_b(t) > \gamma \\
       0.2 \text{,} \quad \text{ if } x_a(t) > \gamma \text{ and } x_b(t) > \gamma \\
       0.2 \text{,} \quad \text{ if } x_a(t) \leq \gamma \text{ and } x_b(t) \leq \gamma
\end{array} \right.
\end{split}
\end{align}

This is the XOR function applied to the thresholded versions of inputs $x_a(t)$ and $x_b(t)$, with a threshold $\gamma$, where $a$ and $b$ are different for each target unit. This creates a square wave, which is finally smoothed by fitting a cubic spline to $\hat{y}^1_j$.

$a$ and $b$ are chosen randomly for each output unit. The threshold $\gamma$ is set to $0.5$.

Thus, the target activity for each of the output units is to be active when either input $x_a$ or $x_b$ is active, but not when both are active, regardless of the activities of the other inputs (see Figure 1).

![Activity sequence of output unit 0 for training class 0, and the activity sequences of the two input units which the target activity of output unit 0 was based on, $a = 417$ and $b = 442$. This shows the XOR function properties of the target activities – the target activity of output unit 0 is high when either input has high activity, but not when both do. \textit{Solid lines:} Canonical sequences for the given class. \textit{Dotted lines:} Training example sequences generated based on the canonical curves.](xor_example.png)

In order to generate training examples from this class, we add random variations to the amplitude and vertical shift of the canonical sequence curves. For input unit $i$ we define:

\begin{align}
\begin{split}
a^x_i (t) &= A^x_i \text{cos}(B^x_i t + C^x_i) + 1 \\
s^x_i (t) &= D^x_i \text{cos}(E^x_i t + F^x_i)
\end{split}
\end{align}

where

\begin{align}
\begin{split}
A^x_i &\sim \text{U}(0.05, 0.2) \\
B^x_i &\sim \text{U}(0.005, 0.05) \\
C^x_i &\sim \text{U}(0, 1) \\
D^x_i &\sim \text{U}(0.01, 0.05) \\
E^x_i &\sim \text{U}(0.005, 0.05) \\
F^x_i &\sim \text{U}(0, 1)
\end{split}
\end{align}

Then, to generate an example sequence, $x_i(t)$ is adjusted so that:

\begin{align}
\begin{split}
x_i(t) &\longrightarrow a^x_i(t) x_i(t) + s^x_i(t)
\end{split}
\end{align}

Similarly, for output unit $j$, we define:

\begin{align}
\begin{split}
a^{\hat{y}^1}_j (t) &= A^{\hat{y}^1}_j \text{cos}(B^{\hat{y}^1}_j t + C^{\hat{y}^1}_j) + 1 \\
s^{\hat{y}^1}_j (t) &= D^{\hat{y}^1}_j \text{cos}(E^{\hat{y}^1}_j t + F^{\hat{y}^1}_j)
\end{split}
\end{align}

where

\begin{align}
\begin{split}
A^{\hat{y}^1}_i &\sim \text{U}(0.05, 0.2) \\
B^{\hat{y}^1}_i &\sim \text{U}(0.005, 0.05) \\
C^{\hat{y}^1}_i &\sim \text{U}(0, 1) \\
D^{\hat{y}^1}_i &\sim \text{U}(0.01, 0.05) \\
E^{\hat{y}^1}_i &\sim \text{U}(0.005, 0.05) \\
F^{\hat{y}^1}_i &\sim \text{U}(0, 1)
\end{split}
\end{align}

and, to generate an example sequence, $\hat{y}^1_i(t)$ is adjusted so that:

\begin{align}
\begin{split}
\hat{y}^1_j(t) &\longrightarrow a^{\hat{y}^1}_j(t) \hat{y}^1_j(t) + s^{\hat{y}^1}_i(t)
\end{split}
\end{align}

Figure 2 shows some of the input & target sequences generated for different classes, and training examples drawn from each class.

![Input & target sequences for different classes, and sample sequences drawn from each class. \textbf{a.} Activity sequences of input unit 0 representing three different classes of training data. \textbf{b.} Activity sequences of output unit 0 representing three different classes of training data. \textit{Solid lines:} Canonical sequences for the given class. \textit{Dotted lines:} Training example sequences generated based on the canonical curves.](training_classes.png)

## Network Structure and Dynamics

![Diagram of the network. \textbf{a.} Network variables and connections. \textbf{b.} Temporal dynamics of the network.](network_diagram.png)

A diagram showing the network structure and dynamics is shown in Figure 3. Assume the network has $l$ inputs, $m$ hidden units and $n$ output units. Unit $j$ in the hidden layer has two compartments: a somatic compartment with voltage $y^0_j$ and an apical dendrite compartment with voltage $g^0_j$. At time $t$, $y^0_j(t)$ is given by:

\begin{align}
\begin{split}
y^0_j(t) &= \sum_{k=1}^l W^0_{jk} \tilde{x}_k (t-1) + b^0_j
\end{split}
\end{align}

where $\bm{W}^0$ is the $m \times l$ matrix of the synaptic weights between the inputs and hidden layer units, $\bm{b}^0$ is a vector containing bias terms for each hidden unit, and $\tilde{\bm{x}}$ is the exponentially smoothed input layer activity:

\begin{align}
\begin{split}
\label{eqn:exponential_smoothing}
\tilde{x}_k(0) &= x_k(0) \\
\tilde{x}_k(t) &= x_k(t) + \tilde{x}_k(t-1)\text{, } t > 0
\end{split}
\end{align}

The hidden unit's \textit{event rate} $\psi^0_j$, defined as the expected number of spike events (either single spikes or bursts) per unit time, is given by a sigmoid applied to the somatic voltage:

\begin{align}
\begin{split}
\psi^0_j(t) &= \sigma(y^0_j(t)) = \frac{1}{1 + e^{-y^0_j(t)}}
\end{split}
\end{align}

This signal is received by units in the output layer. Unit $i$ in the output layer has a somatic compartment with somatic voltage $y^1_i$ given by:

\begin{align}
\begin{split}
y^1_i(t) &= \sum_{k=1}^m W^1_{ik} \tilde{\psi}^0_k(t-1) + b^1_i
\end{split}
\end{align}

where $\bm{W}^1$ are the feedforward synaptic weights between the hidden layer and output layer units, $\bm{b}^0$ are the bias terms for each output unit, and $\tilde{\bm{\psi}}^0$ are the exponentially smoothed event rates of the hidden layer units, computed as in equation \eqref{eqn:exponential_smoothing}. Similarly, the event rate of output unit $i$, $\psi^1_i$, is given by:

\begin{align}
\begin{split}
\psi^1_i(t) &= \sigma(y^1_i(t)) = \frac{1}{1 + e^{-y^1_i(t)}}
\end{split}
\end{align}

Finally, the apical dendrite compartments of hidden layer units receive this signal from the output layer units. The apical voltage $g^0_j$ is given by:

\begin{align}
\begin{split}
g^0_j(t) &= \sum_{k=1}^n Y_{jk} \tilde{\psi}^1_k(t-1)
\end{split}
\end{align}

where $\bm{Y}$ are the feedback synaptic weights between the output layer and hidden layer units, and $\tilde{\bm{\psi}}^1$ are the exponentially smoothed event rates of the output layer units, computed as in equation \eqref{eqn:exponential_smoothing}. The hidden unit's \textit{burst probability} $p^0_j$, defined as the probability that a spike event will be a burst (rather than a single spike), is the given by applying the sigmoid function to the apical voltage:

\begin{align}
\begin{split}
p^0_j(t) &= \sigma(g^0_j(t)) = \frac{1}{1 + e^{-g^0_j(t)}}
\end{split}
\end{align}

## Training

