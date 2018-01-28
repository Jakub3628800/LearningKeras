# Mnist in Keras
- Input layer = 784
- hidden1 = 64
- Output layer = 10

test accuracy = 97 %

I forgot to normalize the data and could not get good results at first. After normalization Its pretty easy, and with almost any shape one can get above 90% accuracy.

# Gradient descent:
Gradient descent is an algorithm for finding a minimum of a function. In neural networks, we are looking to minimize loss function ( 0.1 in the picture ) and therefore optimize the predictive abilities of the neural network.
In each step when we propagate trough neural network, we calculate loss function, but also a gradient of a loss function. Because we have the gradient, we know “which way to go” to decrease the loss function ( The opposite way of a gradient ).

We take a step towards a smaller loss value by adjusting the weights. This process is propagated back trough the neural network, so that all the weights are adjusted and the next forward propagation yields smaller loss value.
Gradient descent does steps towards a minimum. We cannot however guarantee that we are going to a global minimum. But ya know, local minimum is good enough :-)
