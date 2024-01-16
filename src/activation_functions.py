##@package activation_functions
# Activation functions for neural networks.

import math
import numpy as np


##Identity function.
# Range: (-inf, inf)
# @image html https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Activation_identity.svg/120px-Activation_identity.svg.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Activation_identity.svg/120px-Activation_identity.svg.png
# @param x (float): Input value.
# @return float: Identity of input value.
def identity(x):
    return x


##Binary step function.
# Range: {0, 1}
# @image html https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Activation_binary_step.svg/120px-Activation_binary_step.svg.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Activation_binary_step.svg/120px-Activation_binary_step.svg.png
# @param x (float): Input value.
# @return float: Binary step of input value.
def binary_step(x):
    if x < 0:
        return 0
    else:
        return 1


##Sigmoid function.
# Range: (0, 1)
# @image html https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Activation_logistic.svg/120px-Activation_logistic.svg.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Activation_logistic.svg/120px-Activation_logistic.svg.png
# @param x (float): Input value.
# @return float: Sigmoid of input value.
def sigmoid(x):
    x = np.clip(x, -500, 500)  # limit the values of x to prevent overflow
    return 1 / (1 + np.exp(-x))


##Hyperbolic tangent function.
# Range: (-1, 1)
# @image html https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Activation_tanh.svg/120px-Activation_tanh.svg.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Activation_tanh.svg/120px-Activation_tanh.svg.png
# @param x (float): Input value.
# @return float: Hyperbolic tangent of input value.
def tanh(x):
    return math.tanh(x)


##Rectified linear unit function.
# Range: [0, inf)
# @image html https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Activation_rectified_linear.svg/120px-Activation_rectified_linear.svg.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Activation_rectified_linear.svg/120px-Activation_rectified_linear.svg.png
# @param x (float): Input value.
# @return float: Rectified linear unit of input value.
def relu(x):
    if x < 0:
        return 0
    else:
        return x


##Gaussian error linear unit function.
# Range: (-0.17..., inf)
# @image html https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Activation_gelu.png/120px-Activation_gelu.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Activation_gelu.png/120px-Activation_gelu.png
# @param x (float): Input value.
# @return float: Gaussian error linear unit of input value.
def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))


##Softplus function.
# Range: (0, inf)
# @image html https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Activation_softplus.svg/120px-Activation_softplus.svg.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Activation_softplus.svg/120px-Activation_softplus.svg.png
# @param x (float): Input value.
# @return float: Softplus of input value.
def softplus(x):
    x = np.clip(x, -500, 500)  # limit the values of x to prevent overflow
    return math.log(1 + math.exp(x))


##Exponential linear unit function.
# Range: (-alpha, inf)
# @image html https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Activation_elu.svg/120px-Activation_elu.svg.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/Activation_elu.svg/120px-Activation_elu.svg.png
# @param x (float): Input value.
# @param alpha (float): Learnable parameter.
# @return float: Exponential linear unit of input value.
def elu(x, alpha):
    if x < 0:
        return alpha * (math.exp(x) - 1)
    else:
        return x


##Scaled exponential linear unit function.
# Range: (-1.758094282, inf)
# LAMBDA = 1.0507
# ALPHA = 1.67326
# @image html https://upload.wikimedia.org/wikipedia/commons/4/43/Activation_selu.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/4/43/Activation_selu.png
# @param x (float): Input value.
# @return float: Scaled exponential linear unit of input value.
def selu(x):
    LAMBDA = 1.0507
    ALPHA = 1.67326
    if x < 0:
        return LAMBDA * ALPHA * (math.exp(x) - 1)
    else:
        return LAMBDA * x


##Leaky rectified linear unit function.
# Range: (-inf, inf)
# @image html https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Activation_prelu.svg/120px-Activation_prelu.svg.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Activation_prelu.svg/120px-Activation_prelu.svg.png
# @param x (float): Input value.
# @return float: Leaky rectified linear unit of input value.
def leaky_relu(x):
    if x < 0:
        return 0.01 * x
    else:
        return x


##Parametric rectified linear unit function.
# Range: (-inf, inf)
# @image html https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Activation_prelu.svg/120px-Activation_prelu.svg.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Activation_prelu.svg/120px-Activation_prelu.svg.png
# @param x (float): Input value.
# @param alpha (float): Learnable parameter.
# @return float: Parametric rectified linear unit of input value.
def prelu(x, alpha):
    if x < 0:
        return alpha * x
    else:
        return x


##Swish function.
# Range: [-0.278..., inf)
# @image html https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Swish.svg/120px-Swish.svg.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/thumb/1/12/Swish.svg/120px-Swish.svg.png
# @param x (float): Input value.
# @return float: Swish of input value.
def swish(x):
    return x / (1 + math.exp(-x))


##Gaussian function.
# Range: (0, 1]
# @image html https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Activation_gaussian.svg/120px-Activation_gaussian.svg.png
# @image latex https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Activation_gaussian.svg/120px-Activation_gaussian.svg.png
# @param x (float): Input value.
# @return float: Gaussian of input value.
def gaussian(x):
    return math.exp(-(x**2))
