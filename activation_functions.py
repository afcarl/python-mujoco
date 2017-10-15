import numpy as np

def activation(input_layer, function, derivative):
    if function == 'sigmoid':
        if derivative:
            return np.exp(-input_layer) / ((1.0 + np.exp(-input_layer)) ** 2.0)
        else:
            return 1.0 / (1.0 + np.exp(-input_layer))

    if function == 'rectify':
        if derivative:
            return np.where(input_layer > 0, 2., 0.)
        else:
            return np.where(input_layer > 0, 2. * input_layer, 0.)

    if function == 'softplus':
        if derivative:
            return np.exp(input_layer) / (1 + np.exp(input_layer))
        else:
            return np.log(1 + np.exp(input_layer))