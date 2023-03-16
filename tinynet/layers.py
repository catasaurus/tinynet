import numpy as np

class Dense:
    def __init__(self, wide, activation, shape, rng_=np.random.default_rng()):
        self.rng_ = rng_
        self.weights = rng_.uniform(low=0., high=1, size=(wide, *shape))
        self.biases = rng_.uniform(low=0., high=1, size=(wide))
        self.activation = activation

    def forward(self, data):
        assert data.shape == self.weights.shape[1:], "shapes must match"
        return self.activation((self.weights * data)+self.bias)