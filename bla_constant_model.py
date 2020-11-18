# bla_constant_model.py
# a simple model that discards the input features, and predicts the mean of the training labels for each beam

import numpy as np

class BlaConstantModel:
    means = None

    def __init__(self):
        self.means = np.zeros(6)

    def train(self, X, ys):
        """
        :param X: training data
        :param y: list of bla outputs (np.arrays)
        :return: None
        """
        for i in range(len(ys)):
            self.means[i] = np.mean(ys[i])

    def predict(self, X, beam):
        """
        :param X: the data to predict on (discarded)
        :param beam: the beam to be predicted (1-6), but 0 indexed so (0-5)
        :return:
        """
        return np.repeat(self.means[beam], X.shape[0]).reshape((X.shape[0], 1))

    def get_model(self):
        # returns the model so you can get coefficients and intercepts and stuff
        return self.model