# bla_kernel_lr_model.py
# a model that uses kernelized linear regression to fit a model for each bla output

from sklearn.kernel_ridge import KernelRidge
import numpy as np

class BlaKernelLRModel:
    models = None

    def __init__(self, kernel, degree=None, alpha=1.0):
        self.models = [KernelRidge(alpha=alpha, kernel=kernel, degree=degree) for _ in range(6)]

    def train(self, X, ys):
        """
        :param X: training data
        :param ys: list of bla outputs (np.arrays)
        :return: None
        """
        for i in range(6):
            self.models[i].fit(X, ys[i])

    def predict(self, X, beam):
        """
        :param X: the data to predict on
        :param beam: the index of the beam to predict
        :return:
        """
        return self.models[beam].predict(X).reshape((X.shape[0], 1))

    def get_model(self):
        # returns the model so you can get coefficients and intercepts and stuff
        return self.models
