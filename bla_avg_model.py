# bla_avg_model.py
# a simple model that averages all beam outputs and then fits a linear model to it

from sklearn.linear_model import LinearRegression
import numpy as np

class BlaAvgModel:
    model = None

    def __init__(self):
        self.model = LinearRegression()

    def train(self, X, ys):
        """
        :param X: training data
        :param y: list of bla outputs (np.arrays)
        :return: None
        """
        # avg y's
        print(len(ys))
        avg = np.zeros_like(ys[0])

        for y in ys:
            avg = avg + y
        avg = avg / len(ys)

        self.model.fit(X, avg)

    def predict(self, X):
        """
        :param X: the data to predict on
        :return:
        """
        return self.model.predict(X)

    def get_model(self):
        # returns the model so you can get coefficients and intercepts and stuff
        return self.model