import numpy as np
from helperFunctions import *


class LinearRegModel:
    def __init__(self):
        self.trained = False
        self.parameters = None
        self.costsOverTime = None
    

    def train(self, x, y, learn_rate = 0.01, n_itter = 1000, Lambda = 0):

        X = addBiases(x)
        m, n = np.shape(X)
        self.costsOverTime = []
        self.parameters = np.random.random_sample(n)

        # Gradient Descent
        for i in range(n_itter):
            y_hat = np.dot(X, self.parameters)
            grad = (1 / m) * np.dot((y_hat - y), X) + (Lambda / m) * np.hstack((0, self.parameters[1:]))

            self.parameters = self.parameters - learn_rate * grad
            self.costsOverTime.append(self.cost(y, y_hat, m, Lambda))

        self.trained = True


    def cost(self, y, y_hat, m, Lambda):
        return (1 / (2 * m)) * sum(np.square((y_hat - y))) + (Lambda / m) * sum(self.parameters[1:] ** 2)


    def predict(self, x):
        if self.trained:
            X = np.hstack((np.ones((np.shape(x)[0], 1)), x))
            return np.dot(X, self.parameters)
        else: 
            print("Please train the model first")
