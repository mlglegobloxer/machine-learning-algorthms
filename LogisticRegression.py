import numpy as np
from helperFunctions import *


class LogisticRegModel:
    def __init__(self):
        self.trained = False
        self.parameters = None
        self.costsOverTime = None
    

    def train(self, x, y, learn_rate = 0.1, n_itter = 1000, Lambda = 0):
        
        X = addBiases(x)
        m, n = np.shape(X)
        self.costsOverTime = []
        self.parameters = np.random.random_sample(n)

        # Gradient Descent
        for i in range(n_itter):
            y_hat = sigmoid(np.matmul(X, self.parameters))
            grad = (1 / m) * np.matmul((y_hat - y), X) + (Lambda / m) * np.hstack((0, self.parameters[1:]))

            self.parameters = self.parameters - learn_rate * grad
            self.costsOverTime.append(self.cost(y, y_hat, m, Lambda))

        self.trained = True


    def cost(self, y, y_hat, m, Lambda):
        return (1 / m) * sum(-y * np.log(y_hat) - (1 - y) * np.log(1 - y_hat)) + (Lambda / (2*m)) * sum(self.parameters[1:] ** 2)


    def predict(self, x, response = True):
        if self.trained:
            X = addBiases(x)
            p = sigmoid(np.matmul(X, self.parameters))

            return np.round(p) if response else p

        else:
            print("Please train the model first")

