# A file contating my implementations of all the ml algortihms
import numpy as np
from helperFunctions import *


class LinearRegression:
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



class LogisticRegression:
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



class KNNClassifier:
    def __init__(self):
        self.X = None
        self.labels = None
        self.trained = False

    
    def train(self, x, y):
        self.X = x
        self.labels = y
        self.trained = True
    

    def predict(self, x, k = 7):
        if self.trained:
            dist = distance(x, self.X)

            knn_to_x = np.argpartition(dist, k)[:, 0:k]  # Indecies of the knn to x in the train set
            y_hat = mode(self.labels[knn_to_x])          # Predict the knn's modal class
            return np.array(y_hat)
        
        else:
            print("Please first train the model")



class KMeansClustering:
    def __init__(self):
        self.trained = False
        self.centroids = None
        self.costsOverTime = None


    def train(self, x, k, n_itter = 20):
        m, n = np.shape(x)
        self.costsOverTime = []

        # Initilise centroids as random elements of x
        self.centroids = x[np.random.choice(m, k, replace = False), :]
        
        for i in range(n_itter):
            closest_centroids = self.closestCentroids(x)

            # Assign the new centroids as the mean positon of x's assigned to it
            for j in range(k):
                self.centroids[j] = np.mean(x[np.where(closest_centroids == j), :], 1)

            # Record the new centroid choice's cost
            self.costsOverTime.append(self.cost(x))

        self.trained = True


    def cost(self, x):
        closest_centroids = self.closestCentroids(x)

        # Find square distance from x and its corresponding centroid
        sq_dist_xs_centriods = np.sum(np.square(x - self.centroids[[int(i) for i in closest_centroids.tolist()], :]), 1)

        return (1 / len(sq_dist_xs_centriods)) * np.sum(sq_dist_xs_centriods)


    def closestCentroids(self, x):
        closest_centroids = np.array([])

        for entry in x:
            closest_centriod  = np.argmin(np.sum(np.square(entry - self.centroids), 1))
            closest_centroids = np.append(closest_centroids, closest_centriod)
        
        return closest_centroids



def PCA(x, var_to_retain = 0.99):
    Sigma = (1 / np.shape(x)[0]) * np.matmul(np.transpose(x), x)

    # Singular value decomposition of sigma
    results = np.linalg.svd(Sigma)
    U, S = results[0:2]

    # Find min number of principal components to preserve (var_to_retain) varience
    n_pc = np.where((np.cumsum(S) / np.sum(S)) > var_to_retain)[0][0]

    # Return data mapped to a lower dimention
    return np.matmul(x, U[:, 0:n_pc])
