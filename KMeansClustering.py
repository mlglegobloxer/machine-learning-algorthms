import numpy as np
from helperFunctions import *


class KMeansClusteringModel:
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

