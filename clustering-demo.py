import numpy as np
import matplotlib.pyplot as plt
from helperFunctions import *
from KMeansClustering import KMeansClusteringModel

# Test script
half_n = 25

x_class1 = np.array([(10 + 2.7 * np.random.randn(half_n)), (10 + 5.7 * np.random.randn(half_n))])
x_class2 = np.array([(25 + 3.1 * np.random.randn(half_n)), (25 + 5.1 * np.random.randn(half_n))])
x_class3 = np.array([(1.1 * np.random.randn(half_n)), (50 + 7.1 * np.random.randn(half_n))])

X = np.transpose(np.hstack((x_class1, x_class2, x_class3)))

for i in range(np.shape(X)[1]):
    X[:, i] = scaleFeature(X[:, i])

plt.scatter(X[:,0], X[:,1])
plt.title("Generated Dataset, consisting of 3 clusters\n")
plt.show()

wait()
k_means = KMeansClusteringModel()
k_means.train(X, k = 3)

print(f"\nCoordinates of Cluster Centroids: \n{k_means.centroids}\n")

plt.scatter(X[:,0], X[:,1])
plt.scatter(k_means.centroids[:,0], k_means.centroids[:,1], s=300, c="#000000", marker="*")

plt.title("Synthesised clusters, Stars denote cluster centroids")
plt.xlabel("x_0")
plt.ylabel("x_1")
plt.show()

wait()

plotCosts(k_means.costsOverTime)