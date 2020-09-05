import numpy as np
import matplotlib.pyplot as plt
from helperFunctions import *
from models import LogisticRegression


print("Logistic Regression Demo,\n")
lrClassifier = LogisticRegression()
n = 100

# Generate dataset
x_0 = np.linspace(20, 40, n) + (20 * np.random.randn(n))
x_1 = (np.linspace(0, 5, n) ** 2) + (40 * np.random.randn(n))
y = ((x_0 + x_1 + (10 * np.random.randn(n))) > 36.5) * 1    # Multiplying by 1 gives y as 1 or 0 instead of booleans

# # Plot this dataset
# plt.plot(x_0[np.where(y == 0)], x_1[np.where(y == 0)], 'o', c="b")
# plt.plot(x_0[np.where(y == 1)], x_1[np.where(y == 1)], 'o', c="r")
# plt.xlabel("x_0")
# plt.ylabel("x_1")
# plt.show()

x_0 = scaleFeature(x_0)
x_1 = scaleFeature(x_1)


X = np.transpose(np.vstack((x_0, x_1))) # Transform x into a matrix


lrClassifier.train(X, y, 5, 500) # Train model
predictions = lrClassifier.predict(X) # Train set predictions

evaluateBinaryClassifier(predictions, y) # Evaluate train set predictions

# Plot the decition boundary
plt.plot(x_0[np.where(y == 0)], x_1[np.where(y == 0)], 'o', c="b")
plt.plot(x_0[np.where(y == 1)], x_1[np.where(y == 1)], 'o', c="r")
plt.xlabel("x_0")
plt.ylabel("x_1")
plt.title("Learned Decition Boundary for the Generated Dataset")
# Generate the decition boundary
boundary_x = np.linspace(-0.5,0.5,25)
param = lrClassifier.parameters
boundary_y = (-1 / param[2]) * (param[0] + boundary_x * param[1])
# Plot the decition boundary
plt.plot(boundary_x, boundary_y, c="k")
plt.show()

wait()

plotCosts(lrClassifier.costsOverTime)

print(f"\nTrained Parameters: {lrClassifier.parameters}\n")
wait()

print("\nKNN Demo,\n")

from models import KNNClassifier

knn = KNNClassifier()
knn.train(X, y)
predictions = knn.predict(X, 10)

evaluateBinaryClassifier(predictions, y)

plt.plot(x_0[np.where(y == 0)], x_1[np.where(y == 0)], 'o', c="b")
plt.plot(x_0[np.where(y == 1)], x_1[np.where(y == 1)], 'o', c="r")
plt.xlabel("x_0")
plt.ylabel("x_1")
plt.show()

#mesh_x = np.linspace(min(x_0), max(x_0), 0.1)
#mesh_y = np.linspace(min(x_1), max(x_1), 0.1)

#np.meshgrid(mesh_x, mesh_y)
