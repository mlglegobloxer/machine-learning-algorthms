import numpy as np
from helperFunctions import *
import matplotlib.pyplot as plt
from linearRegression import LinearRegModel


# Test scipt
lm = LinearRegModel()
n = 50


# Generate some noisy train data
noise = np.random.randn(n)
x = np.linspace(0, 20, n)
y = 32.3 + 5.2 * (x + noise)

x = scaleFeature(x)
y = scaleFeature(y)

x = x.reshape((-1,1))


# Train the lm, print out the parameters, plot the fit
lm.train(x, y, 0.5, 100, 0)
print(lm.parameters)

# Plot the fit of the linear model
y_hat = lm.predict(x)

# Plot the fit of line and train data
plt.plot(x, y, 'o')
plt.plot(x, y_hat)
plt.ylabel("y")
plt.xlabel("x")
plt.title("Plot of the Models Fit")
plt.show()


wait()

# Plot the costs to check proper convergence of gradient descent
plotCosts(lm.costsOverTime)
