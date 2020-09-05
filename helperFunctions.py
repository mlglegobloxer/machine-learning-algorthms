# This file implements functions for many common when preprocessing,
# training and evaluating ML models
import numpy as np
from math import e
import matplotlib.pyplot as plt


def scaleFeature(vector):
    """ Returns for every element: (element - mean) / range """
    return (vector - (sum(vector) / len(vector))) / (max(vector) - min(vector))


def sigmoid(z):
    """ Element wise sigmoid of the input """
    return 1 / (1 + (e ** -z))


def plotCosts(costs):
    """ Given an input array of costs for each iteration, plots them as a function of number itterations """
    indecies = range(len(costs))
    plt.plot(indecies,costs)
    plt.yscale('log')
    plt.ylabel('Relative Cost (Log Scale)')
    plt.xlabel('Number of Training Iterations')
    plt.title('Cost at Each Training Iteration')
    plt.show()


def wait():
    """ Waits for the user to press enter before continuing the script """
    input("Press Enter to Continue: ")


def addBiases(x):
    """ Appends a row of 1's to the input, facilitating bias terms when multiplying with parameter matricies"""
    return np.hstack((np.ones((np.shape(x)[0], 1)), x))


def evaluateBinaryClassifier(predictions, outcomes):
    """ Prints the confusion matrix and summary statistics to evaluate some binary classifier """
    ## Generate confusion matrix
    TP = sum(predictions[np.where(outcomes == 1)] == 1)
    FP = sum(predictions[np.where(outcomes == 0)] == 1)
    TN = sum(predictions[np.where(outcomes == 0)] == 0)
    FN = sum(predictions[np.where(outcomes == 1)] == 0)
    confusion_matrix = np.array([[TP, FP], [FN, TN]])

    # Print confusion matrix
    print(f"Confusion Matrix:\n{confusion_matrix}\n")

    ## Calculate Accuracy, Precision, Recall, F1
    accuracy =  (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)

    # Print these metrics
    print(f"           Metrics")
    print(f"Accuracy  : {accuracy:.4}")
    print(f"Precision : {precision:.4}")
    print(f"Recall    : {recall:.4}")
    print(f"F1        : {F1:.4}\n")


def distance(p1, p2):
    """ 
    Finds the eucludean distance from each row in p1 from p2
    
    Returns a matrix where, matrix[i,j] is the distance from p1[i] to p2[j]
    """
    dist = []
        
    for i in range(np.shape(p1)[0]):
        rowDist = []

        for j in range(np.shape(p2)[0]):
            rowDist.append(sum((p1[i] - p2[j]) ** 2) ** 0.5)
            
        dist.append(rowDist)

    return np.array(dist)


def mode(matrix):
    """ Computes the row wise mode of the input matrix """
    modes = []
    
    for row in range(np.shape(matrix)[0]):
        modes.append(np.argmax(np.bincount(matrix[row, :])))

    return modes