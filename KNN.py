import numpy as np


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
            dist = self.distance(x, self.X)

            knn_to_x = np.argpartition(dist, k)[:, 0:k]  # Indecies of the knn to x in the train set
            y_hat = self.mode(self.labels[knn_to_x])     # Predict the knn's modal class
            return np.array(y_hat)
        
        else:
            print("Please first train the model")


    def distance(self, p1, p2):
        dist = []
        
        for i in range(np.shape(p1)[0]):
            rowDist = []

            for j in range(np.shape(p2)[0]):
                rowDist.append(sum((p1[i] - p2[j]) ** 2) ** 0.5)  # Euclidian Distance
            
            dist.append(rowDist)

        return np.array(dist)


    def mode(self, matrix):
        modes = []  # Computes the row wise mode of the input matrix

        for row in range(np.shape(matrix)[0]):
            modes.append(np.argmax(np.bincount(matrix[row, :])))

        return modes
