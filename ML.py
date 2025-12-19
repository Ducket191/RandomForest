import numpy as np
import pandas as pd
import random as random
from collections import Counter
from data import PlantData

class Node():
    def __init__(self, featureIndex=None, threshold=None, left=None, right=None, infoGain=None, value=None):
        

        self.featureIndex = featureIndex
        self.threshold = threshold
        self.left = left
        self.right = right
        self.infoGain = infoGain
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, minSampleSplit = 5, maxDepth = 7):
        self.root = None
        self.minSampleSplit = minSampleSplit 
        self.maxDepth = maxDepth

    def buildTree(self, data, currentDepth=0):
        featureCol, label = data[:,:-1], data[:,-1]
        numOfSample, numOfFeature = np.shape(featureCol)
        #splitting
        if numOfSample >= self.minSampleSplit and currentDepth <= self.maxDepth:
            bestSplit = self.getBestSplit(data, numOfSample, numOfFeature) 
            if bestSplit is None or "InformationGain" not in bestSplit or bestSplit["InformationGain"] <= 0:
                return Node(value=self.calculateLeafValue(label))
            if bestSplit["InformationGain"]>0:
                leftSubtree = self.buildTree(bestSplit["dataLeft"], currentDepth + 1)
                rightSubtree = self.buildTree(bestSplit["dataRight"], currentDepth + 1)
                return Node(bestSplit["featureIndex"], bestSplit["threshold"], 
                            leftSubtree, rightSubtree, bestSplit["InformationGain"])

        leafValue = self.calculateLeafValue(label)
        return Node(value=leafValue)
    
    def getBestSplit(self, data, numOfSample, numOfFeature):
        bestSplit = {}
        maxInfoGain = - float("inf")

        for featureIndex in range(numOfFeature):
            featureValue = data[:, featureIndex]
            possibleThreshold = np.unique(featureValue)

            for threshold in possibleThreshold:
                dataLeft, dataRight = self.split(data, featureIndex, threshold)

                if len(dataLeft) > 0 and len(dataRight) > 0:
                    label, leftlabel, rightLable = data[:,-1], dataLeft[:,-1], dataRight[:,-1]
                    currentInfoGain = self.informationGain(label, leftlabel, rightLable)

                    if currentInfoGain > maxInfoGain:
                        bestSplit["featureIndex"] = featureIndex
                        bestSplit["threshold"] = threshold
                        bestSplit["dataRight"] = dataRight
                        bestSplit["dataLeft"] = dataLeft
                        bestSplit["InformationGain"] = currentInfoGain
                        maxInfoGain = currentInfoGain
    
        return bestSplit
    
    def split(self, data, featureIndex, threshold):

        dataLeft = np.array([row for row in data if row[featureIndex] <= threshold])
        dataRight = np.array([row for row in data if row[featureIndex]>threshold])

        return dataLeft, dataRight

    def informationGain(self, parent, leftChild, rightChild):
        leftPossibility = len(leftChild) / len(parent)
        rightPossibility = len(rightChild) / len(parent)
        gain = self.impurityCal(parent) - (leftPossibility*self.impurityCal(leftChild) + rightPossibility*self.impurityCal(rightChild))
        return gain
    
    def impurityCal(self, y):
        classLabels = np.unique(y)
        possibility = 0
        for child in classLabels:
            possibilityChild = len(y[y == child]) / len(y)
            possibility += possibilityChild**2
        return 1 - possibility
        
    def calculateLeafValue(self, Y):
        Y = Y.flatten()
        values, counts = np.unique(Y, return_counts=True)
        return values[np.argmax(counts)]
    
    def fit(self, X, Y):
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)
    
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.buildTree(dataset)

    def predict(self, X):        
        preditions = [self.makePrediction(child, self.root) for child in X]
        return preditions

    def makePrediction(self, x, tree):
        if tree.value!=None: return tree.value 
        featureVal = x[tree.featureIndex]
        if featureVal<=tree.threshold:
            return self.makePrediction(x, tree.left)
        else:
            return self.makePrediction(x, tree.right)
        
class RandomForestClassifier():
    def __init__(self, numberOfTree):
        self.numberOfTree = numberOfTree
        self.tree = []
    def fit(self, XData, YData):
        numOfSample, numOfFeature = np.shape(XData)
        for i in range (1,self.numberOfTree+1):
            # creating randomly xdata for each tree
            numOfSelectedFeatures = int(np.sqrt(numOfFeature))
            selectedFeatureIndex = random.sample(range(numOfFeature), numOfSelectedFeatures)
            selectedSampleIndex = np.random.choice(numOfSample, numOfSample, replace=True)
            selectedFeatures = XData[:, selectedFeatureIndex] 
            correspondingLabel = YData[selectedSampleIndex, :] 
            finalXData = selectedFeatures[selectedSampleIndex, :] 
            Tree = DecisionTreeClassifier(minSampleSplit=5, maxDepth=7)
            Tree.fit(finalXData, correspondingLabel)
            self.tree.append((Tree, selectedFeatureIndex))
    def predict(self, X):
        predictions = []
        for child in X:
            votes = []
            for tree, featureIndex in self.tree:
                Subsample = child[featureIndex].reshape(1, -1)
                votes.append(tree.predict(Subsample)[0])
            count = Counter(votes)
            majority = count.most_common(1)[0][0] 
            predictions.append(majority)
        return predictions
            
        
data = pd.read_csv("test.csv") #USE THIS ONE IF YOUR DATA IS IN A .CSV FILE

data = pd.DataFrame(PlantData) #USE THIS ONE IF YOUR DATA IS A list, dictionary, NumPy array, etc
X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)

classifier = DecisionTreeClassifier(minSampleSplit=5, maxDepth=7)
classifier.fit(X_train, Y_train)

Classifier = RandomForestClassifier(numberOfTree=100)
Classifier.fit(X_train, Y_train)

if __name__ == "__main__":
    Y_pred = classifier.predict(X_test)
    y_predRF = Classifier.predict(X_test)
    from sklearn.metrics import accuracy_score
    output = accuracy_score(Y_test, Y_pred)
    outputRF = accuracy_score(Y_test, y_predRF)
    print("DecisionTree:")
    print(output)
    print("RandomForest")
    print(outputRF)
