import numpy as np
import pandas as pd
import glob
from regression import *
from svm import *
from xgboost__ import *
from mlp import *

"""
LG: Local Goal
FG: Final Goal
RosPos: Robot Position
cmd: command actions
"""

class MLpipeline():
    """
    class for the ML pipeline
    """
    def __init__(self, path_train, path_test, model):
        self.train_X, self.train_y = self.runPipeline(path_train)
        self.test_X, self.test_y = self.runPipeline(path_test)
        self.learning(self.train_X, self.train_y, self.test_X, self.test_y, model)

    def runPipeline(self, path):
        """
        method for returning the X and y matrices
        """
        dataFrame = self.importFiles(path)
        dataFrame_cleaned = self.cleanData(dataFrame)
        self.extractData(dataFrame_cleaned)
        self.A = self.generateA()
        self.y = np.vstack((self.cmdV, self.cmdW)).T
        return self.A, self.y

    def importFiles(self, path): 
        """
        method for importing the files and converting it into a dataframe
        """
        files = glob.glob(path + '/*.csv')
        li = []
        for f in files:
            temp_df = pd.read_csv(f, header=None)
            li.append(temp_df)
        df = pd.concat(li, axis=0)
        return df

    def cleanData(self, dataFrame):
        """
        method for cleaning the dataset
        """
        df = dataFrame.drop_duplicates()
        df = df.dropna()
        df = df.to_numpy()
        return df

    def extractData(self, arrData):
        """
        method for extracting different data from the whole numpy array
        """
        self.numOfPoints, self.numOfCols = np.shape(arrData) 
        self.laserRange = arrData[:,0:1080]
        self.FG = arrData[:, 1080:1084]
        self.LG = arrData[:, 1084:1088]
        self.RobPos = arrData[:, 1088:1092]
        self.cmdV = arrData[:, 1092]
        self.cmdW = np.array(arrData[:, 1093])
                
    def distCurrentPLocalG(self):
        """
        feature: distance b/w local goal anf the current position
        """
        dist = ((self.RobPos[:, 0] - self.LG[:, 0])**2 + (self.RobPos[:, 1] - self.LG[:, 1])**2)**0.5
        dist = dist.reshape((self.numOfPoints, 1))
        return dist

    def distCurrentPFinalG(self):
        """
        feature: distance b/w final goal anf the current position
        """
        dist = ((self.RobPos[:, 0] - self.FG[:, 0])**2 + (self.RobPos[:, 1] - self.FG[:, 1])**2)**0.5
        dist = dist.reshape((self.numOfPoints, 1))
        return dist

    def angleCurrentPLocalG(self):
        """
        feature: angle of the line of sight joining current position and local goal with the x axis
        """
        numerator = (self.LG[:, 1] - self.RobPos[:, 1])
        denominator = (self.LG[:, 0] - self.RobPos[:, 0])
        self.slopeAngle1 = np.zeros((denominator.shape[0],1))
        for i in range(denominator.shape[0]):
            if denominator[i] == 0:
                self.slopeAngle1[i] = np.arctan(np.sign(numerator[i])*np.inf)
            else:
                self.slopeAngle1[i] = np.arctan(numerator[i]/denominator[i])
        return self.slopeAngle1
    
    def angleCurrentPFinalG(self):
        """
        feature: angle of the line of sight joining current position and final goal with the x axis
        """
        numerator = (self.FG[:, 1] - self.RobPos[:, 1])
        denominator = (self.FG[:, 0] - self.RobPos[:, 0])
        self.slopeAngle2 = np.zeros((denominator.shape[0],1))
        for i in range(denominator.shape[0]):
            if denominator[i] == 0:
                self.slopeAngle2[i] = np.arctan(np.sign(numerator[i])*np.inf)
            else:
                self.slopeAngle2[i] = np.arctan(numerator[i]/denominator[i])
        return self.slopeAngle2

    def distFinalGLocalG(self):
        """
        feature: distance b/w the local goal and the final goal
        """
        dist = ((self.LG[:, 0] - self.FG[:, 0])**2 + (self.LG[:, 1] - self.FG[:, 1])**2)**0.5
        dist = dist.reshape((self.numOfPoints, 1))
        return dist


    def distInFrontOfBot(self):
        """
        feature: value of LIDAR infront of the bot
        """
        dist = self.laserRange[:, 539]
        dist = dist.reshape((self.numOfPoints, 1))
        return dist

    def generateA(self):
        """
        method for generating the A matrix
        """
        self.A = np.hstack((np.ones(self.numOfPoints).reshape(self.numOfPoints, 1), \
            self.distCurrentPLocalG(), \
            self.distCurrentPFinalG(), \
            self.angleCurrentPLocalG(), \
            self.angleCurrentPFinalG(), \
            self.distFinalGLocalG(), \
            self.distInFrontOfBot()))
        return self.A

    def learning(self, train_X, train_y, test_X, test_y, model):
        """
        method for running the model depending on user input, also comment/uncomment the 
        lines 151, 153, 157 depending on whether you want to run the model for 
        translational velocity or rotational velocity
        """
        if (model == "1"):
            LinearRegression(train_X, train_y[:, 1], test_X, test_y[:, 1])
            # LinearRegression(train_X, train_y[:, 0], test_X, test_y[:, 0])
        elif (model == "2"):
            # SVM(train_X, train_y[:, 0], test_X, test_y[:, 0])
            SVM(train_X, train_y[:, 1], test_X, test_y[:, 1])
        elif (model == "3"):
            XGBoost(train_X, train_y[:, 1], test_X, test_y[:, 1])
            # XGBoost(train_X, train_y[:, 0], test_X, test_y[:, 0])
        else:
            print("Wrong input, please try again")