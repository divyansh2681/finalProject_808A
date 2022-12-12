import numpy as np
import pandas as pd
import glob
from scipy import stats
from scipy.spatial.transform import Rotation as R
from regression import *
from svm import *
from xgboost__ import *


"""
LG: Local Goal
FG: Final Goal
RosPos: Robot Position
cmd: command actions
"""
class MLpipeline():

    def __init__(self, path_train, path_test, model):
        self.train_X, self.train_y = self.runPipeline(path_train)
        self.test_X, self.test_y = self.runPipeline(path_test)
        self.learning(self.train_X, self.train_y, self.test_X, self.test_y, model)

    def runPipeline(self, path):
        dataFrame = self.importFiles(path)
        dataFrame_cleaned = self.cleanData(dataFrame)
        self.extractData(dataFrame_cleaned)
        self.A = self.generateA()
        self.y = np.vstack((self.cmdV, self.cmdW)).T
        return self.A, self.y

    def importFiles(self, path): 
        files = glob.glob(path + '/*.csv')
        li = []
        for f in files:
            temp_df = pd.read_csv(f, header=None)
            li.append(temp_df)
        df = pd.concat(li, axis=0)
        return df

    def cleanData(self, dataFrame):
        df = dataFrame.drop_duplicates() # dropping rows that are duplicates
        df = df.dropna() # dropping rows that have NaN values
        df = df.to_numpy()
        return df

    def extractData(self, arrData):
        self.numOfPoints, self.numOfCols = np.shape(arrData) 
        self.laserRange = arrData[:,0:1080]
        self.FG = arrData[:, 1080:1084]
        self.LG = arrData[:, 1084:1088]
        self.RobPos = arrData[:, 1088:1092]
        self.cmdV = arrData[:, 1092]
        self.cmdW = np.array(arrData[:, 1093])
                
    def distCurrentPLocalG(self):
        dist = ((self.RobPos[:, 0] - self.LG[:, 0])**2 + (self.RobPos[:, 1] - self.LG[:, 1])**2)**0.5
        dist = dist.reshape((self.numOfPoints, 1))
        return dist

    def distCurrentPFinalG(self):
        dist = ((self.RobPos[:, 0] - self.FG[:, 0])**2 + (self.RobPos[:, 1] - self.FG[:, 1])**2)**0.5
        dist = dist.reshape((self.numOfPoints, 1))
        return dist

    def angleCurrentPLocalG(self):
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
        dist = ((self.LG[:, 0] - self.FG[:, 0])**2 + (self.LG[:, 1] - self.FG[:, 1])**2)**0.5
        dist = dist.reshape((self.numOfPoints, 1))
        return dist
 
    def quaternionToEuler(self):
        quats = []
        euler = []
        for i in range(len(self.RobPos)):
            r = R.from_quat([self.RobPos[i, -1], 0, 0, self.RobPos[i, -2]])
            quats.append(r)
        for i in quats:
            euler.append(i.as_euler('zyx', degrees=True))
        return euler
    
    def lidarValAtLocalG(self): # the laser range here is not the whole array, it is that specific data point - size will be 1 * 1094
        # val = self.laserRange[810 - self.slopeAngle1*4]
        # print(np.shape(np.ones(self.numOfPoints).reshape(self.numOfPoints, 1)), "one vector")
        return np.ones(self.numOfPoints).reshape(self.numOfPoints, 1)

    def lidarValAtFinalG(self):
        return np.ones(self.numOfPoints).reshape(self.numOfPoints, 1)

    def distInFrontOfBot(self):
        dist = self.laserRange[:, 539]
        dist = dist.reshape((self.numOfPoints, 1))
        return dist

    def generateA(self):
        self.A = np.hstack((np.ones(self.numOfPoints).reshape(self.numOfPoints, 1), \
            self.distCurrentPLocalG(), \
            self.distCurrentPFinalG(), \
            self.angleCurrentPLocalG(), \
            self.angleCurrentPFinalG(), \
            self.distFinalGLocalG(), \
            self.distInFrontOfBot(), \
            self.lidarValAtLocalG(), \
            self.lidarValAtFinalG()))
        return self.A
    
    ###Note: Verify that angle units are same and avoid occurence of infinity
    # def angle_diff_goal(self, bot_pose:np.array, x_bot:np.array, y_bot:np.array, \
    #     x_local:np.array, y_local:np.array, x_final:np.array, y_final:np.array)->None:

    #     ####check the closest angle
    #     rows = x_local.shape[0]
    #     target_local_ang = np.zeros((rows,1))
    #     target_final_ang = np.zeros((rows,1))
    #     for i in range(x_local.shape[0]):
    #         # print(x_local[i])   
    #         x_diff = x_local[i] - x_bot[i]
    #         y_diff = y_local[i] - y_bot[i]
    #         if x_diff!=0:
    #             target_local_ang[i] = np.rad2deg(np.arctan(y_diff/x_diff))
    #             target_final_ang[i] = np.rad2deg(np.arctan(y_diff/x_diff))
    #         else:
    #             target_local_ang[i] = np.rad2deg(np.arctan(np.sign(y_diff)*np.inf))
    #             target_final_ang[i] = np.rad2deg(np.arctan(np.sign(y_diff)*np.inf))
    #     target_local_ang[target_local_ang<0] = 360+target_local_ang[target_local_ang<0]
    #     target_final_ang[target_final_ang<0] = 360+target_final_ang[target_final_ang<0]
        
    #     self.local_angdiff = bot_pose.reshape((rows,1))-target_local_ang
    #     self.final_angdiff = bot_pose.reshape((rows,1)) - target_final_ang
    #     # print((self.final_angdiff))


    #      ####return the closest angle c/w or ac/w 
    #     return None


    def learning(self, train_X, train_y, test_X, test_y, model):
        if (model == "1"):
            LinearRegression(train_X, train_y, test_X, test_y)
        elif (model == "2"):
            SVM(train_X, train_y[:, 0], test_X, test_y[:, 0])
        elif (model == "3"):
            XGBoost(train_X, train_y[:, 0], test_X, test_y[:, 0])
            # XGBoost(train_X, train_y, test_X, test_y)
        else:
            print("Wrong input, please try again")