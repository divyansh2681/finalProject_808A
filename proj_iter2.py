import numpy as np
# from sklearn import svm
import pandas as pd
import glob
# from sklearn.linear_model import LogisticRegression
from scipy import stats
from scipy.spatial.transform import Rotation as R
# from sklearn.multioutput import MultiOutputRegressor
# , MultiOutputClassifier
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.neural_network import MLPClassifier
from models import *

"""
LG: Local Goal
FG: Final Goal
RosPos: Robot Position
cmd: command actions
"""
class MLpipeline():

    def __init__(self, path):
        dataFrame = self.importFiles(path)
        dataFrame_cleaned = self.cleanData(dataFrame)
        self.extractData(dataFrame_cleaned)
        self.A = self.generateA()
        # self.learning()

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
    
    # def learning(self, y):
    #     print("Available models are: ")
    #     print('\n', "1: SVM", '\n', "2: Perceptron", '\n', "3: Neural Net", '\n', "4: Regression")
    #     model = input("Please enter the model number: ")
    #     print("Predicting the mean squared error and mean absolute error")
    #     if (model == "1"):
    #         y_pred = self.modelSVR(X, self.A)
    #     elif (model == "2"):
    #         y_pred = self.model_2(X, self.A)
    #     else:
    #         print("Wrong input type, please try again")   

    #     self.errorCalculation(y_pred, y_actual)
        


train_path = '/home/divyansh/Documents/ENPM808A/final-project/data'                    
train_obj = MLpipeline(train_path)
train_A = train_obj.A
train_y = np.vstack((train_obj.cmdV, train_obj.cmdW))

testing_path = '/home/divyansh/Documents/ENPM808A/final-project/testing'
test_obj = MLpipeline(testing_path)
test_A = test_obj.A
test_y = np.vstack((test_obj.cmdV, test_obj.cmdW))


# y = np.vstack((obj.cmdV, obj.cmdW))
# print(np.shape(y))


############################################################## SVR
## Create the SVR regressor
# svr = SVR(epsilon=0.2)

# # Create the Multioutput Regressor
# mor = MultiOutputRegressor(svr)

# # Train the regressor
# mor = mor.fit(A_matrix, y.T)

# Generate predictions for training data
# y_pred = mor.predict(A_matrix)

# # Evaluate the regressor
# mse_one = mean_squared_error(obj.cmdV, y_pred[:,0])
# mse_two = mean_squared_error(obj.cmdW, y_pred[:,1])
# print(f'MSE - Training for first regressor: {mse_one} - second regressor: {mse_two}')
# mae_one = mean_absolute_error(obj.cmdV, y_pred[:,0])
# mae_two = mean_absolute_error(obj.cmdW, y_pred[:,1])
# print(f'MAE - Training for first regressor: {mae_one} - second regressor: {mae_two}')

# # Generate predictions for testing data
# y_pred = mor.predict(test_A)

# # Evaluate the regressor
# mse_one = mean_squared_error(obj_test.cmdV, y_pred[:,0])
# mse_two = mean_squared_error(obj_test.cmdW, y_pred[:,1])
# print(f'MSE - Testing for first regressor: {mse_one} - second regressor: {mse_two}')
# mae_one = mean_absolute_error(obj_test.cmdV, y_pred[:,0])
# mae_two = mean_absolute_error(obj_test.cmdW, y_pred[:,1])
# print(f'MAE - Testing for first regressor: {mae_one} - second regressor: {mae_two}')

################################################### MULTI LAYER PERCEPTRON CLASSIFIER

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# clf.fit(A_matrix, y.T)
# y_pred = clf.predict(test_A)
# mse_one = mean_squared_error(obj_test.cmdV, y_pred[:,0])
# mse_two = mean_squared_error(obj_test.cmdW, y_pred[:,1])
# print(f'MSE - Testing for first regressor: {mse_one} - second regressor: {mse_two}')
# mae_one = mean_absolute_error(obj_test.cmdV, y_pred[:,0])
# mae_two = mean_absolute_error(obj_test.cmdW, y_pred[:,1])
# print(f'MAE - Testing for first regressor: {mae_one} - second regressor: {mae_two}')
