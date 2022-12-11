import numpy as np
from sklearn import svm
# import pandas as pd
# import glob
from sklearn.linear_model import LogisticRegression
# from scipy import stats
# from scipy.spatial.transform import Rotation as R
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.neural_network import MLPClassifierRegressor

class MLmodels():

    def modelSVR(self, train_A, test_A):  # X here is my A matrix
        svr = SVR(epsilon=0.2)

        # Create the Multioutput Regressor
        mor = MultiOutputRegressor(svr)
        y = np.vstack((self.cmdV, self.cmdW))
        # Train the regressor
        mor = mor.fit(train_A, y.T)

        # Generate predictions for training data
        y_pred = mor.predict(test_A)
        return y_pred

    def model_2(self, train_A, test_A):
        y_pred = 0
        return y_pred

    def model_3(self, train_A, test_A):
        y_pred = 0
        return y_pred

    def model_4(self, train_A, test_A):
        y_pred = 0
        return y_pred

    
    def errorCalculation(self, y_pred, y_actual):
        mse_one = mean_squared_error(y_actual[:, 0], y_pred[:,0])
        mse_two = mean_squared_error(y_actual[:, 1], y_pred[:,1])
        print(f'MSE - Training for first regressor: {mse_one} - second regressor: {mse_two}')
        mae_one = mean_absolute_error(y_actual[:, 0], y_pred[:,0])
        mae_two = mean_absolute_error(y_actual[:, 1], y_pred[:,1])
        print(f'MAE - Training for first regressor: {mae_one} - second regressor: {mae_two}')

    
    # def learning(self):
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