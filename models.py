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
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

class MLmodels():

    def modelSVR(self, train_X, test_X, train_y, test_y):  # X here is my A matrix
        svr = SVR(epsilon=0.2)
        mor = MultiOutputRegressor(svr) # Create the Multioutput Regressor
        mor = mor.fit(train_X, train_y) #training
        y_pred_test = mor.predict(test_X) # predicting
        y_pred_train = mor.predict(train_X)
        print("error in the train set")
        self.errorCalculation(y_pred_train, train_y)
        print("error in the test set")
        self.errorCalculation(y_pred_test, test_y)

    def model_2(self, train_X, test_X, test_y, train_y):
        clf = Ridge(alpha=1.0)
        clf = clf.fit(train_X, train_y)
        y_pred_test = clf.predict(test_X) # predicting
        y_pred_train = clf.predict(train_X)
        print("error in the train set")
        self.errorCalculation(y_pred_train, train_y)
        print("error in the test set")
        self.errorCalculation(y_pred_test, test_y)

    def model_3(self, train_X, test_X, test_y, train_y):
        y_pred = 0
        return y_pred

    def model_4(self, train_X, test_X, test_y, train_y):
        y_pred = 0
        return y_pred

    
    def errorCalculation(self, y_pred, y_actual):
        mse_one = mean_squared_error(y_actual[:, 0], y_pred[:,0])
        mse_two = mean_squared_error(y_actual[:, 1], y_pred[:,1])
        print(f'MSE - Training for first regressor: {mse_one} - second regressor: {mse_two}')
        mae_one = mean_absolute_error(y_actual[:, 0], y_pred[:,0])
        mae_two = mean_absolute_error(y_actual[:, 1], y_pred[:,1])
        print(f'MAE - Training for first regressor: {mae_one} - second regressor: {mae_two}')
        # r2_score_val = r2_score(y_actual, y_pred)
        # print(r2_score_val)



