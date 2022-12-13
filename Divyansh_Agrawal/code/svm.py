import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


class SVM():

    def __init__(self, train_full_X, train_full_y, test_X, test_y) -> None:
        self.full_X = train_full_X
        self.full_y = train_full_y
        self.test_X = test_X
        self.test_y = test_y
        self.splitData()
        self.prediction()
        self.curve_plotting(self.full_X, self.full_y)

    def splitData(self):
        """
        method for shuffling the data and splitting into training and validation
        """
        self.full_X = shuffle(self.full_X, random_state=20, n_samples=None)
        self.full_y = shuffle(self.full_y, random_state=20, n_samples=None)
        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.full_X, self.full_y, test_size=0.2, random_state=42)

    def prediction(self):
        """
        method for predicting the outputs using the model
        """
        kk = []
        scores_list = []
        self.var = 'poly'
        for k in range(1, 11, 1):
            kk.append(1/k)
            mor = SVR(C = 1/k, kernel = self.var, epsilon=0.2)
            mor = mor.fit(self.train_X, self.train_y)
            self.predict_y_train = mor.predict(self.train_X)
            self.predict_val_y = mor.predict(self.val_X)
            r2_score_value =  r2_score(self.val_y, self.predict_val_y)
            scores_list.append(r2_score_value)

        min_r2_score_index = scores_list.index(max(scores_list))
        mor = SVR(C = min_r2_score_index + 1, epsilon=0.2)
        mor.fit(self.full_X, self.full_y)
        self.predicted_full_y = mor.predict(self.test_X)
        r2_score_final = r2_score(self.test_y, self.predicted_full_y)
        print(scores_list)
        print("R2 score for the test data: ", r2_score_final)
        plt.plot(kk, scores_list)
        plt.xlabel("Value of lamba, regularization parameter")
        plt.ylabel("R2 Score")
        plt.title("R2 score Vs Regularization parameter")

    def curve_plotting(self, X,y):
        """
        method for plotting the curves
        """
        train_sizes, train_scores, test_scores = learning_curve(SVR(C = 1, kernel = self.var, epsilon=0.2), X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        plt.subplots(1, figsize=(10,10))
        plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
        plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

        plt.title("Learning Curve")
        plt.xlabel("Training Set Size"), plt.ylabel("R2 Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.show()