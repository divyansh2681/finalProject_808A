import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


class LinearRegression():

    def __init__(self, train_full_X, train_full_y, test_X, test_y) -> None:
        self.full_X = train_full_X
        self.full_y = train_full_y
        self.test_X = test_X
        self.test_y = test_y
        self.splitData()
        self.prediction()
        self.curve_plotting(self.full_X, self.full_y)

    def splitData(self):
        self.full_X = shuffle(self.full_X, random_state=20, n_samples=None)
        self.full_y = shuffle(self.full_y, random_state=20, n_samples=None)

        # self.full_y[:, 0][self.full_y[:, 0]>max(self.full_y[:, 0])] = max(self.full_y[:, 0])
        # self.full_y[:, 1][self.full_y[:, 1]>max(self.full_y[:, 1])] = max(self.full_y[:, 1])


        # self.full_y[:, 0][self.full_y[:, 0]<min(self.full_y[:, 0])] = min(self.full_y[:, 0])
        # self.full_y[:, 1][self.full_y[:, 1]<min(self.full_y[:, 1])] = min(self.full_y[:, 1])


        self.train_X, self.val_X, self.train_y, self.val_y = train_test_split(self.full_X, self.full_y, test_size=0.2, random_state=42)

    def prediction(self):
        kk = []
        scores_list = []
        for k in range(1, 11, 1):
            kk.append(k)
            reg = Ridge(alpha = k)
            reg.fit(self.train_X, self.train_y)
            self.predict_y_train = reg.predict(self.train_X)
            self.predict_val_y = reg.predict(self.val_X)
            r2_score_value =  r2_score(self.val_y, self.predict_val_y)
            scores_list.append(r2_score_value)

        min_r2_score_index = scores_list.index(max(scores_list))

        reg = Ridge(alpha = min_r2_score_index + 1)
        reg.fit(self.full_X, self.full_y)

        self.predicted_full_y = reg.predict(self.test_X)

        r2_score_final = r2_score(self.test_y, self.predicted_full_y)
        print(r2_score_final)
        plt.plot(kk, scores_list)
        plt.xlabel("Value of lamba, regularization parameter")
        plt.ylabel("R2 Score")
        plt.title("R2 score Vs Regularization parameter")

    def curve_plotting(self, X,y):
        train_sizes, train_scores, test_scores = learning_curve(Ridge(alpha=1), X, y, cv=10, n_jobs=-1, train_sizes=np.linspace(0.01, 1.0, 50))
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