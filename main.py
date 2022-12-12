from proj_iter2 import *

train_path = '/home/divyansh/Documents/ENPM808A/final-project/data'                    
testing_path = '/home/divyansh/Documents/ENPM808A/final-project/testing'

print("Available models are: ")
print('\n', "1: Linear Regression", '\n', "2: SVM", '\n', "3: XGBoost")
model = input("Please enter the model number: ")
print("Learning...")
MLpipeline(train_path, testing_path, model)