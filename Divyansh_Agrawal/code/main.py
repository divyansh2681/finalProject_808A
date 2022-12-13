from utils import *

train_path = input("Enter the path to the folder containing training files: ")                   
testing_path = input("Enter the path to the folder containing testing files: ")  

print("Available models are: ")
print('\n', "1: Linear Regression", '\n', "2: SVM", '\n', "3: XGBoost")
model = input("Please enter the model number: ")
print("Learning...")
MLpipeline(train_path, testing_path, model)