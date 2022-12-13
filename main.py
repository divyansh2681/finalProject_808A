from proj_iter2 import *

train_path = '/home/divyansh/Documents/ENPM808A/final-project/corridor'                    
testing_path = '/home/divyansh/Documents/ENPM808A/final-project/testing_corridor'

# train_path = '/home/divyansh/Documents/ENPM808A/final-project/box'                    
# testing_path = '/home/divyansh/Documents/ENPM808A/final-project/testing_box'

print("Available models are: ")
print('\n', "1: Linear Regression", '\n', "2: SVM", '\n', "3: XGBoost")
model = input("Please enter the model number: ")
print("Learning...")
MLpipeline(train_path, testing_path, model)