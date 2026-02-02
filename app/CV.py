from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "hand_landmarker.task")

x = []
y = []

path = r"C:\Users\sophs\OneDrive\Desktop\Sign-Language-Interpreter\asl_data"
dir_list = os.listdir(path)

for item in dir_list:
    second_path = os.path.join(path, item)
    for file in os.listdir(second_path):
        y.append(item)
        x.append(np.load(os.path.join(second_path, file)))

# Checked if index size matches the label size. (Also previoussly checked that landmarks are of correct size)
#if len(x) == len(y):
    #print("Data loaded successfully.")

knn = KNeighborsClassifier(n_neighbors=25)

knn.fit(x, y)

k_folds = KFold(n_splits = 20)

scores = cross_val_score(knn, x, y, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))