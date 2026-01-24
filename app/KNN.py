import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

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