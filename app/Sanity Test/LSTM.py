import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    x, y, train_size=0.8, test_size=0.2, shuffle=False
)

model = Sequential()
model.add(LSTM(units=128, return_sequences=True, input_shape=(1, 63)))
model.add(Dropout(0.3))
model.add(LSTM(units=128))
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
