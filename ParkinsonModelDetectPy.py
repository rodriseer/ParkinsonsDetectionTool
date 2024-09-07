import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# training model class
class parkinsonModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = MinMaxScaler((-1,1))
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        # read data and preprocess data
        self.df = pd.read_csv(self.data_path)
        features = self.df.loc[:, self.df.columns != 'status'].values[:, 1:]
        labels = self.df.loc[:, 'status'].values

        # scale features
        features_scaled = self.scaler.fit_transform(features)

        # train test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=7)

    def train_model(self):
        # train model
        self.model = XGBClassifier()
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(model):
        # predict and calculate accuracy
        y_pred = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, y_pred) * 100
        return round(accuracy, 2)
    
    def predict(self, input_data):
        # predict new data if user has new dataset available
        scaled_data = self.scaler.transform(np.array([input_data]))
        prediction = self.model.predict(scaled_data)
        return prediction[0]

class ParkinsonApp:
    def __init__ (self, root, model):
        self.root = root
        self.model = model
        self.create_widgets()

    def create_widgets(self):
        self.root.title("Parkinson's Disease Detector")

        # button to display 
        btn_data = tk.Button(self.root, text = "Display Data", command=self.display_data)
        btn_data.pack()

        # button to display model accuracy
        btn_accuracy = tk.Button(self.root, text = "Model Accuracy Display: ", command = self.show_accuracy)
        btn_accuracy.pack()

        # entry fields for prediction input
        self.input_fields = []
        for col in self.model.df.coluns[1:-1]:
            label = tk.Label(self.root,text=f"Enter{col}:")





# data path temporary sample
model = parkinsonModel(data_path=r"C:\Users\parkinsons.data")
