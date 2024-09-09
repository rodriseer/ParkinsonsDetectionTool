import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# training model class
class ParkinsonModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.model = None
        self.scaler = MinMaxScaler((-1,1))
        self.x_train, self.x_test, self.y_train, self.y_test = None, None, None, None

    def load_data(self):
        # read data and preprocess data
        self.df = pd.read_csv(self.data_path)
        features = self.df.loc[:, self.df.columns != 'status'].values[:, 1:20]
        labels = self.df.loc[:, 'status'].values

        # scale features
        features_scaled = self.scaler.fit_transform(features)

        # train test split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=7)

    def train_model(self):
        # train model
        self.model = XGBClassifier()
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
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
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.create_widgets()

    def create_widgets(self):
        self.root.title("Parkinson's Disease Detector")
        
        # grid layout
        input_frame = tk.Frame(self.root)
        input_frame.pack(pady=20, padx=20)

        # button to display data
        btn_data = tk.Button(self.root, text="Display Data", command=self.display_data)
        btn_data.pack()

        # button to display model accuracy
        btn_accuracy = tk.Button(self.root, text="Show Model Accuracy", command=self.show_accuracy)
        btn_accuracy.pack()

         # button to display "Information Terms"
        btn_terms = tk.Button(self.root, text="Show Information Terms", command=self.display_information_terms)
        btn_terms.pack(pady=10)

        # descriptive labels on put for better user understanding
        labels = [
    "Fundamental Frequency (Fo) in Hz:",
    "Max Fundamental Frequency (Fhi) in Hz:",
    "Min Fundamental Frequency (Flo) in Hz:",
    "Jitter (Percentage):",
    "Jitter (Absolute):",
    "RAP:",
    "PPQ:",
    "DDP:",
    "Shimmer:",
    "Shimmer (dB):",
    "Shimmer APQ3:",
    "Shimmer APQ5:",
    "MDVP APQ:",
    "Shimmer DDA:",
    "NHR:",
    "HNR:",
    "RPDE:",
    "DFA:",
    "PPE:",
]

        # entry fields for prediction input
        self.input_fields = []
        for i, label in enumerate(labels):
            lbl = tk.Label(input_frame, text=label)
            lbl.grid(row=i, column=0, sticky="w", padx=5, pady=5)

            entry = tk.Entry(input_frame)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.input_fields.append(entry)




        # button to predict
        btn_predict = tk.Button(self.root, text="Predict Parkinson's", command=self.predict_parkinsons)
        btn_predict.pack(pady=10)

        # label to display result
        self.result_label = tk.Label(self.root, text="")
        self.result_label.pack()

    def display_data(self):
        # create new window for the data
        data_window = tk.Toplevel(self.root)
        data_window.title("Data Viewer")

        # create a treeview to display the dataframe
        tree = ttk.Treeview(data_window)
        tree["columns"] = list(self.model.df.columns)
        tree["show"] = "headings"
        for column in self.model.df.columns:
            tree.heading(column, text=column)
            tree.column(column, width=100)

        # insert data
        for _, row in self.model.df.iterrows():
            tree.insert("", "end", values=list(row))
        
        tree.pack(fill="both", expand=True)
    
    def show_accuracy(self):
        accuracy = self.model.evaluate_model()
        messagebox.showinfo("Model Accuracy", f"The model has an accuracy of {accuracy}%")

    def predict_parkinsons(self):
        input_data = []
        try:
            for entry in self.input_fields:
                input_data.append(float(entry.get()))
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values")
            return

        prediction = self.model.predict(input_data)
        self.result_label.config(text="Parkinson's Detected!" if prediction == 1 else "No Parkinson's Detected.")

    def display_information_terms(self):
        try:
            with open("C:\\Users\\rodri\\Information_Terms.txt", "r", encoding="utf-8") as file:
                terms_content = file.read()
        except FileNotFoundError:
            messagebox.showerror("File Error", "The file was not found.")
            return

        # window to display information terms
        terms_window = tk.Toplevel(self.root)
        terms_window.title("Terms Dictionary")

        # create text widget to display the content
        text_widget = tk.Text(terms_window, wrap="word", height=20, width=80)
        text_widget.insert('1.0', terms_content)
        text_widget.config(state='disabled')  # Disable editing
        text_widget.pack(pady=10, padx=10)


def main():
    # create an instance of the model and load the data
    # make sure to locate your data file
    model = ParkinsonModel(data_path=r"C:\Users\rodri\parkinsons.data")
    model.load_data()
    model.train_model()
    
    # create the main window for the app
    root = tk.Tk()
    app = ParkinsonApp(root, model)
    
    # run the tkinter main loop
    root.geometry("1280x720")
    root.mainloop()

if __name__ == "__main__":
    main()
