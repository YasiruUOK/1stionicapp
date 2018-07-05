import tkinter
from tkinter import *

import inline
import matplotlib
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

dataset=pd.read_csv('t2.csv')
# dataset.drop(dataset.columns[1])

X=dataset.iloc[:,[4,5,2]]
y=dataset.iloc[:,[15,16,17]]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = linear_model.LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

def write_slogan():
    Stretch = float(e1.get())
    PRL = float(e2.get())
    PRW = float(e3.get())
    predict_values = regressor.predict([[PRL, PRW, Stretch]])
    e4.insert(0, str(np.round(predict_values[0][0])))
    e5.insert(0, str(np.round(predict_values[0][1])))
    e6.insert(0, str(np.round(predict_values[0][2])))




master = Tk()
master.geometry("400x400")
lable1 = Label(master, text="Stretch", font=("Helvetica", 24)).grid(row=0)
Label(master, text="PRL", font=("Helvetica", 24)).grid(row=1)
Label(master, text="PRW", font=("Helvetica", 24)).grid(row=2)
button = tkinter.Button(master, text="Result", command=write_slogan, font=("Helvetica", 24)).grid(row=3)
button.place(relx=0.5, rely=0.5, anchor=CENTER)
Label(master, text="FRJ", font=("Helvetica", 24)).grid(row=4)
Label(master, text="FRP", font=("Helvetica", 24)).grid(row=5)
Label(master, text="FRL", font=("Helvetica", 24)).grid(row=6)

e1 = Entry(master, font=("Helvetica", 24))
e2 = Entry(master, font=("Helvetica", 24))
e3 = Entry(master, font=("Helvetica", 24))
e4 = Entry(master, font=("Helvetica", 24))
e5 = Entry(master, font=("Helvetica", 24))
e6 = Entry(master, font=("Helvetica", 24))

e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
e4.grid(row=4, column=1)
e5.grid(row=5, column=1)
e6.grid(row=6, column=1)
mainloop()