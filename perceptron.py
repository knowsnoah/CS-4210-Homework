#-------------------------------------------------------------------------
# AUTHOR: Noah Ojeda
# FILENAME: perceptron.py
# SPECIFICATION: This program trains and evaluates a Single-Layer Perceptron
#                and a Multi-Layer Perceptron (MLP) on the optdigits dataset.
#                It tests multiple learning rates and shuffle configurations
#                to determine which combination produces the highest accuracy.
#                The program prints the best accuracy found for each classifier
#                along with its corresponding hyperparameters.
# FOR: CS 4210- Assignment #3
# TIME SPENT: ~1hr
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

#learning rates
n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
#shuffle flags
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
Y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
Y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test

#variables to keep the highest accuracy for each algorithm
max_acc_Perceptron = 0
max_acc_MLP = 0

for learning_rate in n: #iterates over n

    for shuf in r: #iterates over r

        #iterates over both algorithms
        #-->add your Python code here
        algorithms = ["Perceptron", "MLP"]

        for algorithm in algorithms: #iterates over the algorithms
            #Create a Neural Network classifier
            #if Perceptron then
            #   clf = Perceptron()    #use those hyperparameters: eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000
            if algorithm == "Perceptron":
                clf = Perceptron(eta0=learning_rate, shuffle=shuf, max_iter=1000)
            #else:
            #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
            #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
            #                          shuffle = shuffle the training data, max_iter=1000
            #-->add your Python code here
            else: 
                clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate, hidden_layer_sizes=(25,),
                                    shuffle = shuf, max_iter=1000) 

            #Fit the Neural Network to the training data
            clf.fit(X_training,Y_training)

            #make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:
            #for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #--> add your Python code here
            correct = 0
            for (X_testSample, Y_testSample) in zip(X_test, Y_test):
                prediction = clf.predict([X_testSample])
                if prediction == Y_testSample:
                    correct+=1

            #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
            #and print it together with the network hyperparameters
            #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
            #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
            #--> add your Python code here
            accuracy = correct / len(Y_test)
            if algorithm == "Perceptron" and accuracy > max_acc_Perceptron:
                max_acc_Perceptron = accuracy
                print(f"Highest Perceptron accuracy so far: {max_acc_Perceptron:0.5f}, Parameters: learning rate={learning_rate}, shuffle={shuf}")
            elif algorithm == "MLP" and accuracy > max_acc_MLP:
                max_acc_MLP = accuracy
                print(f"Highest MLP accuarcy so far: {max_acc_MLP:0.5f}, Parameters: learning rate={learning_rate}, shuffle={shuf}")






