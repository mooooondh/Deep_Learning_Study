# mnist01_building_network.py

import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from book_ex.dataset.mnist import load_mnist

def sigmoid(a):
    return 1/(1+np.exp(-a))

def softmax(a):
    c= np.max(a)
    exp_a= np.sum(a- c)
    sum_exp_a= np.sum(exp_a)
    y= exp_a/sum_exp_a

    return y

def get_data():
    (X_train, y_train), (X_test, y_test)= load_mnist(normalize= True, flatten=True, one_hot_label= False)
    return X_test, y_test

def init_network():
    with open("./book_ex/ch03/sample_weight.pkl", 'rb') as f:
        network= pickle.load(f)

    return network

def predict(network, x):
    w1, w2, w3= network["W1"], network["W2"], network["W3"]
    b1, b2, b3= network["b1"], network["b2"], network["b3"]

    a1= np.dot(x, w1)+ b1
    z1= sigmoid(a1)
    a2= np.dot(z1, w2)+ b2
    z2= sigmoid(a2)
    a3= np.dot(z2, w3)+ b3
    y= softmax(a3)

    return y

X, t= get_data()
network= init_network()

accuracy_cnt= 0

for i in range(len(X)):
    y= predict(network, X[i])
    p= np.argmax(y)
    if (p== t[i]):
        accuracy_cnt+= 1

print("Accuracy: "+ str(float(accuracy_cnt)/len(X)))