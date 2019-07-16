import urllib.request
import sys
import os

import pandas as pd
import numpy as np

from time import time
from dataset import *

import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("usage: ", sys.argv[0], " rowstoread (if -1 read alla file) ")
        exit(1)

    numofrows = int(sys.argv[1])

    remote_data = "./"
    data_files = [remote_data + f for f in os.listdir(remote_data) \
            if "Doublets" in f and f.endswith("h5")]
    print("I will use : ", data_files)

    data = None
    if numofrows < 0:
        data = pd.read_hdf(data_files[0])
    else:
        data = pd.read_hdf(data_files[0], stop=numofrows)

    print(data.head())

    print("File read done")

    true = (data["label"]==1.0)
    fake = (data["label"]==-1.0)
    print("Number of doublets before sampling: %d"%len(data))

    sig = data[true]
    bkg = data[fake].sample(n=len(sig))
    data = pd.concat([sig,bkg]).sample(frac=1.0)

    print("New number of doublets: %d"%len(data))

    X = data[featureLabs].values
    Y = data["label"] == 1.0

    print("Features thata will be used: ")
    print(featureLabs)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

    print('Training Features Shape:', X_train.shape)
    print('Training Labels Shape:', Y_train.shape)
    print('Testing Features Shape:', X_test.shape)
    print('Testing Labels Shape:', Y_test.shape)

    print('Start training')
    numofdecisiontree = 1000
    rf = RandomForestClassifier(n_estimators = numofdecisiontree)
    rf.fit(X_train, Y_train)
    print('Training done')

    Y_pred = rf.predict(X_test)

    print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

    Y_pred = np.where(Y_pred == True, 1, Y_pred)
    Y_pred = np.where(Y_pred == False, 0, Y_pred)

    Y_test = np.where(Y_test == True, 1, Y_test)
    Y_test = np.where(Y_test == False, 0, Y_test)

    #plt.plot(Y_pred)
    #plt.plot(Y_test)

    diff = Y_pred - Y_test

    plt.plot(diff)

    plt.title("RF output diff",fontsize=15,fontweight="bold")
    plt.savefig("finalplot.png")
