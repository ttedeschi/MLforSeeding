import os
import sys

import keras

from dataset import *

import pandas as pd
import numpy as np

from keras.layers import Input, Flatten, Dense
from keras.layers import concatenate, Dropout, BatchNormalization
from keras.models import Model
from keras import optimizers
from keras.constraints import max_norm

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

    features   = Input(shape=(len(featureLabs),), name='hit_input')
    b_norm = BatchNormalization()(features)
    dense = Dense(64, activation='relu', kernel_constraint=max_norm(0.1), name='dense_1')(features)
    dense = Dense(64, activation='relu', kernel_constraint=max_norm(0.1), name='dense_2')(dense)
    drop = Dropout(0.5)(dense)
    dense = Dense(64, activation='relu', kernel_constraint=max_norm(0.1), name='dense_3')(drop)
    dense = Dense(64, activation='relu', kernel_constraint=max_norm(0.1), name='dense_4')(dense)
    b_norm = BatchNormalization()(dense)
    pred = Dense(2, activation='relu', kernel_constraint=max_norm(0.1), name='output')(b_norm)

    model = Model(inputs=features, outputs=pred)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=50),
            ModelCheckpoint("best_model.h5",save_best_only="True")]

    Y_train_onehot = to_categorical(Y_train)

    print(Y_train)
    print(Y_train_onehot)

    X_train = X_train/np.max(X_train,axis=0)

    history = model.fit(X_train, Y_train_onehot, \
            batch_size = 1024 , epochs=100, shuffle=True, \
            validation_split=0.15, callbacks=callbacks, verbose=1)



