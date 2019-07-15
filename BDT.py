import urllib.request
import sys
import os

import pandas as pd
import numpy as np

from time import time
from dataset import *

import matplotlib.pyplot as plt

from matplotlib import colors
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#!curl http://opendata-dev.web.cern.ch/record/12320/files/TTbar_13TeV_PU50_PixelSeeds_pixelTracksDoublets_h5_file_index.txt -o file_index.txt
#file_list = !(cat file_index.txt)
#print(file_list[0])
remote_data = "./"
data_files = [remote_data + "/" + f for f in os.listdir(remote_data) if "Doublets" in f and f.endswith("h5")]
print(data_files)
data = pd.read_hdf(data_files[0])
print("file read done")
true = (data["label"]==1.0)
fake = (data["label"]==-1.0)
print("Number of doublets: %d"%len(data))
data.head()
sig = data[true]
bkg = data[fake].sample(n=len(sig))
data = pd.concat([sig,bkg]).sample(frac=1.0)
print("New number of doublets: %d"%len(data))

X = data[featureLabs].values
Y = data["label"] == 1.0

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                         algorithm="SAMME",
                         n_estimators=1000)
                         
print ("BDT Training")
t = time()
bdt.fit(X_train,Y_train)

twoclass_output = bdt.decision_function(X_test)
bdt_acc = bdt.score(X_test,Y_test)
print ("BDT accuracy = %.3f"%(bdt_acc))

plot_colors = "br"
plot_step = 0.02
class_names = ["True", "Fake"]

plot_range = (-0.45,0.45)#(twoclass_output.min(), twoclass_output.max())
plt.figure(figsize=(12,9))


for i, n, c in zip([0.0,1.0], class_names, plot_colors):
    plt.hist(twoclass_output[ Y_test == i],
             bins=90,
             range=plot_range,
             facecolor=c,
             label='Class %s' % n,
             alpha=.5,
             edgecolor='k')
    
plt.grid()
ax = plt.gca()
#ax.margins(x=0,y=0,tight=False)
legend = plt.legend(title="Classes",fontsize=15)

plt.setp(legend.get_title(),fontsize=18,fontweight="bold",fontstyle="italic")
plt.title("BDT output score",fontsize=15,fontweight="bold")
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

plt.xlabel("BDT Score",fontsize=15)
plt.ylabel("candidates/0.01",fontsize=15)

test_roc = roc_auc_score(Y_test, twoclass_output)
fp, tp, _ = roc_curve(Y_test,twoclass_output)


i= 1 + i
effFive      = fp[np.where(tp > 0.95)[0][0]]
effTen       = fp[np.where(tp > 0.90)[0][0]]
effTwenty    = fp[np.where(tp > 0.8)[0][0]]
effHalf      = fp[np.where(tp > 0.5)[0][0]]
effSeventy   = fp[np.where(tp > 0.25)[0][0]]
effNinenty   = fp[np.where(tp > 0.1)[0][0]]
effNineFive  = fp[np.where(tp > 0.05)[0][0]]
effNineEight = fp[np.where(tp > 0.02)[0][0]]
effNineNine  = fp[np.where(tp > 0.01)[0][0]]

rejFive      = tp[np.where(fp > 0.95)[0][0]]
rejTen       = tp[np.where(fp > 0.90)[0][0]]
rejTwenty    = tp[np.where(fp > 0.8)[0][0]]
rejHalf      = tp[np.where(fp > 0.5)[0][0]]
rejSeventy   = tp[np.where(fp > 0.25)[0][0]]
rejNinenty   = tp[np.where(fp > 0.1)[0][0]]
rejNineFive  = tp[np.where(fp > 0.05)[0][0]]
rejNineEight = tp[np.where(fp > 0.02)[0][0]]
rejNineNine  = tp[np.where(tp > 0.01)[0][0]]
#thrNinenty   = tr[np.where(fp > 0.1)[0][0]]
#thrNineFive  = tr[np.where(fp > 0.05)[0][0]]


print ("==========================================")
print ("Fake Rejection @ 0.50 Efficiency : %.2f " % (rejHalf))
print ("Fake Rejection @ 0.75 Efficiency : %.2f " % (rejSeventy))
print ("Fake Rejection @ 0.90 Efficiency : %.2f " % (rejNinenty))
print ("Fake Rejection @ 0.95 Efficiency : %.2f " % (rejNineEight))
print ("Fake Rejection @ 0.98 Efficiency : %.2f " % (rejNineFive))
print ("Fake Rejection @ 0.99 Efficiency : %.2f " % (rejNineNine))
print ("==========================================")

print ("Efficiency @ 0.50 Fake Rejection  : %.2f " % (effHalf))
print ("Efficiency @ 0.75 Fake Rejection  : %.2f " % (effSeventy))
print ("Efficiency @ 0.90 Fake Rejection  : %.2f " % (effNinenty))
print ("Efficiency @ 0.95 Fake Rejection  : %.2f " % (effNineEight))
print ("Efficiency @ 0.98 Fake Rejection  : %.2f " % (effNineFive))
print ("Efficiency @ 0.99 Fake Rejection  : %.2f " % (effNineNine))
print ("==========================================")

