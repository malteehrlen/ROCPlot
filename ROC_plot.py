#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Plot ROC curve and calculate AUC. Input is a file containing all true
   positive parameter values and one containing all true negataive values."""

# Copyright (C) 2016 Biomagnetik Park GmbH - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Author: Malte Ehrlen malteehrlen@biomagnetik.com


import sys


import numpy as np


import matplotlib.pyplot as plt

show=False
pos_file = sys.argv[1]
neg_file = sys.argv[2]

pos = np.genfromtxt(pos_file, delimiter="\t", dtype=None)
neg = np.genfromtxt(neg_file, delimiter="\t", dtype=None)

pos_data = []
for i in pos:
    pos_data.append(i[1])
neg_data = []
for i in neg:
    neg_data.append(i[1])

neg_data = np.array(neg_data)
pos_data = np.array(pos_data)

P = pos_data.shape[0]
N = neg_data.shape[0]
    
def getSensSpec(cutoff):
    FP = 0
    TP = 0
    for i in neg_data:
        if i > cutoff:
            FP += 1
    for i in pos_data:
        if i > cutoff:
            TP += 1
    TP_rate = float(TP)/float(P)
    FP_rate = float(FP)/float(N)
    return TP_rate, FP_rate
    
min_val = np.min(np.concatenate((pos_data, neg_data)))
max_val = np.max(np.concatenate((pos_data, neg_data)))

cutoff_steps = np.arange(min_val, max_val, (max_val - min_val)/1000)

senspec = []

for i in cutoff_steps:
    senspec.append(np.array(getSensSpec(i)))
    
senspec = np.array(senspec)
auc = -1.0*np.trapz(senspec[:,0], senspec[:,1])
plt.figure()
fig = plt.gcf()
fig.set_size_inches(15.5, 15.5, forward=True)

plt.fill_between(senspec[:, 1], 0, senspec[:, 0], facecolor='blue', alpha=0.3)
plt.plot(senspec[:, 1], senspec[:, 0], color='k', linewidth=4)
plt.plot([senspec[0, 1], senspec[-1, 0]], [senspec[0, 1], senspec[-1, 0]], '--', color='dimgray', linewidth=4)
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.xticks(np.arange(0.0, 1.0, 0.1))
plt.yticks(np.arange(0.0, 1.0, 0.1))
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=15)
ax = plt.gca()
ax.grid()
ax.set_aspect('equal')
plt.text(0.8, 0.2, "AUC = "+str(auc)[0:5], fontsize=30, ha="center", va="center", bbox=dict(boxstyle="square", fc="white", ec="black"))
plt.xlabel("False Positive rate", fontsize=30)
plt.ylabel("True Positive rate", fontsize=30)
plt.title("ROC curve", fontsize=40)
if show:
    plt.show()
else:
    plt.savefig("ROC.png")
