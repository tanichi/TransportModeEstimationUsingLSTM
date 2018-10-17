#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import dataset_info as di

class trainingdata():
    def __init__(self,directory):
        # [np.array(CSVFILE1), np.array(CSVFILE2)]
        self.datasets = []

        self.trainsequences = []
        self.validationsequences = []

        if os.path.isdir(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if(file[-4:] == '.csv' and not file.startswith(".")):
                        filepath = os.path.join(root, file)
                        print(filepath)
                        self.datasets.append(np.loadtxt(filepath,delimiter=",", usecols=(range(4))))
        else:
            self.datasets.append(np.loadtxt(directory,delimiter=",", usecols=(range(4))))
        print('loaded {} csvfiles'.format(len(self.datasets)))

    def make_sequences(self,seq_size):
        self.train_sequences = []
        for dataset in self.datasets:
            n_seq = len(dataset) - seq_size + 1
            for i in range(n_seq):
                self.trainsequences.append(np.asarray(dataset[i:i+seq_size]))
        return self.trainsequences

    def make_slice_sequences(self,seq_size,ratio=0.2):
        self.validationsequences = []
        self.trainsequences = []
        for dataset in self.datasets:
            n_seq = len(dataset) - seq_size + 1
            n_val_seq = int(n_seq * ratio) 
            n_train_seq = n_seq - n_val_seq
            for i in range(n_val_seq):
                self.validationsequences.append(np.asarray(dataset[i:i+seq_size]))
            
            for i in range(n_train_seq):
                self.trainsequences.append(np.asarray(dataset[i+n_val_seq : i+seq_size+n_val_seq]))
                
        return self.trainsequences, self.validationsequences
    
    def analyze_sequences(self):
        print("train seqences")
        label = []
        for sequence in self.trainsequences:
            label.append(sequence[-1][-1])
        label = np.asarray(label)
        hist = np.histogram(label,bins=int(label.max())+1,range=(-0.5,label.max()+0.5))
        for i in range(len(hist[0])):
            print(" {} : {}".format(di.label2name(int((hist[1][i]+hist[1][i+1])/2)),hist[0][i]))
                
        if len(self.validationsequences)!=0:
            label = []
            print("validation seqences")
            for sequence in self.validationsequences:
                label.append(sequence[-1][-1])
            label = np.asarray(label)
            hist = np.histogram(label,bins=int(label.max())+1,range=(-0.5,label.max()+0.5))
            for i in range(len(hist[0])):
                print(" {} : {}".format(di.label2name(int((hist[1][i]+hist[1][i+1])/2)),hist[0][i]))
