#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import chainer
from chainer import optimizers
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import serializers

import os

import dataset_info as di

class trainingdata():
    def __init__(self,directory):
        # [np.array(CSVFILE1), np.array(CSVFILE2)]
        self.datasets = []

        self.trainsequences = []
        self.validationsequences = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if(file[-4:] == '.csv'):
                    filepath = os.path.join(root, file)
                    self.datasets.append(np.loadtxt(filepath,delimiter=",", usecols=(range(4))))                    
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
        label = []            
        if len(self.validationsequences)!=0:
            print("validation seqences")
            for sequence in self.validationsequences:
                label.append(sequence[-1][-1])
            label = np.asarray(label)
            hist = np.histogram(label,bins=int(label.max())+1,range=(-0.5,label.max()+0.5))
            for i in range(len(hist[0])):
                print(" {} : {}".format(di.label2name(int((hist[1][i]+hist[1][i+1])/2)),hist[0][i]))
        print("train seqences")
        label = []
        for sequence in self.trainsequences:
            label.append(sequence[-1][-1])
        label = np.asarray(label)
        hist = np.histogram(label,bins=int(label.max())+1,range=(-0.5,label.max()+0.5))
        for i in range(len(hist[0])):
            print(" {} : {}".format(di.label2name(int((hist[1][i]+hist[1][i+1])/2)),hist[0][i]))
                
# Network definition
class MLP(chainer.Chain):
    def __init__(self, n_units, n_out, train=True):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.LSTM(None, n_units, lateral_init=chainer.initializers.Normal(scale=0.01))  # n_in -> n_units
            self.l2 = L.Linear(None, n_out, initialW=chainer.initializers.Normal(scale=0.01))  # n_units -> n_ou
            self.train = train
    def reset_state(self):
        self.l1.reset_state()
            
    def __call__(self, x):
        with chainer.using_config('train', self.train):
            h1 = self.l1(F.dropout(x))
#            h1 = self.l1(x)
            y = self.l2(h1)
        return y

# batch単位での誤差を求める
def calculate_loss(model, seq):
    # seq = [[[x,y,z,t], ... ,[x,y,z,t]], [[x,y,z,t], ... ,[x,y,z,t]]]
    seq = np.asarray(seq)
    n_seq, len_seq, len_row = seq.shape
    #特徴数
    n_feature = 3 
    loss = 0
    for i in range(len_seq):
        x = chainer.Variable(np.asarray(seq[:,i:i+1,:n_feature],dtype=np.float32)).reshape(n_seq,n_feature)
        t = chainer.Variable(np.asarray(seq[:,i:i+1,-1],dtype=np.int32)).reshape(n_seq)
        loss = model(x,t)
    y = model.predictor(x).array

    return loss, F.accuracy(y,t)

# batch単位で誤差をbackward
def update_model(model, seq):
    loss, acc = calculate_loss(model,seq)

    model.cleargrads()
    loss.backward()

    loss.unchain_backward()

    optimizer.update()
    return loss, acc

def evaluate(model, seqs):
    clone = model.copy()
    clone.train = False
    clone.predictor.reset_state()
    loss, acc = calculate_loss(clone, seqs)
    return loss, acc
   
if __name__ == '__main__':    
    #学習データdir, 検証データdir, シーケンス長, バッチサイズ, エポック, ユニット数, 結果出力dir
    parser = argparse.ArgumentParser(description='axeltraining on LSTM')
    parser.add_argument('--trainfile', '-t', type=str, default='./kas/', help='training data file path')
    parser.add_argument('--validationfile', '-v', type=str, default=None,
                            help='training validation data file path')
    parser.add_argument('--sequencelength', '-l', type=int, default=50, help='Number of sequence length')
    parser.add_argument('--batchsize', '-b', type=int, default=500, help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=200, help='Number of sweeps over the dataset to train')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=20, help='Number of units')
    args = parser.parse_args()

    if (args.validationfile is None):
        print("Create validation sequences from training file dirs...") 
        datasets = trainingdata(args.trainfile)
        train_seqs, val_seqs = datasets.make_slice_sequences(args.sequencelength,0.1)
        datasets.analyze_sequences()

        print('number of sequences {}'.format(len(train_seqs)+len(val_seqs)))
        print('number of train sequences {}'.format(len(train_seqs)))
        print('number of valid sequences {}'.format(len(val_seqs)))
        
        
    else:
        train_datasets = trainingdata(args.trainfile)
        valid_datasets = trainingdata(args.validationfile)
        train_seqs = train_datasets.make_sequences(args.sequencelength)
        val_seqs = valid_datasets.make_sequences(args.sequencelength)
        print('number of sequences {}'.format(len(train_seqs)+len(val_seqs)))
        print('number of train sequences {}'.format(len(train_seqs)))
                                                 
        train_datasets.analyze_sequences()
        print('number of validation sequences {}'.format(len(val_seqs)))
        valid_datasets.analyze_sequences()

    print("unit:{} batchsize:{} seq_len:{}".format(args.unit,args.batchsize,args.sequencelength))
        
    # modelを作成
    model = L.Classifier(MLP(args.unit, 5))

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # train
    n_batches = len(train_seqs) // args.batchsize
    print('number of batches {}'.format(n_batches))

    print("epoch \ttraining loss      \ttraining accuracy   \tvalidation loss   \tvalidation accuracy")
    for epoch in range(args.epoch):
        np.random.shuffle(train_seqs)
        start = 0
        loss_sum = 0
        acc_sum = 0
        for i in range(n_batches):
            seq = train_seqs[start: start + args.batchsize]
            start += args.batchsize

            loss, acc = update_model(model, seq)
            loss_sum += loss.data
            acc_sum += acc.data

        validation_loss, validation_acc = evaluate(model,val_seqs)
        print("{}\t{}\t{}\t{}\t{}".format(epoch, loss_sum/n_batches, acc_sum/n_batches, validation_loss.data, validation_acc.data))
