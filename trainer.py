#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
from chainer import optimizers
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import serializer

import os
import csv
import datetime
import json

from sklearn.metrics import confusion_matrix
import dataset_info as di
import datasets as ds
import export_graph as eg
import args

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
            # h1 = self.l1(x)
            y = self.l2(F.dropout(h1))
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

    # ネットワークの出力(各クラスの確率)
    y = model.predictor(x).array
    # 確率が第第となる推定クラス
    result = y.argmax(axis=1)
    
    return loss, F.accuracy(y,t), confusion_matrix(t.data,result)

# batch単位で誤差をbackward
def update_model(model, seq):
    loss, acc, mat = calculate_loss(model,seq)
    
    model.cleargrads()
    loss.backward()
    loss.unchain_backward()

    optimizer.update()
    return loss, acc

def evaluate(model, seqs):
    clone = model.copy()
    clone.train = False
    clone.predictor.reset_state()
    loss, acc, mat = calculate_loss(clone, seqs)
    return loss, acc, mat

def print_matrix(matrix):
    print('--------------------------------------------------------')
    for i,row in enumerate(matrix):
        print('{:>6}|'.format(di.label2name(i)), end='')
        for item in row:
            print('{:>7.2%}'.format(item/sum(row)), end='')
        print('|{:>6}|{:.2%}'.format(sum(row),matrix[i][i]/sum(row)))

def save_matrix(matrix,path,epoch,acc):
    with open(path+'confusion_matrix.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(['epoch',epoch,'total accuracy',acc])
        writer.writerow(['#']+di.label_name+['total','accuracy'])
        for i,row in enumerate(matrix):
            writer.writerow([di.label2name(i)]+(row/sum(row)).tolist()+[sum(row),row[i]/sum(row)])
            
if __name__ == '__main__':    
    args = args.parser.parse_args()
    
    if (args.validationfile is None):
        print("Create validation sequences from training file dirs...") 
        datasets = ds.trainingdata(args.trainfile)
        train_seqs, val_seqs = datasets.make_slice_sequences(args.sequencelength,args.ratio)
        datasets.analyze_sequences()

    else:
        train_datasets = ds.trainingdata(args.trainfile)
        valid_datasets = ds.trainingdata(args.validationfile)
        train_seqs = train_datasets.make_sequences(args.sequencelength)
        val_seqs = valid_datasets.make_sequences(args.sequencelength)
                                                 
        train_datasets.analyze_sequences()
        valid_datasets.analyze_sequences()

    print('number of sequences {}'.format(len(train_seqs)+len(val_seqs)))
    print('number of train sequences {}'.format(len(train_seqs)))
    print('number of validation sequences {}'.format(len(val_seqs)))
    
    print("\nunit:{} batchsize:{} seq_len:{}".format(args.unit,args.batchsize,args.sequencelength))
        
    # modelを作成
    model = L.Classifier(MLP(args.unit, 5))

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # train
    n_batches = len(train_seqs) // args.batchsize
    print('number of batches {}'.format(n_batches))

    # 結果出力の準備
    dirname="./result/unit{}_batche{}_seqlen{}_".format(args.unit,args.batchsize,args.sequencelength)
    dirname+=datetime.datetime.now().strftime("%Y%m%d_%H%M")+'/'
    os.mkdir(dirname)
    with open(dirname+'result.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(['training loss','training accuracy','validation loss','validation accuracy'])

    argsdict = {'batchsize':args.batchsize, 'epoch':args.epoch, 'out':args.out, 'ratio':args.ratio,
                'sequencelength':args.sequencelength, 'trainfile':args.trainfile,
                'validationfile':args.validationfile, 'unit':args.unit}
    with open(dirname+"params.json", "w") as f:
        json.dump(argsdict, f, ensure_ascii=False)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    
    print("epoch \tt-loss\tt-acc\tv-loss\tv-acc")

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

        validation_loss, validation_acc, validation_matrix = evaluate(model,val_seqs)
        
        train_loss.append(loss_sum/n_batches)
        train_acc.append(acc_sum/n_batches)
        val_loss.append(validation_loss.data)
        val_acc.append(validation_acc.data)
        
        print("{}\t{:.3f}\t{:.2%}\t{:.3f}\t{:.2%}".format(epoch, train_loss[-1], train_acc[-1], val_loss[-1], val_acc[-1])) 
        with open(dirname+'result.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerow([train_loss[-1],train_acc[-1],val_loss[-1],val_acc[-1]])

        # 最高精度を更新
        if max(val_acc) is val_acc[-1]:
            #print_matrix(validation_matrix)
            save_matrix(validation_matrix,dirname,epoch,val_acc[-1])
            chainer.serializers.save_npz(dirname+'model.npz',model)
    eg.export_graph(dirname,args.epoch,train_loss,train_acc,val_loss,val_acc)
