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
import export_matrix as em
import args
import gc
import copy
import collections

import sys

from memory_profiler import profile

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
def calculate_loss(model, seq, window=None):
    # seq = [[[x,y,z,t], ... ,[x,y,z,t]], [[x,y,z,t], ... ,[x,y,z,t]]]
    seq = np.asarray(seq)
    n_seq, len_seq, len_row = seq.shape
    #特徴数
    n_feature = 3 
    loss = 0
    for i in range(len_seq):
        x = chainer.Variable(np.asarray(seq[:,i,:n_feature],dtype=np.float32)).reshape(n_seq,n_feature)
        t = chainer.Variable(np.asarray(seq[:,i,-1],dtype=np.int32)).reshape(n_seq)
        loss = model(x,t)
    # ネットワークの出力(各クラスの確率)
    y = model.predictor(x).array
    #print(y)
    # 確率が第第となる推定クラス
    result = y.argmax(axis=1)
    matrix = confusion_matrix(t.data,result,labels=di.labels())

    if window is not None:
        corrected_result = corrector(result, window)
        corrected_matrix = confusion_matrix(t.data,corrected_result,labels=di.labels())
        corrected_accuracy = (len(np.array(t.data)) - np.count_nonzero(np.array(t.data)-np.array(corrected_result))) / len(np.array(t.data))
        return loss, F.accuracy(y,t), matrix, corrected_accuracy, corrected_matrix
    return loss, F.accuracy(y,t), matrix

# batch単位で誤差をbackward
def update_model(model, seq):
    loss, acc, mat = calculate_loss(model,seq)
    
    model.cleargrads()
    loss.backward()
    loss.unchain_backward()

    optimizer.update()
    return loss.data, acc.data

def evaluate(model, seqs):
    clone = model.copy()
    clone.train = False
    clone.predictor.reset_state()
    loss, acc, mat, cor_acc, cor_mat = calculate_loss(clone, seqs, 300)

    return loss.data, acc.data, mat, cor_acc, cor_mat

def corrector(result,window):
    num = window
    for i in range(len(result)-num+1):
        c = collections.Counter(result[i:i+num])
        result[i] = c.most_common()[0][0]
    return result

if __name__ == '__main__':
    args = args.parser.parse_args()
    
    if (args.validationfile is None):
        print("Create validation sequences from training file dirs...") 
        datasets = ds.trainingdata(args.trainfile)
        train_seqs, val_seqs = datasets.make_slice_sequences(args.sequencelength,args.ratio)
        datasets.analyze_sequences(train_seqs,val_seqs)
        
    else:
        train_datasets = ds.trainingdata(args.trainfile, args.rotate)
        valid_datasets = ds.trainingdata(args.validationfile)

        train_datasets.make_index(args.sequencelength, args.batchsize)
        valid_datasets.make_index(args.sequencelength, args.batchsize)
        
        val_seqs = valid_datasets.make_sequences(args.sequencelength)
        
    print("\nunit:{} batchsize:{} seq_len:{}".format(args.unit,args.batchsize,args.sequencelength))

    # modelを作成
    model = L.Classifier(MLP(args.unit, 5))

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # train
    n_batches = train_datasets.batches_size
    print('number of batches {}'.format(n_batches))

    # 結果出力の準備
    if args.out is None:
        dirname='./result/'+datetime.datetime.now().strftime("%m-%d_%H-%M")
        if args.rotate is 0:
            dirname+="_unit{}_batche{}_seqlen{}/".format(args.unit,args.batchsize,args.sequencelength)
        else:
            dirname+="_unit{}_batche{}_seqlen{}_rotate/".format(args.unit,args.batchsize,args.sequencelength)
        os.mkdir(dirname)
    else:
        dirname = './'+args.out
    os.mkdir(dirname+'matrixes/')
    os.mkdir(dirname+'matrixes/raw/')
    os.mkdir(dirname+'corrected_matrixes/')
    os.mkdir(dirname+'corrected_matrixes/raw/')
    
    with open(dirname+'result.csv', 'w') as csvfile:
        writer = csv.writer(csvfile, lineterminator='\n')
        writer.writerow(['training loss','training accuracy','validation loss','validation accuracy','corrected accuracy'])

    argsdict = {'batchsize':args.batchsize, 'epoch':args.epoch, 'out':args.out, 'ratio':args.ratio,
                'sequencelength':args.sequencelength, 'trainfile':args.trainfile,
                'validationfile':args.validationfile, 'unit':args.unit, 'classies':di.n_class()}
    with open(dirname+"params.json", "w") as f:
        json.dump(argsdict, f, ensure_ascii=False)

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    cor_accuracy = []
    
    print("epoch \tt-loss\tt-acc\tv-loss\tv-acc")

    for epoch in range(args.epoch):
        train_datasets.shuffle_index()
        loss_sum = 0
        acc_sum = 0
        for i in range(n_batches):
            loss, acc = update_model(model, train_datasets.batch())
            
            loss_sum += loss
            acc_sum += acc

        valid_datasets.batch_index = 0

        validation_loss_sum = []
        validation_acc_sum = []
        validation_weight = []
        validation_matrix = np.zeros((5,5))
        
        cor_validation_acc_sum = []
        cor_validation_matrix = np.zeros((5,5))
        
        for i in range(len(valid_datasets.datasets)):
            batch = valid_datasets.validation_batch()
            validation_loss, validation_acc, matrix, cor_acc, cor_mat = evaluate(model,batch)
            validation_loss_sum.append(validation_loss)
            validation_acc_sum.append(validation_acc)
            validation_weight.append(len(batch))
            validation_matrix = validation_matrix + matrix

            cor_validation_acc_sum.append(cor_acc)
            cor_validation_matrix = cor_validation_matrix + cor_mat
            
        train_loss.append(loss_sum/n_batches)
        train_acc.append(acc_sum/n_batches)
        val_loss.append((np.array(validation_loss_sum) * np.array(validation_weight)).sum() / np.array(validation_weight).sum())
        val_acc.append((np.array(validation_acc_sum) * np.array(validation_weight)).sum() / np.array(validation_weight).sum())
        cor_accuracy.append((np.array(cor_validation_acc_sum) * np.array(validation_weight)).sum() / np.array(validation_weight).sum())
        print("{}\t{:.3f}\t{:.2%}\t{:.3f}\t{:.2%}\t{:.2%}".format(epoch, train_loss[-1], train_acc[-1],
                                                                  val_loss[-1], val_acc[-1], cor_accuracy[-1])) 
        with open(dirname+'result.csv', 'a') as csvfile:
            writer = csv.writer(csvfile, lineterminator='\n')
            writer.writerow([train_loss[-1],train_acc[-1],val_loss[-1],val_acc[-1],cor_accuracy[-1]])

        em.save_matrix(validation_matrix, dirname+'matrixes/',str(epoch)+'_confusion_matrix.csv', epoch, val_acc[-1],1)
        em.save_matrix(cor_validation_matrix, dirname+'corrected_matrixes/',str(epoch)+'_confusion_matrix.csv', epoch, cor_accuracy[-1],1)
        
        # 最高精度を更新
        if max(val_acc) is val_acc[-1]:
            #print_matrix(validation_matrix)
            em.save_matrix(validation_matrix,dirname,'confusion_matrix.csv',epoch,val_acc[-1])
            chainer.serializers.save_npz(dirname+'model.npz',model)
    eg.export_graph(dirname,args.epoch,train_loss,train_acc,val_loss,val_acc,cor_accuracy)

