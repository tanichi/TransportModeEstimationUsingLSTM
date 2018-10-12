# -*- coding: utf-8 -*-

import argparse
import chainer
import chainer.links as L

import matplotlib.pyplot as plt
import numpy as np
import os 
import collections

from trainer import MLP
import json
import datasets as ds

def main():
    parser = argparse.ArgumentParser(description='predictor')
    parser.add_argument('--validationfile', '-v', type=str, default='', help='validation file')
    parser.add_argument('--directory', '-d', type=str, default=None, help='use network/params dir')
    args = parser.parse_args()

    with open(args.directory+'/params.json', 'r') as f:
        params = json.load(f)

    model = L.Classifier(MLP(params['unit'], params['classies']))
    chainer.serializers.load_npz(args.directory+'model.npz', model)

    datasets = ds.trainingdata(args.validationfile)
    sequences = np.asarray(datasets.make_sequences(params['sequencelength']),dtype=np.float32)[:,:,:3]
    
    with chainer.using_config('train', False):
        model.predictor.reset_state()
        for i in range(params['sequencelength']):
            seq = sequences[:,i]
            y = model.predictor(seq).array
        result = y.argmax(axis=1)
    print('予想ラベル:{0}'.format(result))

    
    # ラベル補正
    num = 30
    for i in range(len(result)-num+1):
        c = collections.Counter(result[i:i+num])
        result[i] = c.most_common()[0][0]

    # 波形ファイルを読み込み
    x = np.loadtxt(args.validationfile,delimiter=",", usecols=(range(3)))
    plt.plot(range(len(x)), x[:,0])
    plt.plot(range(len(x)), x[:,1])
    plt.plot(range(len(x)), x[:,2])

    # ラベルごとに色付け
    colors = ['m','b','r','y','g','c']
    for i in range(params['sequencelength'],len(x)+1):
        plt.axvspan(i-0.5, i+0.5, facecolor=colors[result[i-params['sequencelength']]], alpha=0.5)
    plt.show()

if __name__ == '__main__':
    main()
