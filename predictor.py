# -*- coding: utf-8 -*-

import argparse
import chainer
import chainer.links as L
import chainer.functions as F
import matplotlib.pyplot as plt
import numpy as np
import os 
import collections

from sklearn.metrics import confusion_matrix

import dataset_info as di
from tqdm import tqdm
from trainer import MLP
import json
import datasets as ds
import os
import export_matrix as em

def main():
    parser = argparse.ArgumentParser(description='predictor')
    parser.add_argument('--validationfile', '-v', type=str, default='', help='validation file')
    parser.add_argument('--directory', '-d', type=str, default=None, help='use network/params dir')
    args = parser.parse_args()

    with open(args.directory+'/params.json', 'r') as f:
        params = json.load(f)

    model = L.Classifier(MLP(params['unit'], params['classies']))
    chainer.serializers.load_npz(args.directory+'model.npz', model)

    if not os.path.exists(args.directory+'/predict'):
        os.mkdir(args.directory+'/predict')
        os.mkdir(args.directory+'/predict/raw')
        
    
    datasets = ds.trainingdata(args.validationfile)
    valid = np.asarray(datasets.make_sequences(params['sequencelength']),dtype=np.float32)
    sequences = valid[:,:,:3]
    t = valid[:,-1,3]

    with chainer.using_config('train', False):
        model.predictor.reset_state()
        for i in tqdm(range(params['sequencelength'])):
            seq = sequences[:,i]
            y = model.predictor(seq).array
        result = y.argmax(axis=1)
    print('予想ラベル:{0}'.format(result))
    
    '''
    # ラベル補正
    num = 30
    for i in range(len(result)-num+1):
        c = collections.Counter(result[i:i+num])
        result[i] = c.most_common()[0][0]
    '''
    
    matrix = confusion_matrix(t,result,labels=di.labels())
    em.save_matrix(matrix, args.directory+'predict/',args.validationfile[args.validationfile.rfind('/'):]+'_confusion_matrix.csv', -1, 0,1)
    
    # 波形ファイルを読み込み
    x = np.loadtxt(args.validationfile,delimiter=",", usecols=(range(4)))
    plt.plot(range(len(x)), x[:,0])
    plt.plot(range(len(x)), x[:,1])
    plt.plot(range(len(x)), x[:,2])
    
    # ラベルごとに色付け
    colors = ['m','b','r','y','g','c']
    start = params['sequencelength']
    
    for i in tqdm(range(params['sequencelength'],len(x))):
        c = result[i-params['sequencelength']]
        if c != result[i-params['sequencelength']+1]:
            #print('fill {} {} col {}'.format(start,i,c))
            plt.axvspan(start-0.5, i+0.5, facecolor=colors[c], alpha=0.5)
            start=i+1
    plt.axvspan(start-0.5, i+0.5, facecolor=colors[c], alpha=0.5)
    plt.savefig(args.directory+'/predict/'+args.validationfile[args.validationfile.rfind('/'):]+'predict.pdf')

if __name__ == '__main__':
    main()
