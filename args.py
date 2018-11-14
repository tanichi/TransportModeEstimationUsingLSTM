#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

#学習データdir, 検証データdir, シーケンス長, バッチサイズ, エポック, ユニット数, 結果出力dir
parser = argparse.ArgumentParser(description='axeltraining on LSTM')
parser.add_argument('--trainfile', '-t', type=str, default='./kas/', help='training data file path')
parser.add_argument('--validationfile', '-v', type=str, default=None,
                        help='training validation data file path')
parser.add_argument('--sequencelength', '-l', type=int, default=50, help='Number of sequence length')
parser.add_argument('--batchsize', '-b', type=int, default=1000, help='Number of images in each mini-batch')
parser.add_argument('--epoch', '-e', type=int, default=200, help='Number of sweeps over the dataset to train')
parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
parser.add_argument('--unit', '-u', type=int, default=20, help='Number of units')
parser.add_argument('--ratio', '-r', type=float, default=0.1, help='validation dataset ratio')
parser.add_argument('--rotate', '-rt', type=int, default=0, help='training dataset rotate')
