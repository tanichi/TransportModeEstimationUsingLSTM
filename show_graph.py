# -*- coding: utf-8 -*-
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description='predictor')
    parser.add_argument('--csvfile', '-c', type=str, default='', help='validation file')
    parser.add_argument('--directory', '-d', type=str, default=None, help='use network/params dir')
    args = parser.parse_args()

    
    # 波形ファイルを読み込み
    x = np.loadtxt(args.csvfile,delimiter=",", usecols=(range(3)))

    plt.plot(range(len(x)), x[:,0])
    plt.plot(range(len(x)), x[:,1])
    plt.plot(range(len(x)), x[:,2])

    plt.show()

if __name__ == '__main__':
    main()
