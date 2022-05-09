#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import sys, argparse
import os
import pandas as pd

def main():

    if not os.path.exists(args.output):
        print("The output path not exist! Create a new folder...\n")
        os.makedirs(args.output)
    if not os.path.exists(args.path):
        print("The input data not exist! Error\n")
        sys.exit()

    funciton(args.path, args.output)


if __name__ == "__main__":

    DATAPATH = './train.fa'
    OUTPATH = './output/'
    parser = argparse.ArgumentParser(description='Manual to the DHS')
    parser.add_argument('-p', '--path', type=str, help=' data', default=DATAPATH)
    parser.add_argument('-o', '--output', type=str, help='output folder', default=OUTPATH)
    args = parser.parse_args()
    print(args.path)
    from train import *
    main()