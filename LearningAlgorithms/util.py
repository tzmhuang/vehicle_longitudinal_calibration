import os, sys
import numpy as np
import pandas as pd

def ImportDataFromFile(data_dir):
    file_list = [f for f in os.listdir(data_dir) if not f.startswith('.')]
    df_list = {}
    #read in data one by one 
    for f in file_list:
        f_dir = data_dir + '/' + f
        df_list[f] = pd.read_csv(f_dir)
        print('File loaded: ', f_dir)
    return df_list

def ImportCSV(file_dir):
    f = pd.read_csv(file_dir)
    return f
