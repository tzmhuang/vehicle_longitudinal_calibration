import os, sys
import numpy as np
import pandas as pd
import data_exploration as de

cwd = os.getcwd()
data_dir = cwd + '/Calibration_data/BeiBen_halfload_20190426/raw_data'


def ProcessData(dataframe, data = 'all'):
    cleaned = de.DataClean(dataframe, data = data)
    processed = de.DataProcess(cleaned)
    return processed


def ReadProcessCombineData(data_dir):
    df_list = de.ImportDataFromFile(data_dir)
    DF = pd.concat(df_list)
    throttle_DF = ProcessData(DF, data = 'throttle')
    brake_DF = ProcessData(DF, data = 'brake')
    # add speed/rpm data
    throttle_quasi_gear =throttle_DF['vehicle_speed']/throttle_DF['engine_rpm']
    brake_quasi_gear =brake_DF['vehicle_speed']/brake_DF['engine_rpm']
    throttle_DF['quasi_gear'] = throttle_quasi_gear
    brake_DF['quasi_gear'] = brake_quasi_gear
    return {'throttle':throttle_DF, 'brake':brake_DF}


def TrainTestSeperation(dataframe, train_test_ratio, seed):
    np.random.seed(seed)
    N = dataframe.shape[0]
    n_train = int(N*train_test_ratio)
    n_test = N - n_train
    rand_idx = np.arange(N)
    np.random.shuffle(rand_idx)
    idx_train = rand_idx[0:n_train]
    idx_test = rand_idx[n_train:]
    df_train = dataframe.iloc[idx_train]
    df_test = dataframe.iloc[idx_test]
    return df_train, df_test


def OutputAsCsv(dataframe, filedir):
    dataframe.to_csv(filedir)
    return     

if __name__ == "__main__":
    Data = ReadProcessCombineData(data_dir)
    throttle_train_data, throttle_test_data = TrainTestSeperation(Data['throttle'], 0.8, 2019)
    brake_train_data, brake_test_data = TrainTestSeperation(Data['brake'], 0.8, 2019)

    throttle_dir = './Data/BeiBen_halfload_20190426/throttle/'
    brake_dir = './Data/BeiBen_halfload_20190426/brake/'

    if not os.path.exists(throttle_dir):
        os.makedirs(throttle_dir)
    if not os.path.exists(brake_dir):
        os.makedirs(brake_dir)

    OutputAsCsv(throttle_train_data,throttle_dir+'train.csv')
    OutputAsCsv(throttle_test_data,throttle_dir+'test.csv')
    print('Throttle Dataset Processed Successfully. DIR = ', throttle_dir)
    
    
    OutputAsCsv(brake_train_data,brake_dir+'train.csv')
    OutputAsCsv(brake_test_data,brake_dir+'test.csv')
    print('Brake Dataset Processed Successfully. DIR = ', brake_dir)
    



    



    

