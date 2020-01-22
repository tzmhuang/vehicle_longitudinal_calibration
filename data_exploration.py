import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from numpy import linalg as la

cwd = os.getcwd()
data_dir = cwd + '/Calibration_data/BeiBen_halfload_20190426/raw_data'

def ImportDataFromFile(data_dir):
    file_list = [f for f in os.listdir(data_dir) if not f.startswith('.')]
    df_list = {}
    #read in data one by one 
    for f in file_list:
        f_dir = data_dir + '/' + f
        df_list[f] = pd.read_csv(f_dir)
        print('File loaded: ', f_dir)
    return df_list

def DataClean(df, data = 'all'):
    #only select io = 1
    idxbool = df['io'] == 1
    idxbool = idxbool & df['ctlmode'] == 1
    idxbool = idxbool & df['driving_mode'] == 1
    if data == 'throttle':
        idxbool = idxbool & df['throttle_percentage']>0
    elif data == 'brake':
        idxbool = idxbool & df['brake_percentage']>0
    elif data == 'all':
        idxbool = idxbool
    else:
        raise ValueError('Please Specify throttle or brake or all')
    return df[idxbool]


def SimpleExplore(df):
    # plot all columns by time, except time
    for i in range(1,len(df.columns)):
        plt.figure()
        plt.plot(df['time'], df[df.columns[i]],'r-')
        plt.plot(df['time'], df[df.columns[i]], 'bx', markersize= 3)
        plt.title(df.columns[i])
    plt.show()
    return
        
def ThrottleExplore(df, df_compare):
    plt.figure()
    plt.plot(df['time'], df['throttle_percentage'], label = 'throttle%')
    plt.plot(df['time'], df['vehicle_speed'], label = 'speed')
    plt.plot(df['time'], df['engine_rpm'], label = 'rpm')
    plt.plot(df['time'], df[' imu'], label = 'imu')
    plt.plot(df_compare['time'], df_compare[' imu'], label = 'imu2')
    plt.axhline(y = 1, color = 'r', linestyle = '-')
    plt.axhline(y =-1, color = 'r', linestyle = '-')

    plt.xlabel('time')
    plt.ylabel('data')
    plt.legend()

    plt.figure()
    plt.plot(df['vehicle_speed'],df['engine_rpm'],'b-', label = 'rpm vs speed')
    plt.xlabel('speed')
    plt.ylabel('rpm')
    plt.legend()

    plt.figure()
    plt.plot(df['engine_rpm'], df[' imu'],'bx' ,label = 'acc vs rpm')
    plt.xlabel('rpm')
    plt.ylabel('imu')

    plt.plot(df_compare['engine_rpm'], df_compare[' imu'],'rx' ,label = 'acc vs rpm compare')
    plt.plot(df_compare['engine_rpm'], df_compare[' imu'],'r-' ,label = 'acc vs rpm compare')

    return


def CorrelationAnalysis(df):

    plt.matshow(df.corr())
    print(df.corr())

    plt.figure()
    mat = np.array(df)[:,(2,3,4,5,6,7,8,9,10,11,13)]
    print('shape: ', mat.shape)
    U, Sigma, VT= la.svd(mat)
    print('SVD U = ',U)
    print('SVD VT = ', VT[0,:])
    print('Sigma%: ', Sigma/np.sum(Sigma)*100)

    plt.plot(Sigma, label = 'sigma')
    plt.title('svd Sigma')
    plt.legend()


    plt.figure()
    plt.plot(df['time'], df[' imu']/df['engine_rpm'], label = 'rpm/imu')
    return




def DataStandardize(df):
    df_after = df.copy()
    # standardize
    throttle_mean = np.mean(df_after['throttle_percentage'])
    throttle_std = np.std(df_after['throttle_percentage'])
   
    speed_mean = np.mean(df_after['vehicle_speed'])
    speed_std = np.std(df_after['vehicle_speed'])

    rpm_mean = np.mean(df_after['engine_rpm'])
    rpm_std = np.std(df_after['engine_rpm'])

    imu_mean = np.mean(df_after[' imu'])
    imu_std = np.std(df_after[' imu'])

    adjusted_throttle = (df_after['throttle_percentage'] - throttle_mean)
    adjusted_speed = (df_after['vehicle_speed'] - speed_mean)/speed_std
    adjusted_rpm = (df_after['engine_rpm'] - rpm_mean)/rpm_std
    adjusted_imu = (df_after[' imu'] - imu_mean)/imu_std

    df_after['throttle_percentage'] = adjusted_throttle
    df_after['vehicle_speed'] = adjusted_speed
    df_after['engine_rpm'] = adjusted_rpm
    df_after[' imu'] = adjusted_imu
    return df_after



def DataProcess(df): 
    # filter imu data
    df_after = df.copy()
    b,a = signal.butter(5, 1*2/100, 'low')
    imu = signal.filtfilt(b,a,df_after[' imu'])
    print('IMU shape', imu.shape)
    df_after[' imu'] = imu
    mean = np.mean(imu)
    std = np.std(imu)
    return df_after



def CrossDfCompare(df_list):
    
    plt.figure()
    for df in df_list:
        df['time'] = df['time'] - df['time'].iloc[0]
        plt.plot(df['time'], df['engine_rpm'], label = df['throttle_percentage'].iloc[0])
    plt.legend()
    plt.title('rpm compare plot')

    plt.figure()
    for df in df_list:
        df['time'] = df['time'] - df['time'].iloc[0]
        plt.plot(df['time'], df['vehicle_speed'], label = df['throttle_percentage'].iloc[0])
    plt.legend()
    plt.title('speed compare plot')

    plt.figure()
    for df in df_list:
        df['time'] = df['time'] - df['time'].iloc[0]
        plt.plot(df['time'], df[' imu'], label = df['throttle_percentage'].iloc[0])
    plt.legend()
    plt.title('imu compare plot')


    plt.figure()
    for df in df_list:
        df['time'] = df['time'] - df['time'].iloc[0]
        plt.plot(df['time'], df['vehicle_speed']/df['engine_rpm'], label = df['throttle_percentage'].iloc[0])
    plt.legend()
    plt.title('speed/rpm plot')

    plt.figure()
    for df in df_list:
        df['time'] = df['time'] - df['time'].iloc[0]
        plt.plot(df['time'], df[' imu']/df['engine_rpm'], label = df['throttle_percentage'].iloc[0])
    plt.legend()
    plt.title('imu/rpm plot')



    return


   
if __name__ == "__main__":
    
    # First import data
    df_list = ImportDataFromFile(data_dir)
    
    # Taking one set of data for exploration
    df1 = df_list['t30b-45r1.csv']
    df1_clean = DataClean(df1, data = 'throttle')
    df1_standardized = DataStandardize(df1_clean)
    df1_standardize_process = DataProcess(df1_standardized)
    df1_process = DataProcess(df1_clean)
    #SimpleExplore(df1_process)

    #SimpleExplore(df1)
    #ThrottleExplore(df1_standardized, df1_standardize_process)
    CorrelationAnalysis(df1_process) 

    # Multiple throttle comparison
    df2 = df_list['t30b-45r1.csv']
    df3 = df_list['t40b-40r0.csv']
    df4 = df_list['t50b-35r0.csv']
    df5 = df_list['t60b-30r0.csv']
    df6 = df_list['t70t20r0.csv']
    compare_df_list = [df1, df2, df3, df4, df5, df6]
    cleaned_list = [DataClean(df, data = 'throttle') for df in compare_df_list]
    processed_list = [DataProcess(df) for df in cleaned_list]

    CrossDfCompare(processed_list)

    plt.show()
    

