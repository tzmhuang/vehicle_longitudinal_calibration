import sys, os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np


STRIDE_FACTOR = 20

def csv_to_calibration_data(path):
    data = np.genfromtxt(path, delimiter = ',', names = True)    
    return data

def plot_calibration_data(calibration_data):
    header = calibration_data.dtype.names
    if len(header) != 3:
        raise ValueError(" Data column number != 3, data must have exactly 3 columns.")
    x_data = calibration_data[header[0]]
    y_data = calibration_data[header[1]]
    z_data = calibration_data[header[2]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Surface plot
    x_num = len(np.unique(x_data))
    y_num = len(np.unique(y_data))

    if x_num * y_num != len(z_data):
        raise ValueError(" Data shape mismatch")

    r_s, c_s = calc_stride(x_num, y_num)
    
    grid_x = x_data.reshape(x_num, y_num)
    grid_y = y_data.reshape(x_num, y_num)

    plot_val = z_data.reshape(grid_x.shape)
    ax.plot_surface(grid_x, grid_y, plot_val,rstride = r_s, cstride = c_s, cmap = cm.coolwarm)

    ax.set_xlabel(header[0])
    ax.set_ylabel(header[1])
    ax.set_zlabel(header[2])

    plt.show()

def calc_stride(x_num, y_num):
    if int(x_num/STRIDE_FACTOR) > 1:
        r_s = int(x_num/STRIDE_FACTOR)
    else:
        r_s = 1
    
    if int(y_num/STRIDE_FACTOR) > 1:
        c_s = int(y_num/STRIDE_FACTOR)
    else:
        c_s = 1
    return r_s, c_s
    

def check_file_exist(path):
    if os.path.isfile(path):
        print("File found: " + path)
    else:
        print("File not found: "  + path)
        sys.exit(0)




if __name__ == '__main__':

    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        print ("Please input file path")
        sys.exit(0)

    check_file_exist(path)
    calibration_data = csv_to_calibration_data(path)
    plot_calibration_data(calibration_data)