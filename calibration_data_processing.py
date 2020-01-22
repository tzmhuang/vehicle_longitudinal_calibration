import sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from surf_function_visualization import surf_function
from math import sqrt

# read in data
dirname = os.path.dirname(os.path.realpath(sys.argv[0]))
filePath = "/data/revised_result.csv"

data = np.genfromtxt(dirname + filePath, delimiter = ',', names = True)


#read in data
folderPath = dirname + "/data/record1/"
fileList = [f for f in os.listdir(folderPath) if f.endswith('.csv')]

dataList = []
for f in fileList:
    data = np.genfromtxt(folderPath + f, delimiter = ',', names = True)
    dataList.append(data)

DATA = np.hstack(dataList)

throttle_data = DATA['ctlthrottle']
brake_data = DATA['ctlbrake'] * -1
cmd_data = throttle_data + brake_data
acc_data = DATA['imu']
speed_data = DATA['vehicle_speed']




# Solve least sqr error problem: order-4 in x and y direction
# * Ax = b
# * A[:,i] = [1 xi xi**2 xi**3 xi**4 yi yi**2 yi**3 yi**4]
# * x = [c ax1 ax2 x3 ax4 ay1 ay2 ay3 ay4]'
# * solution: x = (A'*A)^(-1)* A'*Z, where Z = [z1, z2, z3, ..., zn]


def solve_lstsqr(A, Z):
    ATA = np.matmul(A.T, A)
    inv_ATA = np.linalg.inv(ATA)
    ATZ = np.matmul(A.T, Z)
    return np.matmul(inv_ATA, ATZ)


    
def get_A(x,y):
    data_size = x.shape[0]
    A = np.zeros((data_size,14))
    temp = np.zeros((14))
    for i in range(data_size):
        temp[0] = 1
        temp[1] = x[i]**1
        temp[2] = x[i]**2
        temp[3] = x[i]**3
        temp[4] = x[i]**4
        temp[5] = x[i]**5
        temp[6] = x[i]**6
        temp[7] = x[i]**7
        temp[8] = x[i]**8
        temp[9] = y[i]**1
        temp[10] = y[i]**2
        temp[11] = y[i]**3
        temp[12] = y[i]**4
        temp[13] = y[i]**5

        A[i] = temp
    return A

def root_mean_sqr_error(function, real_x, real_y, real_z):
    data_size = real_x.shape[0]
    error = 0
    print(data_size)
    for i in range(data_size):
        fit_z = function.func_eval(real_x[i], real_y[i])
        print(fit_z)
        err  = fit_z - real_z[i]
        error += (err)**2
    return sqrt(error/data_size)




def data_viz(X,Y,Z):
    # visualizing data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    col_map = plt.get_cmap("jet")
    cNorm = matplotlib.colors.Normalize(vmin=min(Z),vmax=max(Z))
    scalarMap = cm.ScalarMappable(norm = cNorm, cmap = col_map)
    col_vec = [scalarMap.to_rgba(val) for val in Z]
    ax.scatter(X,Y,Z,  c = col_vec)

    ax.set_xlabel('result_speed')
    ax.set_ylabel('result_cmd')    
    ax.set_zlabel('result_acc')

    m = cm.ScalarMappable(cmap=cm.jet)
    m.set_array(Z)
    fig.colorbar(m)

    # plt.show()

if __name__ == "__main__":
    # data_viz()
    # X = data['result_speed']
    # Y = data['result_acc']
    # Z = data['result_cmd']
    header = data.dtype.names

    X = speed_data
    Y = acc_data
    Z = cmd_data
    A = get_A(X, Z)
    print(A.shape)
    print(A)
    coeff = solve_lstsqr(A, Y)
    print(coeff)
    f = surf_function(coeff)
    error = root_mean_sqr_error(f, X, Z, Y)
    print('sqr_error=', error)
    x_range = (np.min(X), np.max(X))
    z_range = (np.min(Z), np.max(Z))
    f.plot(x_range,z_range)
    # plt.hold()
    # data_viz(X,Z,Y)
    plt.show()