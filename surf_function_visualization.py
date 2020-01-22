import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


        
class surf_function():
    def __init__(self, coeff):
        self.coeff = coeff

    def func_eval(self, x, y):
        vec = np.array([1, x, x**2, x**3, x**4, x**5,x**6,x**7, x**8,y, y**2, y**3, y**4, y**5])
        return np.dot(self.coeff,vec.T)


    def plot(self, x_range, y_range):
        x_grid = np.linspace(x_range[0], x_range[1], 25)
        y_grid = np.linspace(y_range[0], y_range[1], 25)
        z_val = []
        x_val = []
        y_val = []
        for x in x_grid:
            for y in y_grid:
                result = self.func_eval(x, y)
                x_val += [x]
                y_val += [y]
                z_val += [result]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')

        # print(x_val)
        col_map = plt.get_cmap("jet")
        cNorm = matplotlib.colors.Normalize(vmin=min(z_val),vmax=max(z_val))
        scalarMap = cm.ScalarMappable(norm = cNorm, cmap = col_map)
        col_vec = [scalarMap.to_rgba(val) for val in z_val]
        ax.scatter(x_val, y_val, z_val,  c = col_vec)
        # ax.scatter(real_x, real_y, real_z, c = 'red', marker = 'x')

        ax.set_xlabel('speed')
        ax.set_ylabel('acc')
        ax.set_zlabel('result_cmd')

        m = cm.ScalarMappable(cmap=cm.jet)
        m.set_array(z_val)
        fig.colorbar(m)

        # plt.show()

if __name__ == "__main__":
    
    def surf_function(x, y):
        a1 = 1
        a2 = 1
        a3 = 1
        b1 = 1
        b2 = 1
        constant = 1
        return a1*x**3 + a2*x**2 + a3*x + b1*y**2 + b2*y + constant


    x_grid = np.linspace(0,10,10)
    y_grid = np.linspace(0,10,10)

    z_val = []
    x_val = []
    y_val = []
    for x in x_grid:
        for y in y_grid:
            result = surf_function(x,y)
            x_val += [x]
            y_val += [y]
            z_val += [result]



    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    print(x_val)
    ax.scatter(x_val, y_val, z_val)

    plt.show()
