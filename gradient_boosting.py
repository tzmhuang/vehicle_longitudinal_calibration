import os, sys
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
import util

#set recursion limit
sys.setrecursionlimit(5000)

#Boosted tree based on sklearn tree
class BoostedTreeRegressor():
    def __init__(self):
        self.treeList = []
        return
    
    def fit(self, X, y, max_iter = 100):
        self.X = X
        self.y = y
        self.max_iter = max_iter
        self.residualList = []
        #initial fit
        t = tree.DecisionTreeRegressor(max_depth=10)
        t.fit(self.X, self.y)
        self.treeList.append(t)
        pred = t.predict(self.X)
        residual = self.y - pred
        self.residualList.append(np.sum(residual**2)/self.X.shape[0])
        self.treeList = self.recursive_call(self.treeList, residual)

    def gen_simple_tree(self, max_depth):
        t = tree.DecisionTreeRegressor(max_depth = max_depth)
        return t
    
    def recursive_call(self, tL,residual):
        if len(tL) >= self.max_iter:
            return tL
        else:
            t = self.gen_simple_tree(max_depth = 1)
            t.fit(self.X, residual)
            pred = t.predict(self.X)
            tL.append(t)
            self._show_progress(len(tL))
            residual = residual - pred
            self.residualList.append(np.sum(residual**2)/self.X.shape[0])  #
            tL = self.recursive_call(tL, residual)
            return tL

    def predict(self, X):
        if not self.treeList:
            raise ValueError("empty treeList")
        else:
            boosted_pred = np.zeros((X.shape[0]))
            for t in self.treeList:
                tree_pred = t.predict(X)
                tree_pred = np.array(tree_pred)
                boosted_pred += tree_pred
        return boosted_pred
    
    def get_residual(self, tL):
        return
    
    def _show_progress(self,l):
        progress = round((l/self.max_iter)*100,2)
        if l % 100 < 1:
            print ("Progress: {0}".format(progress))


class BaggingTreeRegressor():
    def __init__(self):
        self.treeList = []
        return
    
    def fit(self, X, y, max_iter = 100):
        self.X = X
        self.y = y
        self.max_iter = max_iter
        self.residualList = []
        self.bagging_loop(self.treeList, sample_size = 500)


    def gen_simple_tree(self, max_depth):
        t = tree.DecisionTreeRegressor(max_depth = max_depth)
        return t
    
    def bagging_loop(self, tL, sample_size):
        residual = 99999
        for i in range(self.max_iter):
            self._show_progress(i)
            if residual < 0.01:
                print('early termination')
                return tL
            sample_ind = np.random.choice(range(self.X.shape[0]), sample_size)
            sampled_x = self.X[sample_ind]
            sampled_y = self.y[sample_ind]
            t = self.gen_simple_tree(5)
            t.fit(sampled_x, sampled_y)
            tL.append(t)


    def predict(self, X):
        if not self.treeList:
            raise ValueError("empty treeList")
        else:
            bagged_pred = np.zeros((X.shape[0]))
            for t in self.treeList:
                tree_pred = t.predict(X)
                tree_pred = np.array(tree_pred)
                bagged_pred += tree_pred
            mean_bagged_pred = bagged_pred / len(self.treeList)
        return mean_bagged_pred
    
    def get_residual(self, tL):
        return
    
    def _show_progress(self,l):
        progress = round((l/self.max_iter)*100,2)
        if l % 100 < 1:
            print ("Progress: {0}".format(progress))



if __name__ == "__main__":
    dirname = os.path.dirname(os.path.realpath(sys.argv[0]))
    folderName = dirname + '/data_old/record1/'
    DATA = util.read_csv(folderName)

    throttle_data = DATA['ctlthrottle']
    brake_data = DATA['ctlbrake'] * -1
    cmd_data = throttle_data + brake_data
    acc_data = DATA['imu']
    speed_data = DATA['vehicle_speed']

    print (speed_data.shape)
    print (acc_data.shape)
    print (cmd_data.shape)

    test_X = np.concatenate((speed_data.reshape(-1,1), cmd_data.reshape(-1,1)), axis = 1)
    print(test_X.shape)
    # clf = tree.DecisionTreeRegressor(max_depth=20)    
    # clf.fit(test_X, acc_data)
    # pred = clf.predict(test_X)

    # bt = BoostedTreeRegressor()
    # bt.fit(test_X, acc_data, max_iter= 3000)
    # pred = bt.predict(test_X)
    # print(pred)

    # residual_list = bt.residualList

    # plt.plot(residual_list)
    # plt.show()

    bagt = BaggingTreeRegressor()
    bagt.fit(test_X, acc_data, max_iter= 3000)
    pred = bagt.predict(test_X)


    error = util.root_mean_sqr_error(pred, acc_data)
    print(error)
    x_range = (np.min(speed_data), np.max(speed_data))
    z_range = (np.min(cmd_data), np.max(cmd_data))
    util.surface_plot(bagt,x_range,z_range )


