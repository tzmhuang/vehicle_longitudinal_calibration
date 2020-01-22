import os, sys
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import util

#CONFIGS
model_name_ = 'throttle_3LdenseNN_softmax_MSE'
dir_path_ = os.path.dirname(os.path.realpath(__file__))
data_relative_path_ = '../Data/BeiBen_halfload_20190426/throttle/'
log_path_ = dir_path_ + '/log/' + model_name_
model_path_ = dir_path_ + '/trained_models/'+ model_name_ + '.ckpt'
#x_column_ = ['vehicle_speed', 'engine_rpm', 'throttle_percentage', ' pitch', 'quasi_gear']
#x_column_ = ['vehicle_speed', 'throttle_percentage', 'quasi_gear']
x_column_ = ['vehicle_speed', 'throttle_percentage']
y_column_ = [' imu']
input_shape_ = (1,len(x_column_))
output_shape_ = (1,1)


#CREATE LOG DIR IF NOT EXIST
if not os.path.exists(log_path_):
    os.makedirs(log_path_)


#UTIL FUNCTIONS
def DataSelection(data):
    x = data[x_column_]
    y = data[y_column_]
    return x,y

def DataStandardize(data):
    for c in data.columns:
        data[c] = (data[c] - np.mean(data[c]))
    return data



def LoadTrainData():
    data_path = dir_path_ + '/' + data_relative_path_ + 'train.csv'
    data = util.ImportCSV(data_path)
    x,y = DataSelection(data)
    #x = DataStandardize(x)
    return x.values, y.values

def LoadTestData():
    data_path = dir_path_ + '/' + data_relative_path_ + 'test.csv'
    data = util.ImportCSV(data_path)
    x,y = DataSelection(data)
    #x = DataStandardize(x)
    return x.values, y.values



#NN FUNCTIONS
def NN_DenseSoftmax(x, W, b, name):
    with tf.name_scope(name):
        a = tf.nn.softmax(tf.add(tf.matmul(x,W), b))
    return a

def NeuralNet(x, weights, biases):
    layer_1 = NN_DenseSoftmax(x, weights['layer1'], biases['layer1'], name='Layer_1')
    layer_2 = NN_DenseSoftmax(layer_1, weights['layer2'], biases['layer2'], name='Layer_2')
    layer_3 = NN_DenseSoftmax(layer_2, weights['layer3'], biases['layer3'], name='Layer_3')
    with tf.name_scope('outlayer'):
        outlayer = tf.matmul(layer_3, weights['out']) + biases['out']
    return outlayer

def DecendingLearningRate(epoch):
    if epoch < 5:
        return 0.001
    elif epoch < 15:
        return 0.0001
    else:
        return 0.00001


#VARIABLE INIT
with tf.name_scope("Input"):
    X = tf.placeholder(tf.float32, name='input', shape=[None, input_shape_[1]])
    Y_ = tf.placeholder(tf.float32, name='flag')


r_num_ = input_shape_[1]
c_num_ = 8
o_num_ = 1

weights={
    'layer1':tf.Variable(tf.truncated_normal([r_num_, 8], stddev=tf.sqrt(2/r_num_))),
    'layer2':tf.Variable(tf.truncated_normal([8, 8], stddev=tf.sqrt(2/c_num_))),
    'layer3':tf.Variable(tf.truncated_normal([8, 8], stddev=tf.sqrt(2/c_num_))),
    'out':tf.Variable(tf.random_normal([8, 1], stddev=tf.sqrt(2/c_num_)))
    }

biases={
    'layer1':tf.Variable(0.0),
    'layer2':tf.Variable(0.0),
    'layer3':tf.Variable(0.0),
    'out':tf.Variable(0.0)
    }

#GRAPH BUILDING
model_output = NeuralNet(X, weights, biases)
beta = 0.001/4 * 0
with tf.name_scope('Loss_Function'):
    MSE = tf.reduce_mean(tf.square(model_output - Y_))
    L2_1 = tf.nn.l2_loss(weights['layer1'])
    L2_2 = tf.nn.l2_loss(weights['layer2'])
    L2_3 = tf.nn.l2_loss(weights['layer3'])
    L2_o = tf.nn.l2_loss(weights['out'])
    L2_loss = beta*(L2_1 + L2_2 + L2_3 + L2_o)
    cost = MSE + L2_loss


#DEFINE TRAINING PROCESS 
global_step = tf.Variable(0, name='global_step', trainable=False)
n_epochs_ = 10000
display_step_ = 100
#learning_rate_ = tf.placeholder(tf.float32, name='learning_rate')
learning_rate_ = tf.train.exponential_decay(0.0005, global_step, 1000, 0.9, staircase=True)
train = tf.train.AdamOptimizer(learning_rate = learning_rate_).minimize(cost, global_step=global_step)
init = tf.global_variables_initializer()


#PREPARE TRAINING PREPROCESS
data_x, data_y = LoadTrainData()
data_size_ = data_x.shape[0]
batch_size_ = data_size_
n_batch_ = int(data_size_/batch_size_)
saver = tf.train.Saver()

Error = []
Step = []
LR = []


#START PROCESS
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs_):
        idx = np.arange(data_size_)
        np.random.shuffle(idx)
        for batch in range(n_batch_):
            batch_idx = idx[batch*batch_size_:(batch+1)*batch_size_]
            batch_x = data_x[batch_idx,:]
            batch_y = data_y[batch_idx,:]
            #lr = DecendingLearningRate(epoch)
            #sess.run(train, feed_dict={X:batch_x, Y_:batch_y, learning_rate_:lr})
            sess.run(train, feed_dict={X:batch_x, Y_:batch_y})
            if (batch) % display_step_==0 or batch==0:
                lr = sess.run(learning_rate_)
                error, loss=sess.run([MSE,cost], feed_dict={X:batch_x, Y_:batch_y})
                Error.append(error)
                Step.append(sess.run(global_step))
                LR.append(lr)
                sys.stdout.write("Epoch: {0}, loss: {1:2f}, Batch MSE: {2:2f}, StepSize: {3}\r".format(epoch, loss, error, lr))
                sys.stdout.flush()
    print("Done.")
    save_path = saver.save(sess, model_path_, global_step=global_step) 
    print("Model saved in file: {}".format(save_path))

#VISUALIZE TRAINING RESULT

plt.figure()
plt.plot(Step, Error, 'r-', label='MSE')
plt.legend()
plt.title('Training Error')

plt.figure()
plt.plot(Step, LR, 'r-', label='step size')
plt.legend()
plt.title('Step Decay')



#TEST PROCESS
with tf.Session() as sess:
    sess.run(init)
    print(model_path_)
    #saver.restore(sess, model_path_)
    saver.restore(sess, save_path)
    print(save_path)
    print('Model restored from file: {}'.format(save_path))
    test_x, test_y = LoadTestData()
    test_accuracy_ = sess.run(MSE, feed_dict={X:test_x, Y_:test_y})
    print('Testing Accuracy: {}'.format(test_accuracy_))
    
    #GENERATE CALIBRATION GRID
    max_v = np.max(data_x[:,0])
    min_v = np.min(data_x[:,0])
    max_cmd = np.max(data_x[:,1])
    min_cmd = np.min(data_x[:,1])

    v_values = np.arange(min_v, max_v, 0.2)
    v_mean = np.mean(v_values)
    print('v_mean: ', v_mean)
    cmd_values = np.arange(min_cmd, max_cmd, 0.2)
    cmd_mean = np.mean(cmd_values)
    feed_data = []

    for v in v_values:
        for cmd in cmd_values:
            feed_data.append([v,cmd])
    feed_data = np.array(feed_data)
    
    print(feed_data.shape)
    a_values = sess.run(model_output, feed_dict={X:feed_data})

    grid_x = feed_data[:,0].reshape((len(v_values), len(cmd_values)))
    grid_y = feed_data[:,1].reshape((len(v_values), len(cmd_values)))
    plot_val = a_values.reshape(grid_x.shape)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(feed_data[:,0], a_values, feed_data[:,1])
    ax.plot_surface(grid_x, grid_y, plot_val, cmap = cm.coolwarm)
    ax.set_xlabel('v')
    ax.set_ylabel('cmd')
    ax.set_zlabel('a')



plt.show()




            







    
















