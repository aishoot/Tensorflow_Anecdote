# 使用tensorflow对数据进行回归预测-用该组三个数据预测下一组三个数据
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import pandas as pd
import requests
import io

# 加载数据
'''
url = 'http://blog.topspeedsnail.com/wp-content/uploads/2016/12/铁路客运量.csv'
ass_data = requests.get(url).content
df = pd.read_csv(io.StringIO(ass_data.decode('utf-8')))
'''
df = pd.read_csv("RailwayVolume.csv")
data = np.array(df['Number(10 thousand)'])
normalized_data = (data - np.mean(data)) / np.std(data)

seq_size = 3
train_x, train_y = [], []  # expand_dims: 在第axis位置增加一个维度
for i in range(len(normalized_data) - seq_size - 1):
    train_x.append(np.expand_dims(normalized_data[i: i + seq_size], axis=1).tolist())
    train_y.append(normalized_data[i + 1: i + seq_size + 1].tolist())

input_dim = 1
X = tf.placeholder(tf.float32, [None, seq_size, input_dim])
Y = tf.placeholder(tf.float32, [None, seq_size])

# train_x: 138 list,每个list中又有list: [[-1.31], [-1.02], [-1.31]]
# train_y:138 list,每个list又有list:[-1.02, -1.31, -1.35]

# regression
def ass_rnn(hidden_layer_size=6):
    W = tf.Variable(tf.random_normal([hidden_layer_size, 1]), name='W')
    b = tf.Variable(tf.random_normal([1]), name='b')

    cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_layer_size)
    outputs, states = tf.nn.dynamic_rnn(cell, dtype=tf.float32, inputs=X)
    W_repeated = tf.tile(tf.expand_dims(W, 0), [tf.shape(X)[0], 1, 1]) #对矩阵进行自身进行复制,第一维复制tf.shape(X)[0]份,第二三维复制1份

    #out = tf.batch_matmul(outputs, W_repeated) + b  # Tensorflow old version
    out = tf.matmul(outputs, W_repeated) + b  #outputs:(?,3,6), W_re:(?,6,1), b:(1,), out:(?,3,1); 三维矩阵乘,第一维度相同
    out = tf.squeeze(out) # 从tensor中删除所有大小是1的维度
    return out


def train_rnn():
    out = ass_rnn()
    loss = tf.reduce_mean(tf.square(out - Y)) # 求取矩阵中所有数的平均值
    train_op = tf.train.AdamOptimizer(learning_rate=0.003).minimize(loss)

    saver = tf.train.Saver(tf.global_variables()) # defaults to saving all variables
    with tf.Session() as sess:
        # tf.get_variable_scope().reuse_variables()
        sess.run(tf.global_variables_initializer())

        for step in range(10000):
            _, loss_ = sess.run([train_op, loss], feed_dict={X: train_x, Y: train_y})
            if step % 10 == 0:
                # 用测试数据评估loss
                print(step, loss_)
        print("Saved the model: ", saver.save(sess, './ass.model'))


def prediction():
    out = ass_rnn()
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        # tf.get_variable_scope().reuse_variables()
        saver.restore(sess, './ass.model')
        prev_seq = train_x[-1]  # <class 'list'>:[[1.77], [2.58], [2.84]]
        predict = []

        for i in range(12):
            next_seq = sess.run(out, feed_dict={X: [prev_seq]})  # next_seq:[1.65, 2.49, 1.84]
            predict.append(next_seq[-1])
            prev_seq = np.vstack((prev_seq[1:], next_seq[-1]))

        plt.figure()
        plt.plot(list(range(len(normalized_data))), normalized_data, color='b')
        plt.plot(list(range(len(normalized_data), len(normalized_data) + len(predict))), predict, color='r')
        plt.savefig("volumePre.png")
        try:
            plt.show()
        except:
            print("Linux not support plt.show().")

# Main Program,每次运行其中之一
#train_rnn()
prediction()
