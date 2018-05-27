import tensorflow as tf
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np

training_data = pd.read_csv("trainingData.csv", header=0)  # 把第一行當作索引,第一行不作为正式数据
# print(training_data.head())  # training_data: <class 'tuple'>: (19937, 529)
# array和asarray都可将结构数据转化为ndarray,但主要区别是当数据源是ndarray时,array仍然会copy出一个副本,占用新的内存,但asarray不会
train_x = scale(np.asarray(training_data.ix[:, 0:520])) # scale数据缩放: (X-mean)/std,计算时对每个属性/每列分别进行
train_y = np.asarray(training_data["BUILDINGID"].map(str) + training_data["FLOOR"].map(str))
train_y = np.asarray(pd.get_dummies(train_y))  # one-hot编码

test_dataset = pd.read_csv("validationData.csv", header=0)
test_x = scale(np.asarray(test_dataset.ix[:, 0:520]))
test_y = np.asarray(test_dataset["BUILDINGID"].map(str) + test_dataset["FLOOR"].map(str))
test_y = np.asarray(pd.get_dummies(test_y))

# train_x:(19937, 520); train_y:(19937, 13); test_x:(1111, 520); test_y:(1111, 13)

output = train_y.shape[1]  # 13
X = tf.placeholder(tf.float32, shape=[None, 520])  # 网络输入
Y = tf.placeholder(tf.float32, [None, output])  # 网络输出(?,13)


def neural_networks():
    """
    tf.truncated_normal:截断正太分布函数,产生正太分布的值如果与均值的差值大于两倍的标准差,那就重新生成
    tf.constant: 创建一个常量tensor
    """
    # --------------------------- Encoder ------------------------- #
    e_w_1 = tf.Variable(tf.truncated_normal([520, 256], stddev=0.1))
    e_b_1 = tf.Variable(tf.constant(0.0, shape=[256]))
    e_w_2 = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1))
    e_b_2 = tf.Variable(tf.constant(0.0, shape=[128]))
    e_w_3 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.1))
    e_b_3 = tf.Variable(tf.constant(0.0, shape=[64]))
    # --------------------------- Decoder  ------------------------ #
    d_w_1 = tf.Variable(tf.truncated_normal([64, 128], stddev=0.1))
    d_b_1 = tf.Variable(tf.constant(0.0, shape=[128]))
    d_w_2 = tf.Variable(tf.truncated_normal([128, 256], stddev=0.1))
    d_b_2 = tf.Variable(tf.constant(0.0, shape=[256]))
    d_w_3 = tf.Variable(tf.truncated_normal([256, 520], stddev=0.1))
    d_b_3 = tf.Variable(tf.constant(0.0, shape=[520]))
    # ---------------------------- DNN  --------------------------- #
    w_1 = tf.Variable(tf.truncated_normal([64, 128], stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.0, shape=[128]))
    w_2 = tf.Variable(tf.truncated_normal([128, 128], stddev=0.1))
    b_2 = tf.Variable(tf.constant(0.0, shape=[128]))
    w_3 = tf.Variable(tf.truncated_normal([128, output], stddev=0.1))
    b_3 = tf.Variable(tf.constant(0.0, shape=[output]))
    #################################################################
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(X, e_w_1), e_b_1))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, e_w_2), e_b_2))
    encoded = tf.nn.tanh(tf.add(tf.matmul(layer_2, e_w_3), e_b_3))
    layer_4 = tf.nn.tanh(tf.add(tf.matmul(encoded, d_w_1), d_b_1))
    layer_5 = tf.nn.tanh(tf.add(tf.matmul(layer_4, d_w_2), d_b_2))
    decoded = tf.nn.tanh(tf.add(tf.matmul(layer_5, d_w_3), d_b_3))
    layer_7 = tf.nn.tanh(tf.add(tf.matmul(encoded, w_1), b_1))  # !!!从encoder部分引出
    layer_8 = tf.nn.tanh(tf.add(tf.matmul(layer_7, w_2), b_2))
    out = tf.nn.softmax(tf.add(tf.matmul(layer_8, w_3), b_3))
    return (decoded, out)  # decoded:(?,520); out:(?,13)


# 训练神经网络
def train_neural_networks():
    decoded, predict_output = neural_networks()
    """
    tf.pow:两个数组对应数的幂
    tf.reduce_mean(sum类似):不指定第二个参数,所有元素取平均值; 指定第二个参数为为0,则第一维元素取平均值,即每一列求平均值,第二个参数为1,第二维取平均
    tf.log: 计算e的ln, 两输入以第二输入为底
    tf.cast: 将x的数据格式转化成对应的dtype
    """
    us_cost_function = tf.reduce_mean(tf.pow(X - decoded, 2))
    s_cost_function = -tf.reduce_sum(Y * tf.log(predict_output))
    us_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(us_cost_function)
    s_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(s_cost_function)

    correct_prediction = tf.equal(tf.argmax(predict_output, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Start
    training_epochs = 20  # 训练次数
    batch_size = 10  # 一个batch大小
    total_batches = training_data.shape[0]  # 19937

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # ------------ Training Autoencoders - Unsupervised Learning ----------- #
        # autoencoder是一种非监督学习算法,他利用反向传播算法,让目标值等于输入值
        for epoch in range(training_epochs):  # 20
            epoch_costs = np.empty(0)
            for b in range(total_batches):  # 19937,有问题,需要修改
                offset = (b * batch_size) % (train_x.shape[0] - batch_size)  # ???
                batch_x = train_x[offset:(offset + batch_size), :]  # ???, 感觉这两行代码有问题,batch_size可参考voice.py

                _, c = sess.run([us_optimizer, us_cost_function], feed_dict={X: batch_x})
                epoch_costs = np.append(epoch_costs, c)
            print("Epoch: ", epoch, " Loss: ", np.mean(epoch_costs))  # 只显示一个batch的误差会使界面有失酷炫
        print("------------------------------------------------------------------")

        # ---------------- Training NN - Supervised Learning ------------------ #
        for epoch in range(training_epochs):
            epoch_costs = np.empty(0)
            for b in range(total_batches):
                offset = (b * batch_size) % (train_x.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :]
                batch_y = train_y[offset:(offset + batch_size), :]
                _, c = sess.run([s_optimizer, s_cost_function], feed_dict={X: batch_x, Y: batch_y})
                epoch_costs = np.append(epoch_costs, c)

            accuracy_in_train_set = sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
            accuracy_in_test_set  = sess.run(accuracy, feed_dict={X: test_x,  Y: test_y})
            print("Epoch: ", epoch, " Loss: ", np.mean(epoch_costs), " Accuracy: ", accuracy_in_train_set,
                  ' ',accuracy_in_test_set)

# Main Program
train_neural_networks()
