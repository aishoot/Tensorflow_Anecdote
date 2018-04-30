# 利用深度学习对离散数据作回归
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 使用numpy生成200个随机点，从-0.5到0.5生成200个点
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]  # np.newaxis:增加一个轴,None的一个别名(200,1)
noise = np.random.normal(0, 0.02, x_data.shape)  #生成随机噪声,正态分布
y_data = np.square(x_data) + noise  # shape: (200,1), x_data中每个数平方后加上对应维度的noise.训练目标

# 定义placeholder
x = tf.placeholder(tf.float32, [None, 1])  #行不确定,只有一列
y = tf.placeholder(tf.float32, [None, 1])

# 定义神经网络中间层
Weights_L1 = tf.Variable(tf.random_normal([1, 10]))  #1行10列
biases_L1 = tf.Variable(tf.zeros([1, 10])) # (1,10)
Wx_plus_b_L1 = tf.matmul(x, Weights_L1) + biases_L1  # (?, 10)
L1 = tf.nn.tanh(Wx_plus_b_L1)  # (?, 10)

# 定义神经网络输出层
Weights_L2 = tf.Variable(tf.random_normal([10, 1]))
biases_L2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_L2 = tf.matmul(L1, Weights_L2) + biases_L2  #(?,10)*(10,1)+(1,1)
prediction = tf.nn.tanh(Wx_plus_b_L2)  # (?, 1)

# 优化
loss = tf.reduce_mean(tf.square(y - prediction))  # 损失函数
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)  # 使用梯度下降法训练

# 赋值运行
with tf.Session() as sess:
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2001):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
        #print(sess.run([Weights_L1, biases_L1]))
        #print(Weights_L2, biases_L2)

    # 获得预测值
    prediction_value = sess.run(prediction, feed_dict={x: x_data})
    # 画图
    plt.figure()
    plt.scatter(x_data, y_data)  #样本点
    plt.plot(x_data, prediction_value, 'r-', lw=5)  #红色,实线,线宽5
    plt.savefig("fit.png")
    plt.show()