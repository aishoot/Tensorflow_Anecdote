import os
import tensorflow as tf
from mnist import read_data_sets

if not os.path.exists("MNIST_data"):
    os.mkdir("MNIST_data")
mnist = read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()

""""
函数:
tf.InteractiveSession():可先构建一个session再定义操作operation
tf.Session():需要在会话构建之前定义好全部操作operation再构建会话
tf.truncated:如果x的取值在区间(μ-2σ,μ+2σ)之外则重新进行选择(截断正态分布)
tf.random_normal:从正态分布中输出随机值
步长(Strides): [1, stride, stride, 1]
过滤器移动的步长，第一位和第四位一般恒定为1，第二位指水平移动时候的步长，第三位指垂直移动的步长。strides = [1, stride, stride, 1].
池化和卷积后大小padding方式阅读:http://blog.sina.com.cn/s/blog_53dd83fd0102x356.html
tf.equal:对比这两个矩阵或者向量的相等的元素,如果是相等的那就返回True,反正返回False

"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # 补完零后满足卷积核的扫描,就是same; 把刚才不足以扫描的元素位置抛弃掉,就是valid方式
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # strides:卷积时在图像每一维的步长

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])  # [batch, in_height, in_width, in_channels]

# Conv1 Layer
W_conv1 = weight_variable([5, 5, 1, 32])  # filter:[卷积核的高度,卷积核的宽度,图像通道数,卷积核个数]
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 输出[?,28,28,32]
h_pool1 = max_pool_2x2(h_conv1)  # 输出[?,14,14,32]

# Conv2 Layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # [?,14,14,64]
h_pool2 = max_pool_2x2(h_conv2)  # [?,7,7,64]

W_fc1 = weight_variable([7 * 7 * 64, 1024])  #
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # [?,1024]

keep_prob = tf.placeholder(tf.float32)  # float类型,每个元素被保留下来的概率
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # softmax映射到(0,1), [?,10]

test1 = tf.log(y_conv)  # (?,10)
test2 = y_ * test1 # (?,10), 注意!!! Tensorflow中常量constant的"*"和numpy一样对应点乘,而不是矩阵乘
test3 = -tf.reduce_sum(test2, reduction_indices=[1])  # (?,)
cross_entropy = tf.reduce_mean(test3)
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 数据格式转化


tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)  # 每一步迭代都会加载50个训练样本
    if i % 1000 == 0:  # 1000整数倍代时输出结果, 只是为了查看此时准确率,并没有最小化accuracy
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        # 上句写法与此同: sess.run(accuracy, {x:mnist.test.images,y_: mnist.test.labels})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
