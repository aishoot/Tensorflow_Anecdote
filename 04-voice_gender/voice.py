import os
import requests
import pandas as pd
import numpy as np
import random
import tensorflow as tf  # 0.12
from sklearn.model_selection import train_test_split

'''
# Download the dataset
if not os.path.exists('voice.csv'):
    url = 'http://blog.topspeedsnail.com/wp-content/uploads/2016/12/voice.csv'
    data = requests.get(url).content
    with open('voice.csv', 'wb') as f:
        f.write(data)
'''

voice_data = pd.read_csv('voice.csv')
voice_data = voice_data.values
# Voice features
voices = voice_data[:, :-1] # (3168,20)
labels = voice_data[:, -1:]  # ['male']  ['female'], (3168,1)

# one-hot vector
labels_tmp = []
for label in labels:
    tmp = []
    if label[0] == 'male':
        tmp = [1.0, 0.0]
    else:  # 'female'
        tmp = [0.0, 1.0]
    labels_tmp.append(tmp)
labels = np.array(labels_tmp) # (3168,2)

# Shuffle
voices_tmp = []
lables_tmp = []
index_shuf = [i for i in range(len(voices))]  #0-3167
random.shuffle(index_shuf)

for i in index_shuf:
    voices_tmp.append(voices[i])
    lables_tmp.append(labels[i])
voices = np.array(voices_tmp)  # (3168,20)
labels = np.array(lables_tmp)  # (3168,2)
train_x, test_x, train_y, test_y = train_test_split(voices, labels, test_size=0.1)
# train_x:(2851,20), test_x:(317,20), train_y:(2851,2), test_y:(317,2)


banch_size = 64
n_banch = len(train_x) // banch_size
X = tf.placeholder(dtype=tf.float32, shape=[None, voices.shape[-1]])  # (?,20)
Y = tf.placeholder(dtype=tf.float32, shape=[None, 2])

# 3 layers (feed-forward)
def neural_network():
    w1 = tf.Variable(tf.random_normal([voices.shape[-1], 512], stddev=0.5))  #(20,512)
    b1 = tf.Variable(tf.random_normal([512])) #(512)
    output = tf.matmul(X, w1) + b1  #(?,512)

    w2 = tf.Variable(tf.random_normal([512, 1024], stddev=.5))  #(512,1024)
    b2 = tf.Variable(tf.random_normal([1024]))
    output = tf.nn.softmax(tf.matmul(output, w2) + b2)  #(?,1024)

    w3 = tf.Variable(tf.random_normal([1024, 2], stddev=.5))  #(1024,2)
    b3 = tf.Variable(tf.random_normal([2]))
    output = tf.nn.softmax(tf.matmul(output, w3) + b3)  #(?,2)
    return output


# Train neural network
def train_neural_network():
    output = neural_network()

    #cost = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(output, Y)))  # False
    cost = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output)))
    lr = tf.Variable(0.001, dtype=tf.float32, trainable=False)
    opt = tf.train.AdamOptimizer(learning_rate = lr)
    var_list = [t for t in tf.trainable_variables()]
    train_step = opt.minimize(cost, var_list = var_list)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(200):
            sess.run(tf.assign(lr, 0.001 * (0.97 ** epoch)))  # Reset,改变学习率

            for banch in range(n_banch):
                voice_banch = train_x[banch*banch_size : (banch + 1)*(banch_size)]
                label_banch = train_y[banch*banch_size : (banch + 1)*(banch_size)]
                _, loss = sess.run([train_step, cost], feed_dict={X:voice_banch, Y:label_banch})
                print(epoch, banch, loss)

        # Accuracy
        prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction, dtype=tf.float32))
        accuracy = sess.run(accuracy, feed_dict={X:test_x, Y:test_y})
        print("Accuracy:", accuracy)

# Main Program
if __name__ == "__main__":
    train_neural_network()