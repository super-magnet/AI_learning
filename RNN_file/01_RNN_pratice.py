import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets(r'MNIST_data-bak/', one_hot=True)

learning_rate = 0.001
train_step = 1001
batch_size=1280
display_step = 10

frame_size = 28
sequence_length = 28
hidden_num = 100
n_classes=10

# 定义输入，输出
x = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, sequence_length * frame_size], name='inputx')
y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_classes], name='y_pred')

# 定义网络最后一层的权值
weights = tf.compat.v1.Variable(tf.random.truncated_normal(shape=[hidden_num, n_classes]))
bias = tf.compat.v1.Variable(tf.zeros(shape=[n_classes]))

def RNN(x, weights, bias):
    x = tf.reshape(x, shape=[-1, sequence_length, frame_size])
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num)
    init_state = tf.zeros(shape=[batch_size, rnn_cell.state_size])
    # 其实这是单层RNN网络，对于每一个长度为n的序列[x1,x2,x3...,xn]的每一个xi，都会单独通过一次RNN
    # 的hidden_num个隐藏单元
    output, states = tf.nn.dynamic_rnn(rnn_cell, x, initial_state=init_state)
    #y_ = tf.matmul(output[:, -1, :], weights) + bias
    y_ = tf.matmul(states, weights) + bias
    return y_
y_pred = RNN(x, weights, bias)
loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_function)

correct_pred = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())
step=1
x_test, y_test = mnist.test.next_batch(batch_size)
while step < train_step:
    x_train, y_train = mnist.train.next_batch(batch_size)
    # x_train = tf.reshape(x_train, shape=[batch_size, sequence_length, frame_size])
    _loss, _ = sess.run([loss_function, train_op], feed_dict={x: x_train, y: y_train})
    if step % display_step == 0:
        acc, loss = sess.run([accuracy, loss_function], feed_dict={x: x_test, y: y_test})
        print(step, acc, loss)
    step += 1


