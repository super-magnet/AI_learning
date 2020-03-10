import tensorflow as tf
import numpy as np

'''
   TensorFlow中的RNN的API主要包括以下两个路径：
      1）tf.nn.rnn_cell（主要定义RNN的几种常见的cell）
      2）tf.nn（RNN中的辅助操作）
'''
# 一RNN中的cell
# 基类（最顶级的父类）：tf.nn.rnn_cell.RNNCell()
# 最基础的RNN的实现：tf.nn.rnn_cell.BasicRNNCell()
# 简单的LSTM cell实现：tf.nn.rnn_cell.LSTMCell()
# RGU cell实现：tf.nn.rnn_cell.RGUCell()
# 多层RNN结构网络的实现：tf.nn.rnn_cell.MultiRNNCell()

# 创建cell
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)
print(cell.state_size)
print(cell.output_size)
# #
# # # shape=[4, 64] 表示每次输入4个样本，每个样本有64个特征
# inputs = tf.compat.v1.placeholder(dtype=tf.float32, shape=[4, 64])
# # 给定RNN的初始状态
# s0 = cell.zero_state(4, tf.float32)
# print(s0.shape)
# #
# # # 对于t=1时刻传入输入和state0，获取结果值
# output, s1 = cell(inputs, s0)
# print(output.shape)
# print(s1.shape)

# 定义LSTM_cell
# lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=128)
# # # shape=[4, 64]表示每次输入4个样本，每个样本有64个特征
# inputs= tf.compat.v1.placeholder(tf.float32, shape=[4, 48])
# # 给定初始状态
# s0 = lstm_cell.zero_state(4, tf.float32)
# print(s0.h.shape)
# print(s0.c.shape)
# # 对于t=1时刻传入输入和state0，获取结果值
# output, s1 = lstm_cell(inputs, s0)
# print(output.get_shape())
# print(s1.h.shape)
# print(s1.c.shape)
