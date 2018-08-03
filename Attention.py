'''
Created on August 4, 2018
@author : hsiaoyetgun (yqxiao)
'''
# coding: utf-8

import tensorflow as tf
from Utils import print_shape

class AttentionLayer(object):
    def __call__(self, p, h):
        w_att = tf.matmul(p, tf.transpose(h, [0, 2, 1]))
        print_shape('w_att', w_att)

        softmax_p = tf.nn.softmax(w_att)
        softmax_h = tf.nn.softmax(tf.transpose(w_att))
        softmax_h = tf.transpose(softmax_h)
        print_shape('softmax_p', softmax_p)
        print_shape('softmax_h', softmax_h)

        p_hat = tf.matmul(softmax_p, h)
        h_hat = tf.matmul(softmax_h, p)
        return p_hat, h_hat

class SelfAttentionLayer(object):
    def __init__(self, rnn_size, attention_size, r, name):
        initializer = tf.contrib.layers.xavier_initializer()
        self.W_s1 = tf.get_variable('W_s1_{}'.format(name), [attention_size, rnn_size * 2], tf.float32, initializer=initializer)
        self.W_s2 = tf.get_variable('W_s2_{}'.format(name), [r, attention_size], tf.float32, initializer=initializer)

    def __call__(self, H):
        # A = softmax(W_s2 * tanh(W_s1 * H.T)) (7)
        Ws1Ht = tf.map_fn(fn = lambda x: tf.matmul(self.W_s1, tf.transpose(x)), elems=H)
        print_shape('Ws1Ht', Ws1Ht)

        e = tf.map_fn(fn = lambda x: tf.matmul(self.W_s2, tf.tanh(x)), elems=Ws1Ht)
        print_shape('e', e)

        A = tf.nn.softmax(e)
        print_shape('A', A)

        # M = A * H (8)
        M = tf.matmul(A, H)
        print_shape('M', M)
        return M, A