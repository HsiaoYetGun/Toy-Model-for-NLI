'''
Created on August 4, 2018
@author : hsiaoyetgun (yqxiao)
Reference : 1. A Structured Self-Attentive Sentence Embedding (ICLR 2017)
            2. A Decomposable Attention Model for Natural Language Inference (EMNLP 2016)
            3. Enhanced LSTM for Natural Language Inference (ACL 2017)
'''
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
from Utils import print_shape
from Attention import *

class Contradict(object):
    def __init__(self, seq_length, n_vocab, embedding_size, hidden_size, rnn_size, attention_size, self_attention_r, n_classes, batch_size, learning_rate, optimizer, l2, lambda_penalty, clip_value):
        # model init
        self._parameter_init(seq_length, n_vocab, embedding_size, hidden_size, rnn_size, attention_size, self_attention_r, n_classes, batch_size, learning_rate, optimizer, l2, lambda_penalty, clip_value)
        self._placeholder_init()

        # model operation
        self.logits = self._logits_op()
        self.loss = self._loss_op()
        self.acc = self._acc_op()
        self.train = self._training_op()

        tf.add_to_collection('train_mini', self.train)

    # init hyper-parameters
    def _parameter_init(self, seq_length, n_vocab, embedding_size, hidden_size, rnn_size, attention_size, self_attention_r, n_classes, batch_size, learning_rate, optimizer, l2, lambda_penalty, clip_value):
        """
        :param seq_length: max sentence length
        :param n_vocab: word nums in vocabulary
        :param embedding_size: embedding vector dims
        :param hidden_size: hidden dims
        :param rnn_size: rnn hidden state dims
        :param attention_size: attention dims
        :param self_attention_r: nums of attention directions
        :param n_classes: nums of output label class
        :param batch_size: batch size
        :param learning_rate: learning rate
        :param optimizer: optimizer of training
        :param l2: l2 regularization constant
        :param lambda_penalty: penalty normalization constant
        :param clip_value: if gradients value bigger than this value, clip it
        """
        self.seq_length = seq_length
        self.n_vocab = n_vocab
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn_size = rnn_size
        self.attention_size = attention_size
        self.self_attention_r = self_attention_r
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.lambda_penalty = lambda_penalty
        self.l2 = l2
        self.clip_value = clip_value

    # placeholder declaration
    def _placeholder_init(self):
        """
        premise_mask: actual length of premise sentence
        hypothesis_mask: actual length of hypothesis sentence
        embed_matrix: with shape (n_vocab, embedding_size)
        dropout_keep_prob: dropout keep probability
        :return:
        """
        self.premise = tf.placeholder(tf.int32, [None, self.seq_length], 'premise')
        self.hypothesis = tf.placeholder(tf.int32, [None, self.seq_length], 'hypothesis')
        self.y = tf.placeholder(tf.float32, [None, self.n_classes], 'y_true')
        self.premise_mask = tf.placeholder(tf.int32, [None], 'premise_mask')
        self.hypothesis_mask = tf.placeholder(tf.int32, [None], 'hypothesis_mask')
        self.embed_matrix = tf.placeholder(tf.float32, [self.n_vocab, self.embedding_size], 'embed_matrix')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    # build graph
    def _logits_op(self):
        m_p, self.A_p, m_h, self.A_h = self._selfAttentiveEncodingBlock('Self_Attentive_Encoding')
        v_p, v_h = self._attentionBlock(m_p, m_h, 'Sentence_Attention')
        logits = self._compositionBlock(v_p, v_h, 'Composition')
        return logits

    # feed forward unit
    def _feedForwardBlock(self, inputs, num_units, scope, isReuse = False, initializer = None):
        """
        :param inputs: tensor with shape (batch_size, seq_length, embedding_size)
        :param num_units: dimensions of each feed forward layer
        :param scope: scope name
        :return: output: tensor with shape (batch_size, num_units)
        """
        with tf.variable_scope(scope, reuse = isReuse):
            if initializer is None:
                initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope('feed_foward_layer1'):
                inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)
                outputs = tf.layers.dense(inputs, num_units, tf.nn.relu, kernel_initializer = initializer)
            with tf.variable_scope('feed_foward_layer2'):
                outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)
                resluts = tf.layers.dense(outputs, num_units, tf.nn.relu, kernel_initializer = initializer)
                return resluts

    # biLSTM unit
    def _biLSTMBlock(self, inputs, num_units, scope, seq_len=None, isReuse=False):
        with tf.variable_scope(scope, reuse=isReuse):
            lstmCell = LSTMCell(num_units=num_units)
            dropLSTMCell = lambda: DropoutWrapper(lstmCell, output_keep_prob=self.dropout_keep_prob)
            fwLSTMCell, bwLSTMCell = dropLSTMCell(), dropLSTMCell()
            output = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwLSTMCell,
                                                     cell_bw=bwLSTMCell,
                                                     inputs=inputs,
                                                     sequence_length=seq_len,
                                                     dtype=tf.float32)
            return output

    # input encoding block
    def _selfAttentiveEncodingBlock(self, scope):
        """
        :param scope: scope name

        embeded_left, embeded_right: tensor with shape (batch_size, seq_length, embedding_size)
        rnn_p, rnn_h: output of biLSTM layer, tensor with shape (batch_size, seq_length, 2 * rnn_size)

        :return: m_premise, m_hypothesis: output of self-attention layer, tensor with shape (batch_size, self_attention_r, 2 * rnn_size)
                 A_premise, A_hypothesis: self attention weights matrix, tensor with shape (batch_size, seq_attention_r, attention_size)
        """
        with tf.device('/cpu:0'):
            self.Embedding = tf.get_variable('Embedding', [self.n_vocab, self.embedding_size], tf.float32)
            self.embeded_left = tf.nn.embedding_lookup(self.Embedding, self.premise)
            self.embeded_right = tf.nn.embedding_lookup(self.Embedding, self.hypothesis)
            print_shape('embeded_left', self.embeded_left)
            print_shape('embeded_right', self.embeded_right)

        with tf.variable_scope(scope):
            rnn_outputs_premise, final_state_premise = self._biLSTMBlock(self.embeded_left, self.rnn_size, 'R', self.premise_mask)
            rnn_outputs_hypothesis, final_state_hypothesis = self._biLSTMBlock(self.embeded_right, self.rnn_size, 'R', self.hypothesis_mask, isReuse = True)

            rnn_p = tf.concat(rnn_outputs_premise, axis=2)
            rnn_h = tf.concat(rnn_outputs_hypothesis, axis=2)
            print_shape('rnn_p', rnn_p)
            print_shape('rnn_h', rnn_h)

            self_attention_layer1 = SelfAttentionLayer(self.rnn_size, self.attention_size, self.self_attention_r, 'premise')
            self_attention_layer2 = SelfAttentionLayer(self.rnn_size, self.attention_size, self.self_attention_r, 'hypothesis')
            m_premise, A_premise = self_attention_layer1(rnn_p)
            m_hypothesis, A_hypothesis = self_attention_layer2(rnn_h)
            print_shape('m_premise', m_premise)
            print_shape('m_hypothesis', m_hypothesis)
            print_shape('A_premise', A_premise)
            print_shape('A_hypothesis', A_hypothesis)

            return m_premise, A_premise, m_hypothesis, A_hypothesis

    # attention block
    def _attentionBlock(self, m_p, m_h, scope):
        """
        :param m_p: output of self-attention layer, tensor with shape (batch_size, self_attention_r, 2 * rnn_size)
        :param m_q: output of self-attention layer, tensor with shape (batch_size, self_attention_r, 2 * rnn_size)
        :param scope: scope name

        a_p, a_h: output of attention layer, tensor with shape (batch_size, self_attention_r, 2 * rnn_size)
        sub_p, sub_h: difference of m_p and a_p, m_h and a_h, tensor with shape (batch_size, self_attention_r, 2 * rnn_size)
        mul_p, mul_h: hadamard product of m_p and a_p, m_h and a_h, tensor with shape (batch_size, self_attention_r, 2 * rnn_size)

        :return: v_p: concat of [m_p, a_p, sub_p, mul_p], tensor with shape (batch_size, self_attention_r, 4 * 2 * rnn_size)
                 v_h: concat of [m_h, a_h, sub_h, mul_h], tensor with shape (batch_size, self_attention_r, 4 * 2 * rnn_size)
        """
        with tf.variable_scope(scope):
            attention_layer = AttentionLayer()
            a_p, a_h = attention_layer(m_p, m_h)
            print_shape('a_p', a_p)

            sub_p = tf.subtract(m_p, a_p)
            sub_h = tf.subtract(m_h, a_h)
            mul_p = tf.multiply(m_p, a_p)
            mul_h = tf.multiply(m_h, a_h)
            print_shape('sub_p', sub_p)
            print_shape('mul_p', mul_p)

            v_p = tf.concat([m_p, a_p, sub_p, mul_p], axis=2)
            v_h = tf.concat([m_h, a_h, sub_h, mul_h], axis=2)
            print_shape('v_p', v_p)

            return v_p, v_h

    # composition block
    def _compositionBlock(self, v_p, v_h, scope):
        """
        :param v_p: concat of [m_p, a_p, sub_p, mul_p], tensor with shape (batch_size, self_attention_r, 4 * 2 * rnn_size)
        :param v_h: concat of [m_h, a_h, sub_h, mul_h], tensor with shape (batch_size, self_attention_r, 4 * 2 * rnn_size)
        :param scope: scope name

        v_mean_p, v_mean_h: self-attentive directions (axis = self_attention_r) average of v_p, v_h, tensor with shape (batch_size, 4 * 2 * hidden_size)
        v_max_p, v_max_h: self-attentive directions (axis = self_attention_r) max value of v_p, v_h, tensor with shape (batch_size, 4 * 2 * hidden_size)
        v: concat of [v_mean_p, v_mean_h, v_max_p, v_max_h], tensor with shape (batch_size, 4 * 4 * 2 * hidden_size)
        ff_outputs: output of feed forward layer, tensor with shape (batch_size, hidden_size)

        :return: y_hat: output of a linear layer, tensor with shape (batch_size, n_classes)
        """
        with tf.variable_scope(scope):
            v_mean_p = tf.reduce_mean(v_p, axis=1)
            v_mean_h = tf.reduce_mean(v_h, axis=1)
            v_max_p = tf.reduce_max(v_p, axis=1)
            v_max_h = tf.reduce_max(v_h, axis=1)
            print_shape('v_mean_p', v_mean_p)
            print_shape('v_max_p', v_max_p)

            v = tf.concat([v_mean_p, v_mean_h, v_max_p, v_max_h], axis=1)
            print_shape('v', v)

            ff_outputs = self._feedForwardBlock(v, self.hidden_size, 'H')
            print_shape('ff_outputs', ff_outputs)

            y_hat = tf.layers.dense(ff_outputs, self.n_classes)
            print_shape('y_hat', y_hat)
            return y_hat

    # calculate classification loss
    def _loss_op(self, l2_lambda=0.0001):
        """
        :param l2_lambda: L2 normalization constant

        AAt_p, AAt_h: product of self attention weight vector (A * At), tensor with shape (batch_size, self_attention_r, self_attention_r)
        batch_I: batch identity matrix, tensor with shape (batch_size, self_attention_r, self_attention_r)
        penalty_p, penalty_h: penalty of premise's self attention vector, hypothesis's self attention vector, tensor with shape (batch_size)
        lambda_penalty: penalty normalization constant
        penalty: penalty of self attention vector, a scalar

        :return: loss: training loss
        """
        with tf.name_scope('cost'):
            AAt_p = tf.matmul(self.A_p, tf.transpose(self.A_p, [0, 2, 1]))
            AAt_h = tf.matmul(self.A_h, tf.transpose(self.A_h, [0, 2, 1]))
            print_shape('AAt_p', AAt_p)

            I = tf.eye(self.self_attention_r)
            batch_I = tf.reshape(tf.tile(I, [tf.shape(self.A_p)[0], 1]), [-1, self.self_attention_r, self.self_attention_r])
            print_shape('batch_I', batch_I)

            penalty_p = tf.square(tf.norm(AAt_p - batch_I, axis=[-2, -1], ord='fro'))
            penalty_h = tf.square(tf.norm(AAt_h - batch_I, axis=[-2, -1], ord='fro'))
            print_shape('penalty_p', penalty_p)

            penalty = tf.reduce_mean((penalty_p + penalty_h) * self.lambda_penalty)
            print_shape('penalty', penalty)

            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            label_loss = tf.reduce_mean(losses, name='loss_val')
            weights = [v for v in tf.trainable_variables() if 'kernel' in v.name]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            loss = label_loss + l2_loss + penalty
        return loss

    # calculate classification accuracy
    def _acc_op(self):
        with tf.name_scope('acc'):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            label_true = tf.argmax(self.y, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    # define optimizer
    def _training_op(self):
        with tf.name_scope('training'):
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            elif self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                ValueError('Unknown optimizer : {0}'.format(self.optimizer))
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        if self.clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
        train_op = optimizer.apply_gradients(zip(gradients, v))
        return train_op