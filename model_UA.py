import tensorflow as tf
import numpy as np
import pandas as pd
import time
import math
from six.moves import xrange
import scipy.io
import os
import collections
from attention_operation import *
#from metric import *
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class UA(object):
    dic = {}
    def __init__(self, config):
        self.task = config['task']
        self.num_features = config['num_features']
        self.steps = config['steps']
        self.pre_step = config['pre_step']
        self.y_mean = config['y_mean']
        self.y_std = config['y_std']

        self.num_layers = config['num_layers']
        self.hidden_units = config['hidden_units']
        self.embed_size = config['embed_size']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.save_iter = config['save_iter']
        self.max_epoch = config['max_epoch']

        self.train_x = config['train_x']
        self.train_y = config['train_y']
        # self.val_x = config['val_x']
        # self.val_y = config['val_y']
        self.eval_x = config['eval_x']
        self.eval_y = config['eval_y']
        self.train_range = np.array(range(len(config['train_x'])))
        self.test_range = np.array(range(len(config['eval_x'])))
        self.sess = config['sess']

        self.x = tf.placeholder(shape=[None, config['steps'], config['num_features']], dtype=tf.float32, name='data')
        self.y = tf.placeholder(shape=[None, config['pre_step']], dtype=tf.float32, name='labels')
        self.input_keep_prob = tf.placeholder('float')
        self.output_keep_prob = tf.placeholder('float')
        self.state_keep_prob = tf.placeholder('float')
        self.num_sampling = config['num_sampling']
        self.lamb = config['lamb']

    def build_model(self):
        print("Start building a model.")
        def single_cell():
            lstm_cell = tf.contrib.rnn.LSTMCell(self.hidden_units)
            return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell,
                                                 input_keep_prob=self.input_keep_prob, \
                                                 output_keep_prob=self.output_keep_prob,
                                                 state_keep_prob=self.state_keep_prob, \
                                                 dtype=tf.float32
                                                 )

        cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layers)])

        with tf.variable_scope('embedded'):
            self.V = tf.get_variable('v_weight', shape=[self.num_features, self.embed_size], dtype=tf.float32)
        with tf.variable_scope('output_v'):
            #self.out_weight = tf.get_variable('weight', shape=[self.embed_size, 1])
            #self.out_bias   = tf.get_variable('bias', shape=[self.pre_step, 1])
            self.out_weight = tf.get_variable('weight', shape=[self.embed_size, self.pre_step])
            self.out_bias   = tf.get_variable('bias', shape=[self.pre_step])
            self.sigma_weight = tf.get_variable('sigma_weight', shape=[self.hidden_units, 1])
            self.sigma_bias   = tf.get_variable('sigma_bias', shape=[1])

        v_emb = []
        with tf.variable_scope('embedded', reuse=True):
            for _j in range(self.steps):
                self.V = tf.get_variable(name='v_weight')
                embbed = tf.matmul(self.x[:, _j, :], self.V)
                v_emb.append(embbed)
            self.embedded_v = tf.reshape(tf.concat(v_emb, 1), [-1, self.steps, self.embed_size])

        #Reverse embedded_v
        reversed_v_outputs = tf.reverse(self.embedded_v, [1])

        with tf.variable_scope("myrnns_alpha") as scope:
            alpha_rnn_outputs, _ = tf.nn.dynamic_rnn(cell,
                                                     reversed_v_outputs,
                                                     dtype=tf.float32
                                                     )

        with tf.variable_scope("myrnns_beta") as scope:
            beta_rnn_outputs, _ = tf.nn.dynamic_rnn(cell,
                                                    reversed_v_outputs,
                                                    dtype=tf.float32
                                                    )

        #alpha
        alpha_embed_output = attention_op('alpha', alpha_rnn_outputs, self.hidden_units, self.embed_size, self.steps)
        self.rev_alpha_embed_output = tf.reverse(alpha_embed_output, [1])

        #beta
        beta_embed_output = attention_op('beta', beta_rnn_outputs, self.hidden_units, self.embed_size, self.steps)
        self.rev_beta_embed_output = tf.reverse(beta_embed_output, [1])

        # attention_sum 
        c_i = tf.reduce_sum(self.rev_alpha_embed_output * (self.rev_beta_embed_output * self.embedded_v), 1)
        # decoder_input = tf.reduce_mean(self.rev_alpha_embed_output * (self.rev_beta_embed_output * self.embedded_v), 1, keepdims=True)
        # decoder_hidden = c_i
        #
        # #mu
        # gru = tf.contrib.rnn.GRUCell(num_units=self.embed_size)
        # count = self.pre_step
        # #outputs = tf.zeros([self.batch_size, count, self.embed_size], dtype=tf.float32)
        # outputs = []
        # with tf.variable_scope("decoder"):
        #     for i in range(count):
        #         decoder_output, decoder_hidden = tf.nn.dynamic_rnn(gru, decoder_input,initial_state=decoder_hidden,dtype =tf.float32)
        #         decoder_input = decoder_output
        #         outputs.append(decoder_output)
        #
        # outputs = tf.concat(outputs, axis=1)
        # logits = tf.matmul(outputs, self.out_weight) + self.out_bias
        # self.preds = tf.squeeze(logits, axis=2)

        y_mean = tf.reshape(self.y_mean, [1, -1])
        y_std = tf.reshape(self.y_std, [1, -1])

        logits = tf.matmul(c_i, self.out_weight) + self.out_bias
        self.preds = logits * y_std + y_mean

        all_variables = tf.trainable_variables()
        l2_losses = []
        for variable in all_variables:
            variable = tf.cast(variable, tf.float32)
            l2_losses.append(tf.nn.l2_loss(variable))
        regul = self.lamb*tf.reduce_sum(l2_losses)

        self.mse = tf.reduce_mean(tf.square(self.y - self.preds))
        self.loss = self.mse + regul
        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.mae = tf.reduce_mean(tf.abs(self.y - self.preds))
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(self.y - self.preds)))
        self.mape = tf.reduce_mean(tf.abs(self.preds - self.y )/ self.y)

        print ('Done with builing the model.')


    def data_iteration(self, data_x, data_y, is_train=True):
        data_range=None
        if is_train:
            data_range = self.train_range
            random.shuffle(data_range)
        
            batch_len = len(data_range) // self.batch_size
            for i_ in xrange(batch_len):
                b_idx = data_range[self.batch_size*i_:self.batch_size*(i_+1)]

                batch_inputs = np.zeros((self.batch_size, self.steps, self.num_features), np.float32)
                batch_labels = np.zeros((self.batch_size, self.pre_step), np.float32)

                for j_ in range(self.batch_size):
                    inp   = np.copy(data_x[b_idx[j_]])
                    label = np.copy(data_y[b_idx[j_]])
                    batch_inputs[j_] = inp
                    batch_labels[j_] = label
                yield batch_inputs, batch_labels

        else:
            yield data_x, data_y


    def run_epoch(self, ops, data_x, data_y, is_train=True):
        total_loss= []
        total_mae = []
        total_rmse = []
        input_keep_prob = 1.
        output_keep_prob = 1.
        state_keep_prob = 1.

        if is_train:
            input_keep_prob = 0.95
            output_keep_prob = 0.75
            state_keep_prob = 0.95

        for step, (data_in, label_in) in enumerate(self.data_iteration(data_x, data_y, is_train)):
            _, loss, mae, rmse, preds = self.sess.run([ops, self.loss, self.mae, self.rmse, self.preds],
                             feed_dict = {
                                          self.x: data_in,
                                          self.y: label_in,
                                          self.input_keep_prob: input_keep_prob,
                                          self.output_keep_prob: output_keep_prob,
                                          self.state_keep_prob: state_keep_prob,
                                          })
            total_loss.append(loss)
            total_mae.append(mae)
            total_rmse.append(rmse)
        mse_loss = np.mean(total_loss, axis=0)
        MAE = np.mean(total_mae, axis=0)
        RMSE = np.mean(total_rmse, axis=0)

        return mse_loss, MAE, RMSE

    
    def run(self):
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=None)
        #writer = tf.summary.FileWriter('logs', self.sess.graph)
        best_test_mae = float("inf")
        best_test_rmse = float("inf")

        for i_ in range(self.max_epoch):
            train_loss, _, _ = self.run_epoch(self.optimize, self.train_x, self.train_y, is_train=True)
            print(" [*] Epoch: %d, Train loss: %.4f" % (i_+1, train_loss))
            
            _, test_mae, test_rmse = self.run_epoch(tf.no_op(), self.eval_x, self.eval_y, is_train=False)
            if test_rmse < best_test_rmse:
                best_test_mae = test_mae
                best_test_rmse = test_rmse
                print("-----best mae: %.4f, best rmse: %.4f" % (best_test_mae, best_test_rmse))

                save_path = saver.save(self.sess, 'checkpoints_UA/sp500-UA.ckpt')

            print(" --Evaluation mae: %.4f, Evaluation rmse: %.4f" % (test_mae, test_rmse))

            # tf.summary.scalar('loss', train_loss)
            # tf.summary.scalar('mse', test_mae)
            # tf.summary.scalar('rmse', test_rmse)
            # merged = tf.summary.merge_all()
            # result = self.sess.run(merged)
            # writer.add_summary(result, i_)
            print("=======================================================================================")

        print(" --best mae: %.4f, best rmse: %.4f" % (best_test_mae, best_test_rmse))


    def prediction(self, config):
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            #saver = tf.train.import_meta_graph('./checkpoints_UA/sp500-UA.ckpt.meta')
            module_file = tf.train.latest_checkpoint('./checkpoints_UA/')
            saver.restore(sess, module_file)

            #x = tf.placeholder(shape=[206, config['steps'], config['num_features']], dtype=tf.float32)

            _, rmse, mape, pred = sess.run([tf.no_op(), self.rmse, self.mape, self.preds],
                                           feed_dict = {self.x: self.eval_x,
                                              self.y: self.eval_y,
                                              self.input_keep_prob: 1,
                                              self.output_keep_prob: 1,
                                              self.state_keep_prob: 1,})

            test_pred = np.array(pred.squeeze(1))
            test_y = np.array(self.eval_y.squeeze(1))
            train_y = np.array(self.train_y.squeeze(1))
            y_all = np.concatenate([train_y,test_y])

            R1 = pd.Series(test_pred)
            R2 = pd.Series(test_y)
            R = R1.corr(R2)

            print('rmse: {}. mape: {}. R: {}'.format(rmse, mape, R))

            plt.figure(1)
            plt.subplot(2, 1, 1)
            plt.plot(list(range(len(test_pred))), test_pred, color='b', label='predict values')
            plt.plot(list(range(len(test_y))), test_y, color='r', label='true values')
            plt.title('Fitted values of predicted part of SP500 dataset')
            #plt.title('Fitted values of predicted part of CSI300 dataset')
            #plt.title('Fitted values of predicted part of NIKKEI225 dataset')
            plt.legend()
            # plt.savefig("fit.png")
            plt.figure(1)
            plt.subplot(2, 1, 2)
            plt.subplots_adjust(hspace=0.5)
            plt.plot(y_all, label='original series')
            plt.plot([None for _ in range(len(train_y))] + [x for x in test_pred], label='predicted part')
            plt.title('Predicted values among the whole series of SP500 dataset')
            #plt.title('Predicted values among the whole series of CSI300 dataset')
            #plt.title('Predicted values among the whole series of NIKKEI225 dataset')
            plt.legend()
            plt.savefig("./results/SP500-UA-new.eps")
            plt.savefig("./results/SP500-UA-new.jpg")
            #plt.savefig("./results/CSI300-UA-new.eps")
            #plt.savefig("./results/CSI300-UA-new.jpg")
            #plt.savefig("./results/NIK225-UA-new.eps")
            #plt.savefig("./results/NIK225-UA-new.jpg")


