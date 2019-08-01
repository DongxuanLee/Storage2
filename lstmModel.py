import  numpy as np
import os
import tensorflow as tf
import random
import re
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector

class LstmRNN(object):
    def __init__(self,sess,sym='Storage',lstm_size=128,num_layers=3,num_steps=10,input_size=1,output_size=1,logs_dir='./log/',plots_dir="./Storage_images/"):
        self.sess=sess
        self.sym=sym
        self.lstm_size=lstm_size
        self.num_layers=num_layers
        self.num_steps=num_steps
        self.input_size=input_size
        self.output_size=output_size
        self.logs_dir=logs_dir
        self.plots_dir=plots_dir
        self.build_graph()

    def build_graph(self):
        self.learning_rate=tf.placeholder(tf.float32,None,name="learning_rate")
        self.keep_prob=tf.placeholder(tf.float32,None,name="keep_prob")
        self.inputs=tf.placeholder(tf.float32,[None,self.num_steps,self.input_size],name="inputs")
        self.targets=tf.placeholder(tf.float32,[None,self.output_size],name="targets")

        def _create_one_cell():
            lstm_cell=tf.contrib.rnn.LSTMCell(self.lstm_size,state_is_tuple=True)
            lstm_cell=tf.contrib.rnn.DropoutWrapper(lstm_cell,output_keep_prob=self.keep_prob)
            return lstm_cell
        cell=tf.contrib.rnn.MultiRNNCell(
            [_create_one_cell() for _ in range(self.num_layers)],
            state_is_tuple = True
        )if self.num_layers >1 else _create_one_cell()
        print("input.shape:",self.inputs.shape)

        val,state_= tf.nn.dynamic_rnn(cell,self.inputs,dtype=tf.float32,scope="dynamic_rnn")

        val = tf.transpose(val,[1,0,2])
        last = tf.gather(val, int(val.get_shape()[0]) - 1, name="lstm_state")
        ws = tf.Variable(tf.truncated_normal([self.lstm_size, self.output_size]), name="weight")
        bias = tf.Variable(tf.constant(0.1, shape=[self.output_size]), name="bias")
        self.pred = tf.matmul(last, ws) + bias

        self.last_sum = tf.summary.histogram("lstm_state", last)
        self.w_sum = tf.summary.histogram("weight", ws)
        self.b_sum = tf.summary.histogram("bias", bias)
        self.pred_summ = tf.summary.histogram("pred", self.pred)

        # self.loss = -tf.reduce_sum(targets * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
        self.loss = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_train")
        self.optim = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss, name="rmsprop_optim")
        #self.optim = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss,name = "adm_optim")

        # accuracy
        # self.var = tf.Variable(0, dtype=tf.float32)
        # self.mean = tf.Variable(0, dtype=tf.float32)
        # self.distance = tf.Variable(0, dtype=tf.float32)
        # # self.cos = tf.Variable(0, dtype=tf.float32)
        # self.reerror = tf.Variable(0,dtype=tf.float32)
        self.mean, self.var = tf.nn.moments(self.pred - self.targets, axes=0)

        self.distance, self.cos, self.reerror = self.compute_distance(self.pred, self.targets), self.compute_cos(self.pred, self.targets),self.compute_reerror(self.pred,self.targets)

        # Separated from train loss.
        self.loss_test = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_test")

        self.loss_sum = tf.summary.scalar("loss_mse_train", self.loss)
        self.loss_test_sum = tf.summary.scalar("loss_mse_test", self.loss_test)
        self.learning_rate_sum = tf.summary.scalar("learning_rate", self.learning_rate)

        self.t_vars = tf.trainable_variables()
        self.saver = tf.train.Saver()

    def train(self, dataset_list, config):
        """
        Args:
            dataset_list (<StockDataSet>)
            config (tf.app.flags.FLAGS)
        """
        # assert len(dataset_list) > 0
        self.merged_sum = tf.summary.merge_all()

        # Set up the logs folder
        self.writer = tf.summary.FileWriter(os.path.join("./Storage_logs", self.model_name))
        self.writer.add_graph(self.sess.graph)
        tf.global_variables_initializer().run()

        test_data_feed = {
            self.keep_prob: 1.0,
            self.inputs: dataset_list.test_x,
            self.targets: dataset_list.test_y,
        }

        global_step = 0
        # 1epoch=num_batches batch
        num_batches = len(dataset_list.train_y) // config.batch_size
        random.seed(time.time())

        print("Start training for Storage_sym:")
        epoch_step = 0
        for epoch in list(range(config.max_epoch)):
            learning_rate = config.init_learning_rate * (
                    config.learning_rate_decay ** max(float(epoch + 1 - config.init_epoch), 0.0)
            )

            for batch_X, batch_y in dataset_list.generate_one_epoch(config.batch_size):
                global_step += 1
                train_data_feed = {
                    self.learning_rate: learning_rate,
                    self.keep_prob: config.keep_prob,
                    self.inputs: batch_X,
                    self.targets: batch_y,
                }
                train_loss, _, train_merged_sum, train_pred, train_mean, train_var, train_distance, train_cos, train_reerror = self.sess.run(
                    [self.loss, self.optim, self.merged_sum, self.pred, self.mean, self.var, self.distance, self.cos, self.reerror],
                    train_data_feed)

                self.writer.add_summary(train_merged_sum, global_step=global_step)
                if np.mod(global_step, 20 / config.input_size) == 1:
                    test_mean, test_var, test_loss, test_pred, test_distance, test_cos,test_reerror = self.sess.run(
                        [self.mean, self.var, self.loss_test, self.pred, self.distance, self.cos,self.reerror], test_data_feed)


                    print("Step:%d [Epoch:%d] [Learning rate: %.6f] train_loss:%.6f test_loss:%.6f" % (
                        global_step, epoch, learning_rate, train_loss, test_loss))
                    print("train:mean:%.6f,var:%.6f,distance:%.6f,cos:%.6f,reerror:%.6f" % (
                        train_mean, train_var, train_distance, train_cos, train_reerror))
                    #Plot samples
                    image_path = os.path.join(self.model_plots_dir, "test:{}_epoch{:02d}_step{:04d}.png".format(
                        self.sym, epoch, epoch_step))
                    image_path1 = os.path.join(self.model_plots_dir, "train:{}_epoch{:02d}_step{:04d}.png".format(
                        self.sym, epoch, epoch_step))
                    sample_preds = test_pred
                    # print(sample_preds)
                    sample_truth = dataset_list.test_y
                    sample_x = sample_preds - sample_truth
                    # self.mean,self.var=tf.nn.moments(sample_x,axes=0)
                    # self.distance,self.cos=self.compute_distance(sample_preds,sample_truth),self.compute_cos(sample_preds,sample_truth)
                    print("test: mean:%.6f,var:%.6f,distance:%.6f,cos:%.6f,reerror:%.6f" % (
                    test_mean, test_var, test_distance, test_cos , test_reerror))

                    # print(sample_truth)
                    self.plot_samples(sample_preds, sample_truth, image_path, Storage_sym=self.sym, multiplier=3)
                    self.plot_samples(train_pred, batch_y, image_path, Storage_sym=self.sym, multiplier=1)

                    self.save(global_step)
            epoch_step += 1
        final_pred, final_loss = self.sess.run([self.pred, self.loss], test_data_feed)

        # Save the final model
        self.save(global_step)
        return final_pred

    @property
    def model_name(self):
        name = "Storage_rnn_lstm%d_step%d_input%d" % (
            self.lstm_size, self.num_steps, self.input_size)

        return name

    @property
    def model_logs_dir(self):
        model_logs_dir = os.path.join(self.logs_dir, self.model_name)
        if not os.path.exists(model_logs_dir):
            os.makedirs(model_logs_dir)
        return model_logs_dir

    @property
    def model_plots_dir(self):
        model_plots_dir = os.path.join(self.plots_dir, self.model_name)
        if not os.path.exists(model_plots_dir):
            os.makedirs(model_plots_dir)
        return model_plots_dir


    def save(self, step):
        model_name = self.model_name + ".model"
        self.saver.save(
            self.sess,
            os.path.join(self.model_logs_dir, model_name),
            global_step=step
        )

    def load(self):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(self.model_logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_logs_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter

        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def plot_samples(self, preds, targets, figname, Storage_sym=None, multiplier=5):
        def _flatten(seq):
            return np.array([x for y in seq for x in y])

        truths = _flatten(targets)
        preds = _flatten(preds) * multiplier
        time = range(len(truths))

        plt.figure(figsize=(12, 6))
        plt.plot(time, truths, label='truth')
        plt.plot(time, preds, label='pred')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("time")
        plt.ylabel("Storage_location")
        plt.ylim((min(truths), max(truths)))
        plt.grid(ls='--')

        if Storage_sym:
            plt.title("Storage_location")

        plt.savefig(figname, format='png', bbox_inches='tight', transparent=True)
        plt.close()

    def compute_cos(self, sample_preds, sample_truth):
        # mod
        sample_preds_norm = tf.sqrt(tf.reduce_sum(tf.square(sample_preds)))
        sample_truth_norm = tf.sqrt(tf.reduce_sum(tf.square(sample_truth)))
        # neiji
        sample = tf.reduce_sum(tf.multiply(sample_preds, sample_truth))
        # cast转换格式浮点数32位
        sample_truth_norm = tf.cast(sample_truth_norm, dtype=tf.float32)
        sample_preds_norm = tf.cast(sample_preds_norm, dtype=tf.float32)
        # tf.multiply为元素各自相乘，并非矩阵乘法
        mul = tf.multiply(sample_preds_norm, sample_truth_norm)
        # 对应元素相除
        cosin = tf.divide(sample, mul)
        return cosin

    def compute_distance(self, sample_preds, sample_truth):
        sample_x = sample_preds - sample_truth
        distance = tf.sqrt(tf.reduce_mean(tf.square(sample_x), 0))
        return distance
    def compute_reerror(self,sample_preds, sample_truth):
        sample_x= tf.abs(sample_preds - sample_truth)
        reerror = tf.reduce_mean(sample_x / sample_truth)
        return reerror
