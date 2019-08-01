import os
import pandas as pd
import pprint

import tensorflow as tf
import tensorflow.contrib.slim as slim

from data_read import StorageDataSet
from lstmModel import LstmRNN

flags=tf.app.flags
flags.DEFINE_integer("input_size",1,"Input size[1]")
flags.DEFINE_integer("output_size",1,"Output size[1]")
flags.DEFINE_integer("num_steps",10,"Num of steps[30]")
flags.DEFINE_integer("num_layers", 4, "Num of layer [1]")
flags.DEFINE_integer("lstm_size", 256, "Size of one LSTM cell [128]")
flags.DEFINE_integer("batch_size", 32, "The size of batch data")
flags.DEFINE_float("keep_prob",0.8, "Keep probability of dropout layer. [0.8]")
flags.DEFINE_float("init_learning_rate", 0.01, "Initial learning rate at early stage. [0.001]")
flags.DEFINE_float("learning_rate_decay", 0.99, "Decay rate of learning rate. [0.99]")
flags.DEFINE_integer("init_epoch", 5, "Num. of epoches considered as early stage. [5]")
flags.DEFINE_integer("max_epoch", 100, "Total training epoches. [50]")
flags.DEFINE_string("Storage_symbol", 'Storage', "Target stock symbol [None]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

if not os.path.exists("Storage_logs"):
    os.mkdir("Storage_logs")


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def load_Storage(input_size,output_size, num_steps):
        return StorageDataSet(
                input_size=input_size,
                output_size=output_size,
                num_steps=num_steps)


def main(_):
    pp.pprint(flags.FLAGS.__flags)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        rnn_model = LstmRNN(
            sess,
            lstm_size=FLAGS.lstm_size,
            num_layers=FLAGS.num_layers,
            num_steps=FLAGS.num_steps,
            input_size=FLAGS.input_size,
            output_size=FLAGS.output_size,
        )

        show_all_variables()

        storage_data_list = load_Storage(
            input_size=FLAGS.input_size,
            num_steps=FLAGS.num_steps,
            output_size=FLAGS.output_size,
        )

        if FLAGS.train:
            rnn_model.train(storage_data_list, FLAGS)
        else:
            if not rnn_model.load()[0]:
                raise Exception("[!] Train a model first, then run test mode")


if __name__ == '__main__':
    tf.app.run()
