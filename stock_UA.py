from model_UA import *
from data_prepro import *
import tensorflow as tf
import numpy as np
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = {}

# Data info
config['task'] = 'UA_SP500'
config['num_features'] = 0
config['steps'] = 0
config['pre_step'] = 1

# Model info
config['max_epoch'] = 300
config['num_layers'] = 1
config['hidden_units'] = 70
config['embed_size'] = 7
config['lr'] = 0.01
config['batch_size'] = 60
config['save_iter'] = 40
config['num_sampling'] = 30
config['lamb'] = 0.01

def main():
    data_path = 'raw_data/SP500_new.csv'
    target_col = 0
    indep_col = [0, 7]
    win_size = 20
    pre_T = 1
    train_share = 0.9
    is_stateful = False
    normalize_pattern = 2
    generator = getGenerator(data_path)
    datagen = generator(data_path, target_col, indep_col, win_size, pre_T,
                        train_share, is_stateful, normalize_pattern)

    train_x, eval_x, train_y, eval_y, y_mean, y_std = datagen.with_target()

    print(" --- Data shapes: ", np.shape(train_x), np.shape(train_y), np.shape(eval_x), np.shape(eval_y))

    num_features = train_x.shape[2]
    steps = train_x.shape[1]
    pre_step = train_y.shape[1]

    print('shape of train_x:', train_x.shape)

    config['num_features'] = num_features
    config['steps'] = steps
    config['pre_step'] = pre_step
    config['train_x'] = train_x
    config['train_y'] = train_y
    # config['val_x'] = val_x
    # config['val_y'] = val_y
    config['eval_x'] = eval_x
    config['eval_y'] = eval_y
    config['y_mean'] = y_mean
    config['y_std'] = y_std

    #GPU Option
    #gpu_usage = 0.95
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage)
    sess = tf.Session()
    config['sess'] = sess


    with tf.Session() as sess:

        model = UA(config)
        model.build_model()
        model.run()
        model.prediction(config)


if __name__ == '__main__':
    main()
