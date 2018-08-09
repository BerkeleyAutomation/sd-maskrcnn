import os 
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def set_tf_config():
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        set_session(sess)

def mkdir_if_missing(output_dir):
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except:
            print("Something went wrong in mkdir_if_missing. "
                  "Probably some other process created the directory already.")