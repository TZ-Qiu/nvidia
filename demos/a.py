"""
author: zj
file: a.py
time: 17-6-8 
"""


import tensorflow as tf
sess = tf.InteractiveSession()

saver = tf.train.Saver()
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     ckpt = tf.train.get_checkpoint_state('/home/zj/Downloads/DeepNuc-master/demos/demos/nvidia/checkpoints')
#
#     saver.restore(sess, ckpt.model_checkpoint_path)

