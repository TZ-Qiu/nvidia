
"""
author: zj
file: nvidia.py
time: 17-6-8
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import driving_data
import deepnuc.dtlayers as dtl
import scipy
import numpy as np
from deepnuc.formatplot import Formatter
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))



flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS


flags.DEFINE_string('mode','train',"""Options: \'train\' or \'visualize\' """)
flags.DEFINE_string('save_dir','demos/nvidia',"""Directory under which to place checkpoints""")

flags.DEFINE_integer('num_iterations',50,""" Number of training iterations """)
flags.DEFINE_integer('num_visualize',10,""" Number of samples to visualize""")
flags.DEFINE_integer('batch_size',25,""" Number of samples to visualize""")

sess = tf.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.trainable_variables()

def custom(y):
    return tf.atan(y,name='output')

def main(_):

    stride_1 = [1, 1, 1, 1]
    stride_2 = [1, 2, 2, 1]
    mode = FLAGS.mode
    num_visualize = FLAGS.num_visualize
    num_iterations = FLAGS.num_iterations
    batch_size = FLAGS.batch_size

    save_dir = FLAGS.save_dir
    checkpoint_dir = save_dir + "/checkpoints"
    summary_dir = save_dir + "/summaries"

    x = tf.placeholder(tf.float32, [None, 66, 200, 3], name='input')
    keep_prob = tf.placeholder(tf.float32, name='drop')
    x_image = dtl.ImageInput(x, image_shape=[66, 200, 3])

    conv1 = dtl.Conv2d(x_image, filter_shape=[5, 5, 3, 24], strides = stride_2 , padding = 'VALID', name='conv1')
    relu1 = dtl.Relu(conv1, name='relu1')

    conv2 = dtl.Conv2d(relu1, filter_shape=[5, 5, 24, 36], strides=stride_2, padding='VALID', name='conv2')
    relu2 = dtl.Relu(conv2, name='relu2')

    conv3 = dtl.Conv2d(relu2, filter_shape=[5, 5, 36, 48], strides=stride_2, padding='VALID', name='conv3')
    relu3 = dtl.Relu(conv3, name='relu3')

    conv4 = dtl.Conv2d(relu3, filter_shape=[3, 3, 48, 64], strides=stride_1, padding='VALID', name='conv4')
    relu4 = dtl.Relu(conv4, name='relu4')

    conv5 = dtl.Conv2d(relu4, filter_shape=[3, 3, 64, 64], strides=stride_1, padding='VALID', name='conv5')
    relu5 = dtl.Relu(conv5, name='relu5')

    flat = dtl.Flatten(relu5)

    fc1 = dtl.Linear(flat , 1164, name='fc1')
    fc1drop = dtl.Dropout(fc1 , keep_prob=keep_prob, name='fc1drop')

    fc2 = dtl.Linear(fc1drop, 100, name='fc2')
    fc2drop = dtl.Dropout(fc2, keep_prob=keep_prob, name='fc2drop')

    fc3 = dtl.Linear(fc2drop, 50, name='fc3')
    fc3drop = dtl.Dropout(fc3, keep_prob=keep_prob, name='fc3drop')

    fc4 = dtl.Linear(fc3drop, 10, name='fc4')
    fc4drop = dtl.Dropout(fc4, keep_prob=keep_prob, name='fc4drop')

    output = dtl.Linear(fc4drop , 1, name='fc5',custom=custom)

    nn = dtl.Network(x_image, [output])

    y = nn.forward()
    y_ = tf.placeholder(tf.float32, shape=[None, 1])

    train_vars = tf.trainable_variables()
    loss = tf.reduce_mean(tf.square(tf.subtract(y_, y))) + tf.add_n(
        [tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)


    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver()
        merged_summary_op = tf.summary.merge_all()

        if mode == "train":
            summary_writer = tf.summary.FileWriter(summary_dir,sess.graph)
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            for i in range(num_iterations):
                xs, ys = driving_data.LoadTrainBatch(batch_size)
                if i % 10 == 0:
                    print (i, "training iterations passed")
                    loss_value = loss.eval(feed_dict={x: xs, y_: ys, keep_prob: 1.0})
                    print ("step:{}  loss:{}".format(i , loss_value))
                sess.run(train_step, feed_dict={x: xs, y_: ys, keep_prob:1.0})
                summary = merged_summary_op.eval(feed_dict={x: xs, y_: ys, keep_prob: 1.0})
                summary_writer.add_summary(summary, i)
                if i % 25 == 0:
                    ckpt_name = "model_ckpt"
                    save_path = saver.save(sess, checkpoint_dir + os.sep + ckpt_name)
                    print "save"
            ckpt_name = "model_ckpt"
            save_path = saver.save(sess, checkpoint_dir + os.sep + ckpt_name)
            print "done"
            # ckpt_name = "model_ckpt"
            # save_path = saver.save(sess, checkpoint_dir + os.sep + ckpt_name)
            # print("Model saved in file: %s" % save_path)
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            # f = open('/home/zj/Desktop/data.txt', 'w')
            # for i in range(2000):
            #     full_image = scipy.misc.imread(
            #         "/home/zj/Downloads/Autopilot-TensorFlow-master/driving_dataset/" + str(i) + ".jpg", mode="RGB")
            #     image = scipy.misc.imresize(full_image[-150:], [66, 200]) / 255.0
            #
            #     print y.eval(feed_dict={x: [image], keep_prob: 1.0})[0][0]
            #     f.write(str(y.eval(feed_dict={x: [image], keep_prob: 1.0})[0][0]* 180.0 / scipy.pi)+'\n')
            #     # plt.imshow(full_image)
            #     # plt.show()
            # f.close()
            xs, ys = driving_data.LoadTrainBatch(1)
            # plt.imshow(xs[0])
            # plt.show()
            rj_final_op = tf.multiply(y, y_)
            r_input, rj_final_val = sess.run([nn.relevance_backprop(rj_final_op), rj_final_op],
                                             feed_dict={x: xs, y_: ys, keep_prob: 1.0})

            r_input_img = np.squeeze(r_input )

            r_input_sum = np.sum(r_input)

            print "Rj final {}, Rj sum {}".format(np.sum(rj_final_val), r_input_sum)

            # utils.visualize(r_input[:,2:-2,2:-2],utils.heatmap,'deeptaylortest_'+str(i)+'_.png')

            # Display original input
            # plt.imshow(np.reshape(np.asarray(batch_xs),(28,28)))

            yguess_mat = sess.run(y,
                                  feed_dict={x: xs, y_: ys,keep_prob:1.0})
            # yguess = yguess_mat.tolist()
            # yguess = yguess[0].index(max(yguess[0]))
            # actual = ys[0].tolist().index(1.)
            #
            # print ("Guess:", (yguess))
            # print ("Actual:", (actual))

            # Display relevance heatmap
            # fig, ax = plt.subplots()
            # im = ax.imshow(r_input_img, cmap=plt.cm.Reds, interpolation='nearest')
            # ax.format_coord = Formatter(im)
            plt.imshow(r_input_img )
            plt.show()
if __name__ == '__main__':
    tf.app.run()