
"""
author: zj
file: nvidia.py
time: 17-6-8
"""
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import deepnuc.readData as Batch
import deepnuc.dtlayers as dtl
import scipy
import numpy as np
import sys
import signal
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__),os.path.pardir)))



flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS


flags.DEFINE_string('mode','train',"""Options: \'train\' or \'visualize\' """)
flags.DEFINE_string('save_dir','demos/nvidia1',"""Directory under which to place checkpoints""")
flags.DEFINE_string('image_dir','/home/zj/Desktop/save',"""Directory under which to place checkpoints""")
flags.DEFINE_integer('num_epochs',30,""" Number of training epochs """)
flags.DEFINE_integer('num_visualize_save',10000,""" Number of VisualizeBackProp image to save""")
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
    num_visualize_save = FLAGS.num_visualize_save
    epochs = FLAGS.num_epochs
    image_save_dir = FLAGS.image_dir
    save_dir = FLAGS.save_dir
    checkpoint_dir = save_dir + "/checkpoints"
    summary_dir = save_dir + "/summaries"

    if mode == "train":
        batch_size = FLAGS.batch_size
        batch = Batch.readData('/home/zj/Downloads/Autopilot-TensorFlow-master/driving_dataset/',batch_size,4,1000,'data.txt',True)
    else:
        batch_size = 1
        batch = Batch.readData('/home/zj/Downloads/Autopilot-TensorFlow-master/driving_dataset/',batch_size,1,1000,'data.txt',False)

    batch.getFiles()
    image_batch, label_batch = batch.getBatch()
    keep_prob = tf.placeholder(tf.float32, name='drop')
    x_image = dtl.ImageInput(image_batch, pad_size=0,image_shape=[66, 200, 3])

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


    train_vars = tf.trainable_variables()
    loss = tf.reduce_mean(tf.square(tf.subtract(label_batch, y))) + tf.add_n(
        [tf.nn.l2_loss(v) for v in train_vars]) * L2NormConst
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    visualizeBackProp = tf.image.grayscale_to_rgb(nn.visualize_back_prop([batch_size,66,200,1]))

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver()

        imageShow = [tf.concat([image_batch[i,:,:,:]*255,visualizeBackProp[i,:,:,:]*255] ,axis=0) for i in range(batch_size)]
        summary_writer = tf.summary.FileWriter(summary_dir, sess.graph)


        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        if mode == "train":
            tf.summary.image("image", imageShow, max_outputs=4)
            tf.summary.scalar("loss", loss)
            merged_summary_op = tf.summary.merge_all()

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

            if ckpt and ckpt.model_checkpoint_path:
                print ("Checkpoint restored")
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print ("No checkpoint found on ", ckpt)

            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

            try:
                for epoch in range(epochs):
                    for i in range(int(batch.size / batch.batch_size)):
                        if coord.should_stop():
                            print ("stop")
                            break

                        sess.run(train_step, feed_dict={keep_prob: 1.0})

                        summary = merged_summary_op.eval(feed_dict={keep_prob: 1.0})
                        summary_writer.add_summary(summary, i)
                        loss_value = loss.eval(feed_dict={keep_prob: 1.0})
                        save_path = saver.save(sess, checkpoint_dir + os.sep + "model_ckpt")
                        print ("loss: {}, {} training iterations passed".format(loss_value,i))

                    print ("finish epoch:{}".format(epoch))
            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads)
        else:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            tf.summary.image("image", imageShow, max_outputs=1)
            merged_summary_op = tf.summary.merge_all()

            if ckpt and ckpt.model_checkpoint_path:
                print ("Checkpoint restored")
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print ("No checkpoint found on ", ckpt)
            try:

                for i in range(num_visualize_save):
                    if coord.should_stop():
                        print ("stop")
                        break
                    scipy.misc.imsave('image_save_dir' +'/' +str(i) +'.png',imageShow[0].eval(feed_dict={keep_prob: 1.0}))
                    summary = merged_summary_op.eval(feed_dict={keep_prob: 1.0})
                    summary_writer.add_summary(summary)

            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads)
                # x = nn.visualize_back_prop([batch_size,66,200])
            # aaa = tf.placeholder(tf.float32, shape=[None, 66, 200, 3])
            # image = scipy.misc.imresize(
            #     scipy.misc.imread('/home/zj/Downloads/Autopilot-TensorFlow-master/driving_dataset/' + '1.jpg')[ -150:], [66, 200])
            # b = sess.run(x,feed_dict={aaa:[image],keep_prob: 1.0})
            # plt.imshow(b[0],plt.cm.gray)
            # plt.show()



if __name__ == '__main__':
    tf.app.run()