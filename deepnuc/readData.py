"""
author: zj
file: readData.py
time: 17-6-23 
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np


class readData():
    def __init__(self,path,batch_size,thread_nums,capacity,label_file_name,shuffle):
        """
        read image and label with tf.queue
        :param path: label and image path where saved
        :param batch_size:
        :param thread_nums: queue thread count
        :param capacity: queue size
        :param label_file_name: label file name
        """
        self.batch_size = batch_size
        self.path = path
        self.thread_nums = thread_nums
        self.capacity = capacity
        self.label_file_name = label_file_name
        self.images = []
        self.labels = []
        self.size = 0
        self.shuffle = shuffle
    def getFiles(self):
        """
        get files, include images and labels
        :return: image list and label list
        """
        with open(self.path +'/'+ self.label_file_name) as f:
            for line in f:
                self.images.append(self.path + line.split()[0])
                self.labels.append(line.split()[1])

        temp = list(zip(self.images,self.labels))
        if(self.shuffle):
            np.random.shuffle(temp) # shuffle there better than tensorflow shuffle

        self.size = len(self.images)
        self.images, self.labels = zip(*temp)

        self.labels = [float(i) for i in self.labels]


    def getBatch(self):
        """
        get batch from folder
        :return: image batch and label batch
        """

        input_queue = tf.train.slice_input_producer([self.images, self.labels],shuffle=False)
        label = tf.atan(input_queue[1])
        # image = input_queue[0]
        image = tf.read_file(input_queue[0])
        image = tf.image.decode_jpeg(image,channels=3)
        image = tf.image.crop_to_bounding_box(image,100,0,66,200)
        image = tf.image.per_image_standardization(image)

        image_batch, label_batch = tf.train.batch([image, label], batch_size=self.batch_size
                                                  ,num_threads=self.thread_nums, name='input_batch')

        return image_batch, label_batch

    def getSize(self):
        return self.size

if __name__ == '__main__':
    """
    before test, edit getBatch function, make sure image = input_queue[1]
    """
    test = readData('/home/zj/Downloads/DeepNuc-master/data/',16,4,2000,'data.txt',False)
    test.getFiles()
    image_batch, label_batch = test.getBatch()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    x = 0
    test2 = []
    try:
        for i in np.arange(41):

            if coord.should_stop():
                print ('stop')
                break
            img, label = sess.run([image_batch, label_batch])

            for j in np.arange(test.batch_size):
                x+=1
                print('x: {} image: {}'.format(x,img[j]))
                test2.append(img[j])


    except tf.errors.OutOfRangeError:
        print('done!')
    finally:
        coord.request_stop()
    # test2.sort()
    [print(i) for i in test2]
    coord.join(threads)
