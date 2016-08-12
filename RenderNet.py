import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ops
import glob


class RenderNet:

    def __init__(self, image_size=(64, 64), input_size=23, n_iterations=500, batch_size=64, lrate=0.001):
        self.image_size = image_size
        self.input_size = input_size
        self.n_pixels = self.image_size[0] * self.image_size[1] * 3  # number of channels
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lrate = lrate

        # Network architecture
        self.img_params = tf.placeholder(tf.float32, shape=[None, self.input_size], name='input')
        self.final_image = tf.placeholder(tf.float32, shape=[None, self.n_pixels], name='final_image')

        self.h1 = ops.linear(self.img_params, self.input_size, self.n_pixels, activation=tf.nn.tanh, scope='h1')
        self.prediction = ops.linear(self.h1, self.n_pixels, self.n_pixels, scope='prediction')

        self.loss = tf.reduce_mean(ops.l2norm(self.prediction, self.final_image))

        self.optimizer = tf.train.AdamOptimizer(self.lrate).minimize(self.loss)

    def train(self):
        dataset_files = np.array(glob.glob("data/*.png"))
        dataset_params = np.load("data_params.npy")

        n_files = dataset_params.shape[0]

        testset_idxs = np.random.choice(range(n_files), self.batch_size)

        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for epoch in range(self.n_iterations):

                rand_idxs = np.random.permutation(range(n_files))
                n_batches = n_files // self.batch_size

                for batch_i in range(n_batches):
                    idxs_i = rand_idxs[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
                    imgs_batch = ops.load_flatten_imgbatch(dataset_files[idxs_i])
                    sess.run(self.optimizer, feed_dict={self.img_params: dataset_params[idxs_i, :],
                                                        self.final_image: imgs_batch})

                test_imgs = ops.load_flatten_imgbatch(dataset_files[testset_idxs])
                current_loss = sess.run(self.loss, feed_dict={self.img_params: dataset_params[testset_idxs, :],
                                                        self.final_image: test_imgs})
                print "EPOCH ", epoch, ", Loss: ", current_loss


    def architecture_check(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
        ops.show_graph_operations()


if __name__ == '__main__':
    rnet = RenderNet()
    # rnet.architecture_check()
    rnet.train()


