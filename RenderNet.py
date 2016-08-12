import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ops
import glob


class RenderNet:

    def __init__(self, image_size=(128, 128), input_size=23, n_iterations=500, batch_size=64):
        self.image_size = image_size
        self.input_size = input_size
        self.n_pixels = self.image_size[0] * self.image_size[1]
        self.n_iterations = n_iterations
        self.batch_size = batch_size

        # Network architecture
        self.img_params = tf.placeholder(tf.float32, shape=[None, self.input_size], name='input')
        self.final_image = tf.placeholder(tf.float32, shape=[None, self.n_pixels], name='final_image')

        self.h1 = ops.linear(self.img_params, self.input_size, 4096, activation=tf.nn.tanh, scope='h1')
        self.prediction = ops.linear(self.h1, 4096, self.n_pixels, scope='prediction')

        self.loss = ops.l2norm(self.prediction, self.final_image)

        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)

    def train(self):
        dataset_files = np.array(glob.glob("data/*.png"))
        dataset_params = np.load("data_params.npy")

        n_files = dataset_params.shape[0]

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

                print "EPOCH ", epoch


    def architecture_check(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
        ops.show_graph_operations()


if __name__ == '__main__':
    rnet = RenderNet()
    # rnet.architecture_check()
    rnet.train()


