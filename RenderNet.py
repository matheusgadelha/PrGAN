import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import ops
import glob
import os


class RenderNet:

    def __init__(self, sess=tf.Session(), image_size=(64, 64), input_size=23, n_iterations=50, batch_size=64, lrate=0.001):
        self.image_size = image_size
        self.input_size = input_size
        self.n_pixels = self.image_size[0] * self.image_size[1] * 3  # number of channels
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lrate = lrate
        self.session = sess
        self.base_dim = 1024

        # Network architecture
        with tf.variable_scope("rendernet"):
            self.img_params = tf.placeholder(tf.float32, shape=[batch_size, self.input_size], name='input')
            self.final_image = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], 3], name='final_image')

            self.h1 = ops.linear(self.img_params, self.input_size, 1024, activation=tf.nn.relu, scope='h1')
            self.h2 = tf.reshape(ops.linear(self.h1, 1024, self.base_dim * 4 * 4, activation=tf.nn.relu, scope='h2'),
                                 [-1, 4, 4, self.base_dim])
            self.h3 = tf.nn.relu(ops.deconv2d(self.h2, [self.batch_size, 8, 8, self.base_dim/2], name='h3'))
            self.h4 = tf.nn.relu(ops.deconv2d(self.h3, [self.batch_size, 16, 16, self.base_dim/4], name='h4'))
            self.h5 = tf.nn.relu(ops.deconv2d(self.h4, [self.batch_size, 32, 32, self.base_dim/8], name='h5'))
            self.prediction = tf.nn.tanh(ops.deconv2d(self.h5, [self.batch_size, 64, 64, 3], name='prediction'))

            self.loss = tf.reduce_mean(ops.l2norm_sqrd(self.prediction, self.final_image))

            self.optimizer = tf.train.AdamOptimizer(self.lrate).minimize(self.loss)

            self.saver = tf.train.Saver()

    def train(self):
        dataset_files = glob.glob("data/*.png")
        dataset_files.sort(key=ops.alphanum_key)
        dataset_files = np.array(dataset_files)
        dataset_params = np.load("data_params.npy")

        n_files = dataset_params.shape[0]

        testset_idxs = np.random.choice(range(n_files), self.batch_size)
        test_imgs = ops.load_imgbatch(dataset_files[testset_idxs])
        training_step = 0

        self.session.run(tf.initialize_all_variables())
        for epoch in xrange(self.n_iterations):

            rand_idxs = np.random.permutation(range(n_files))
            n_batches = n_files // self.batch_size

            for batch_i in xrange(n_batches):
                idxs_i = rand_idxs[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
                imgs_batch = ops.load_imgbatch(dataset_files[idxs_i])
                self.session.run(self.optimizer, feed_dict={self.img_params: dataset_params[idxs_i, :],
                                                    self.final_image: imgs_batch})
                training_step += 1

                current_loss = self.session.run(self.loss, feed_dict={self.img_params: dataset_params[testset_idxs, :],
                                                              self.final_image: test_imgs})

                print "Epoch {}/{}, Batch {}/{}, Loss {}".format(epoch + 1, self.n_iterations,
                                                                 batch_i + 1, n_batches, current_loss)

                # ops.show_graph_operations()
                # Save checkpoint
                if training_step % 100 == 0:
                    if not os.path.exists("checkpoint"):
                        print "Checkpoint folder not found. Creating one..."
                        os.makedirs("checkpoint")
                        print "Done."
                    self.saver.save(self.session, 'checkpoint/model.ckpt', global_step=training_step)

    def forward(self, render_params):
        ckpt = tf.train.get_checkpoint_state("checkpoint")
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
        else:
            print "No saved model found. Train the network first."

        img = np.array(self.prediction.eval(session=self.session, feed_dict={self.img_params: render_params}))
        print img.shape
        plt.imshow(img[63,:,:,:])
        plt.show()
        return img

    def architecture_check(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
        ops.show_graph_operations()


if __name__ == '__main__':
    rnet = RenderNet()
    #rnet.architecture_check()
    #data = np.load("data_params.npy")[10000:10064, :]
    rnet.train()
    #rnet.forward(data)


