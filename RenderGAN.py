import tensorflow as tf
import numpy as np
import ops
import glob
import os
from RenderNet import RenderNet


class RenderGAN:

    def __init__(self, sess=tf.Session(), image_size=(64, 64), z_size=10,
                 n_iterations=50, batch_size=64, lrate=0.002, d_size=64):

        self.image_size = image_size
        self.n_pixels = self.image_size[0] * self.image_size[1] * 3  # number of channels
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lrate = lrate
        self.session = sess
        self.base_dim = 512
        self.d_size = 64
        self.z_size = z_size

        self.rendernet = RenderNet(sess=self.session, image_size=self.image_size,
                                   batch_size=self.batch_size)

        self.g_bn0 = ops.batch_norm(name='g_bn0')
        self.g_bn1 = ops.batch_norm(name='g_bn1')
        self.g_bn2 = ops.batch_norm(name='g_bn2')
        self.g_bn3 = ops.batch_norm(name='g_bn3')
        self.g_bn4 = ops.batch_norm(name='g_bn4')

        with tf.variable_scope('gan'):
            self.images = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], 3],
                                         name='final_image')
            self.z = tf.placeholder(tf.float32, shape=[batch_size, self.z_size], name='z')

        self.G = self.generator(self.z)

        with tf.variable_scope('gan'):
            self.D_real, self.D_real_logits = self.discriminator(self.images)
            self.D_fake, self.D_fake_logits = self.discriminator(self.G, reuse=True)

            self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_real_logits,
                                                                                      tf.ones_like(self.D_real)))
            self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_logits,
                                                                                      tf.zeros_like(self.D_fake)))
            self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_logits,
                                                                                 tf.ones_like(self.D_fake)))
            self.D_loss = self.D_loss_real + self.D_loss_fake

            allvars = tf.trainable_variables()
            self.D_vars = [v for v in allvars if 'd_' in v.name]
            self.G_vars = [v for v in allvars if 'g_' in v.name]

            self.D_optim = tf.train.GradientDescentOptimizer(self.lrate).minimize(self.D_loss, var_list=self.D_vars)
            self.G_optim = tf.train.GradientDescentOptimizer(self.lrate).minimize(self.G_loss, var_list=self.G_vars)

            self.saver = tf.train.Saver()

    def train(self):
        if not os.path.exists(os.path.join("data", "gan")):
            print "No GAN training files found. Training aborted. =("
            return

        dataset_files = glob.glob("data/gan/*.png")
        dataset_files = np.array(dataset_files)
        n_files = dataset_files.shape[0]
        sample_z = np.random.uniform(-1, 1, [self.batch_size, self.z_size])

        self.session.run(tf.initialize_all_variables())
        self.rendernet.load('checkpoint')
        for epoch in xrange(self.n_iterations):

            rand_idxs = np.random.permutation(range(n_files))
            n_batches = n_files // self.batch_size

            for batch_i in xrange(n_batches):
                idxs_i = rand_idxs[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
                imgs_batch = ops.load_imgbatch(dataset_files[idxs_i])
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_size])

                dloss_fake = self.D_loss_fake.eval(session=self.session, feed_dict={self.z: batch_z})
                dloss_real = self.D_loss_real.eval(session=self.session, feed_dict={self.images: imgs_batch})
                gloss = self.G_loss.eval(session=self.session, feed_dict={self.z: batch_z})

                train_discriminator = True
                train_generator = True

                margin = 0.3
                if dloss_fake < margin or dloss_real < margin:
                    train_discriminator = False
                if dloss_fake > 1.0-margin or dloss_real > 1.0-margin:
                    train_generator = False
                if train_discriminator is False and train_generator is False:
                    train_generator = train_discriminator = True

                # Update discriminator
                if train_discriminator:
                    self.session.run(self.D_optim, feed_dict={self.images: imgs_batch, self.z: batch_z})
                # Update generator
                if train_generator:
                    for i in xrange(5):
                        self.session.run(self.G_optim, feed_dict={self.z: batch_z})


#                test_params = np.zeros((64,23))
#                test_params[0, :] = np.array([5, 5, 5,
#                                        1.2, 0.5,
#                                        1, 0, 0,
#                                        1, 1, 0,
#                                        0, 0, 1,
#                                        0, 1, 0,
#                                        0, 1, 1,
#                                        1, 1, 1])

                if batch_i % 10 == 0:
                    rendered_images = self.G.eval(session=self.session, feed_dict={self.z: sample_z})
                    rendered_images = np.array(rendered_images)
                    ops.save_images(rendered_images, [8, 8], "results/gancubes{}.png".format(epoch*n_batches+batch_i))

                print "EPOCH[{}], BATCH[{}/{}]".format(epoch, batch_i, n_batches)
                print "Discriminator Loss - Real:{} / Fake:{} - Total:{}".format(dloss_real, dloss_fake,
                                                                                 dloss_real + dloss_fake)
                print "Generator Loss:{}".format(gloss)

    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = ops.lrelu(ops.conv2d(image, self.d_size, name='d_h0_conv'))
        h1 = ops.lrelu(ops.conv2d(h0, self.d_size*2, name='d_h1_conv'))
        h2 = ops.lrelu(ops.conv2d(h1, self.d_size*4, name='d_h2_conv'))
        h3 = ops.lrelu(ops.conv2d(h2, self.d_size*8, name='d_h3_conv'))
        h4 = ops.linear(tf.reshape(h3, [self.batch_size, -1]), 4*4*self.d_size*8, 1, scope='d_h5_lin')

        return tf.nn.sigmoid(h4), h4

    def generator(self, z_enc):
        with tf.variable_scope('gan'):
            h0 = ops.linear(z_enc, self.z_size, 256, scope='g_h0')
            h0 = ops.lrelu(self.g_bn0(h0))
            h1 = ops.linear(h0, 256, 256, activation=ops.lrelu, scope='g_h1')
            h1 = ops.lrelu(self.g_bn1(h1))
            h2 = ops.linear(h1, 256, 256, activation=ops.lrelu, scope='g_h2')
            h2 = ops.lrelu(self.g_bn2(h2))
            h3 = ops.linear(h2, 256, 256, activation=ops.lrelu, scope='g_h3')
            h3 = ops.lrelu(self.g_bn3(h3))
            h4 = ops.linear(h3, 256, 256, activation=ops.lrelu, scope='g_h4')
            h4 = ops.lrelu(self.g_bn4(h4))
            self.img_params = ops.linear(h4, 256, self.rendernet.input_size, scope='g_img_params')

        with tf.variable_scope('rendernet'):
            rendered_img = self.rendernet.render(self.img_params, reuse=True)
        return rendered_img


def main():
    rgan = RenderGAN()
    rgan.train()


if __name__ == '__main__':
    main()
