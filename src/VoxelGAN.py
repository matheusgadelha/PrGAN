import tensorflow as tf
import numpy as np
import ops
import glob
import os

class RenderGAN:

    def __init__(self, sess=tf.Session(), image_size=(32, 32), z_size=201,
                 n_iterations=1000, batch_size=64, lrate=0.002, d_size=64):

        self.image_size = image_size
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lrate = lrate
        self.session = sess
        self.base_dim = 512
        self.d_size = 128
        self.z_size = z_size
        self.tau = 1e-1
        self.dataset_name = "chairs_voxel"
        #self.alpha = tf.constant(1e-6)
        self.alpha = tf.constant(0.0)
        self.beta = tf.constant(0.0)
        self.size = image_size[0]
        self.logpath = "log"

        self.lr = tf.constant(np.load('data/lr.npy'))
        self.rl = tf.constant(np.load('data/rl.npy'))
        self.fb = tf.constant(np.load('data/fb.npy'))
        self.bf = tf.constant(np.load('data/bf.npy'))

        self.g_bnf0 = ops.batch_norm_linear(name='g_bnf0')
        self.g_bn0 = ops.batch_norm(name='g_bn0', dim=3)
        self.g_bn1 = ops.batch_norm(name='g_bn1', dim=3)
        self.g_bn2 = ops.batch_norm(name='g_bn2', dim=3)
        self.g_bn3 = ops.batch_norm(name='g_bn3', dim=3)
        self.g_bn4 = ops.batch_norm(name='g_bn4', dim=3)

        self.d_bn0 = ops.batch_norm(name='d_bn0', dim=3)
        self.d_bn1 = ops.batch_norm(name='d_bn1', dim=3)
        self.d_bn2 = ops.batch_norm(name='d_bn2', dim=3)
        self.d_bn3 = ops.batch_norm(name='d_bn3', dim=3)
        self.d_bn4 = ops.batch_norm(name='d_bn4', dim=3)

        self.history = {}
        self.history["generator"] = []
        self.history["discriminator_real"] = []
        self.history["discriminator_fake"] = []

        with tf.variable_scope('gan'):
            self.images = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], image_size[0]],
                                         name='final_image')
            self.z = tf.placeholder(tf.float32, shape=[batch_size, self.z_size], name='z')

        self.G = self.generator(self.z)

        with tf.variable_scope('gan'):
            self.D_real, self.D_real_logits, self.D_stats_real = self.discriminator(self.images)
            self.D_fake, self.D_fake_logits, self.D_stats_fake = self.discriminator(self.G, reuse=True)

            self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_real_logits,
                                                                                      tf.ones_like(self.D_real)))
            self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_logits,
                                                                                      tf.zeros_like(self.D_fake)))
            #self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_logits,
            #                                                                     tf.ones_like(self.D_fake)))
            dr = tf.reduce_mean(self.D_stats_real, 0)
            dl = tf.reduce_mean(self.D_stats_fake, 0)
            self.G_loss = ops.l2(dr, dl)
            #print self.G_loss.get_shape()
            self.D_loss = 0.5*(self.D_loss_real + self.D_loss_fake)

            allvars = tf.trainable_variables()
            self.D_vars = [v for v in allvars if 'd_' in v.name]
            self.G_vars = [v for v in allvars if 'g_' in v.name]

            self.D_optim = tf.train.AdamOptimizer(1e-5).minimize(self.D_loss, var_list=self.D_vars)
            self.G_optim = tf.train.AdamOptimizer(0.0025).minimize(self.G_loss, var_list=self.G_vars)

            self.saver = tf.train.Saver()

    def train(self):
        if not os.path.exists(os.path.join("data", self.dataset_name)):
            print "No GAN training files found. Training aborted. =("
            return

        dataset_files = glob.glob("data/"+self.dataset_name+"/*.npy")
        dataset_files = np.array(dataset_files)
        n_files = dataset_files.shape[0]
        sample_z = np.random.uniform(-1, 1, [self.batch_size, self.z_size])

        self.session.run(tf.initialize_all_variables())
        for epoch in xrange(self.n_iterations):

            rand_idxs = np.random.permutation(range(n_files))
            n_batches = n_files // self.batch_size

            for batch_i in xrange(n_batches):
                idxs_i = rand_idxs[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
                #imgs_batch = ops.load_imgbatch(dataset_files[idxs_i], color=False)
                imgs_batch = ops.load_voxelbatch(dataset_files[idxs_i])
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_size])

                dloss_fake = self.D_fake.eval(session=self.session, feed_dict={self.z: batch_z})
                dloss_real = self.D_real.eval(session=self.session, feed_dict={self.images: imgs_batch})
                gloss = self.G_loss.eval(session=self.session, feed_dict={self.z: batch_z, self.images: imgs_batch})

                train_discriminator = True
                train_generator = True

                margin = 0.8
                dacc = 1. - (np.mean(np.abs(np.ones_like(dloss_real)-dloss_real)) + np.mean(dloss_fake))*0.5
                #print np.mean(dloss_real)
                #print np.mean(dloss_fake)
                if dacc > margin:
                    train_discriminator = False
                #if dloss_fake > 1.0-margin or dloss_real > 1.0-margin:
                #    train_generator = False
                #if train_discriminator is False and train_generator is False:
                #    train_generator = train_discriminator = True

                # Update discriminator
                if train_discriminator:
                    self.session.run(self.D_optim, feed_dict={self.images: imgs_batch, self.z: batch_z})
                # Update generator
                if train_generator:
                    for i in xrange(1):
                        self.session.run(self.G_optim, feed_dict={self.z: batch_z, self.images: imgs_batch})

                if batch_i % 50 == 0:
                    #voxels = self.voxels.eval(session=self.session, feed_dict={self.z: sample_z})
                    rendered_images = self.G.eval(session=self.session, feed_dict={self.z: sample_z, self.images: imgs_batch})
                    rendered_images = np.array(rendered_images)
                    ops.save_voxels(rendered_images, "results/chairs_voxel")
                    #ops.save_images(rendered_images, [8, 8],
                    #                "results/voxelchairs{}.png".format(epoch*n_batches+batch_i))
                    #ops.save_voxels(voxels, "results/chairs_voxel")

                print "EPOCH[{}], BATCH[{}/{}]".format(epoch, batch_i, n_batches)
               # print "Discriminator Loss - Real:{} / Fake:{} - Total:{}".format(dloss_real, dloss_fake,
               #                                                                  dloss_real + dloss_fake)
                print "Discriminator avg acc: {}".format(dacc)
                print "Generator Loss:{}".format(gloss)
                self.history["generator"].append(gloss)
                self.history["discriminator_real"].append(dloss_real)
                self.history["discriminator_fake"].append(dloss_fake)

        np.save(os.path.join(self.logpath, "generator.npy"), np.array(self.history["generator"]))
        np.save(os.path.join(self.logpath, "discriminator_real.npy"), np.array(self.history["discriminator_real"]))
        np.save(os.path.join(self.logpath, "discriminator_fake.npy"), np.array(self.history["discriminator_fake"]))

    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        reshaped_img = tf.reshape(image, [self.batch_size, self.image_size[0], self.image_size[1], self.image_size[0], 1])
        h0 = ops.conv3d(reshaped_img, self.d_size, name='d_h0_conv')
        h0 = ops.lrelu(self.d_bn0(h0))
        h1 = ops.conv3d(h0, self.d_size*2, name='d_h1_conv')
        h1 = ops.lrelu(self.d_bn1(h1))
        h2 = ops.conv3d(h1, self.d_size*4, name='d_h2_conv')
        h2_tensor = ops.lrelu(self.d_bn2(h2))
        h2 = tf.reshape(h2_tensor, [self.batch_size, -1])
        h3 = ops.linear(h2, h2.get_shape()[1], 1, scope='d_h5_lin')

        return tf.nn.sigmoid(h3), h3, h1

    def generator(self, z_enc):
        with tf.variable_scope('gan'):
          #  stddev = 2e-3
          #  f0 = ops.linear(z_enc[:, 0:(self.z_size-1)], self.z_size-1, 32*32, scope='g_f0')
          #  f0 = ops.lrelu(self.g_bnf0(f0))
          #  h0 = ops.linear(f0, 32*32, 4*4*32*16, scope='g_h0')
          #  h0 = tf.reshape(h0, [-1, 4, 4, 32*16])
          #  h0 = ops.lrelu(self.g_bn0(h0))
          #  h1 = ops.deconv2d(h0, [self.batch_size, 8, 8, 32*8], name='g_h1', stddev=stddev)
          #  h1 = ops.lrelu(self.g_bn1(h1))
          #  h2 = ops.deconv2d(h1, [self.batch_size, 16, 16, 32*4], name='g_h2', stddev=stddev)
          #  h2 = ops.lrelu(self.g_bn2(h2))
          #  h3 = ops.deconv2d(h2, [self.batch_size, 32, 32, 32], name='g_h4', stddev=stddev)
          #  h3 = ops.lrelu(self.g_bn4(h3))
          #  self.voxels = h3

            base_filters = self.d_size
            h0 = ops.linear(z_enc[:, 0:(self.z_size-1)], self.z_size-1, 4*4*4*base_filters, scope='g_f0')
            h0 = tf.reshape(h0, [self.batch_size, 4, 4, 4, base_filters])
            h0 = tf.nn.relu(self.g_bn0(h0))
            h1 = ops.deconv3d(h0, [self.batch_size, 8, 8, 8, base_filters/2], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1))
            h2 = ops.deconv3d(h1, [self.batch_size, 16, 16, 16, base_filters/4], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2))
            h3 = ops.deconv3d(h2, [self.batch_size, 32, 32, 32, 1], name='g_h3')
            h3 = tf.nn.sigmoid(self.g_bn3(h3))
            self.voxels = tf.reshape(h3, [64, 32, 32, 32])
           # v = z_enc[:, self.z_size-1]

           # rendered_imgs = []
           # for i in xrange(self.batch_size):
           #     img = tf.case({tf.less(v[i], tf.constant(-0.5)): lambda: tf.reduce_sum(self.voxels[i], 0),
           #                     tf.logical_and(tf.less(v[i], tf.constant(0.0)), tf.less(tf.constant(-0.5), v[i])): lambda: tf.reduce_sum(self.voxels[i], 1),
           #                     tf.logical_and(tf.less(v[i], tf.constant(0.5)), tf.less(tf.constant(0.0), v[i])): lambda: tf.reverse(tf.reduce_sum(self.voxels[i], 0), [False, True])},
           #                     # tf.less(v[i], tf.constant(1.0)): lambda: tf.reverse(tf.reduce_sum(self.voxels, 1), [False, False, True])},
           #                   default=lambda: tf.reverse(tf.reduce_sum(self.voxels[i], 1), [False, True]),
           #                   exclusive=True)
           #     img = tf.sub(tf.ones_like(img), tf.exp(tf.mul(img, -self.tau)))
           #     #if i == 0:
           #     #    self.img0 = img
           #     rendered_imgs.append(img)

           # self.final_imgs = tf.reshape(tf.pack(rendered_imgs), [64, 32, 32, 1])
        return self.voxels


def main():
    rgan = RenderGAN()
    rgan.train()


if __name__ == '__main__':
    main()
