import tensorflow as tf
import ops
from RenderNet import RenderNet


class RenderGAN:

    def __init__(self, sess=tf.Session(), image_size=(64, 64), z_size=10,
                 n_iterations=50, batch_size=64, lrate=0.001, d_size=64):

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

            self.D_optim = tf.train.AdamOptimizer(self.lrate).minimize(self.D_loss, var_list=self.D_vars)
            self.G_optim = tf.train.AdamOptimizer(self.lrate).minimize(self.G_loss, var_list=self.G_vars)

            self.saver = tf.train.Saver()

    def train(self):
        pass

    def discriminator(self, image, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        h0 = ops.lrelu(ops.conv2d(image, self.d_size, name='d_h0_conv'))
        h1 = ops.lrelu(ops.conv2d(h0, self.d_size*2, name='d_h1_conv'))
        h2 = ops.lrelu(ops.conv2d(h1, self.d_size*4, name='d_h2_conv'))
        h3 = ops.lrelu(ops.conv2d(h2, self.d_size*8, name='d_h3_conv'))
        print h3.get_shape()
        h4 = ops.linear(tf.reshape(h3, [self.batch_size, -1]), 4*4*self.d_size*8, 1, scope='d_h5_lin')

        return tf.nn.sigmoid(h4), h4

    def generator(self, z_enc):
        with tf.variable_scope('gan'):
            h0 = ops.linear(z_enc, self.z_size, 256, activation=ops.lrelu, scope='g_h0')
            h1 = ops.linear(h0, 256, 256, activation=ops.lrelu, scope='g_h1')
            img_params = ops.linear(h1, 256, self.rendernet.input_size, scope='g_img_params')

        with tf.variable_scope('rendernet'):
            rendered_img = self.rendernet.render(img_params, reuse=True)
        return rendered_img


def main():
    rgan = RenderGAN()
    rgan.rendernet.architecture_check()


if __name__ == '__main__':
    main()
