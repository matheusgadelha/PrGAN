import tensorflow as tf
import numpy as np
import ops
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='This program trains a PrGAN model.')
parser.add_argument("-e", "--epochs", type=int, help="Number training epochs.", default=50)
parser.add_argument("-ims", "--image_size", type=int, help="Image size (single dimension).", default=32)
parser.add_argument("-bs", "--batch_size", type=int, help="Minibatch size.", default=64)
parser.add_argument("-d", "--dataset", type=str,
                    help="Dataset name. There must be a folder insde of the data folder with the same name.",
                    default="chairs_canonical")
parser.add_argument("-z", "--encoding", type=str,
                    help="Path to a .npy file containing an encoding to generate shapes.",
                    default="None")
parser.add_argument("--train", dest='train', action='store_true')
parser.set_defaults(train=False)

def create_folder(path):
    if os.path.exists(path):
        return
    else:
        print "Folder {} not found. Creating one...".format(path)
        os.makedirs(path)
        print "Done."

class PrGAN:

    def __init__(self, sess=tf.Session(), image_size=(32, 32), z_size=201,
                 n_iterations=50, dataset="None", batch_size=64, lrate=0.002, d_size=64):

        self.image_size = image_size
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lrate = lrate
        self.session = sess
        self.base_dim = 512
        self.d_size = 128
        self.z_size = z_size
        self.tau = 1
        self.dataset_name = dataset
        #self.alpha = tf.constant(1e-6)
        self.alpha = tf.constant(0.0)
        self.beta = tf.constant(0.0)
        self.size = image_size[0]
        self.logpath = "log"

        self.g_bn0 = ops.BatchNormalization([self.d_size], 'g_bn0')
        self.g_bn1 = ops.BatchNormalization([self.d_size/2], 'g_bn1')
        self.g_bn2 = ops.BatchNormalization([self.d_size/4], 'g_bn2')
        self.g_bn3 = ops.BatchNormalization([self.d_size/8], 'g_bn2')

        self.d_bn0 = ops.BatchNormalization([self.d_size], 'd_bn0')
        self.d_bn1 = ops.BatchNormalization([self.d_size*2], 'd_bn1')
        self.d_bn2 = ops.BatchNormalization([self.d_size*4], 'd_bn2')
        self.d_bn3 = ops.BatchNormalization([self.d_size*8], 'd_bn2')

        self.history = {}
        self.history["generator"] = []
        self.history["discriminator_real"] = []
        self.history["discriminator_fake"] = []

        with tf.variable_scope('gan'):
            self.images = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], 1],
                                         name='final_image')
            self.z = tf.placeholder(tf.float32, shape=[batch_size, self.z_size], name='z')

        self.train_flag = tf.placeholder(tf.bool)
        self.G = self.generator(self.z, self.train_flag)

        with tf.variable_scope('gan'):
            self.D_real, self.D_real_logits, self.D_stats_real = self.discriminator(self.images, self.train_flag)
            self.D_fake, self.D_fake_logits, self.D_stats_fake = self.discriminator(self.G, self.train_flag, reuse=True)

            self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_real_logits,
                                                                                      tf.ones_like(self.D_real)))
            self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_logits,
                                                                                      tf.zeros_like(self.D_fake)))
            self.G_loss_classic = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_logits,
                                                                                 tf.ones_like(self.D_fake)))
            dr_mean, dr_var = tf.nn.moments(self.D_stats_real, axes=[0])
            dl_mean, dl_var = tf.nn.moments(self.D_stats_fake, axes=[0])
            self.G_loss = ops.l2(dr_mean, dl_mean)
            #self.G_loss += ops.l2(dr_var, dl_var)
            #print self.G_loss.get_shape()
            self.D_loss = self.D_loss_real + self.D_loss_fake

            allvars = tf.trainable_variables()
            self.D_vars = [v for v in allvars if 'd_' in v.name]
            self.G_vars = [v for v in allvars if 'g_' in v.name]

            self.D_optim = tf.train.AdamOptimizer(1e-5, beta1=0.5).minimize(self.D_loss, var_list=self.D_vars)
            self.G_optim = tf.train.AdamOptimizer(0.0025, beta1=0.5).minimize(self.G_loss, var_list=self.G_vars)
            self.G_optim_classic = tf.train.AdamOptimizer(0.0025, beta1=0.5).minimize(self.G_loss_classic, var_list=self.G_vars)

            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def load(self):
        ckpt_folder = "checkpoint"
        ckpt = tf.train.get_checkpoint_state(os.path.join(ckpt_folder, "PrGAN"+self.dataset_name))
        if ckpt and ckpt.model_checkpoint_path:
            print "Loading previous model..."
            self.saver.restore(self.session, ckpt.model_checkpoint_path)
            print "Done."
        else:
            print "No saved model found."

    def train(self):
        if not os.path.exists(os.path.join("data", self.dataset_name)):
            print "No GAN training files found. Training aborted. =("
            return

        dataset_files = glob.glob("data/"+self.dataset_name+"/*.png")
        dataset_files = np.array(dataset_files)
        n_files = dataset_files.shape[0]
        sample_z = np.random.uniform(-1, 1, [self.batch_size, self.z_size])
        training_step = 0

        self.session.run(tf.initialize_all_variables())
        self.load()
        for epoch in xrange(self.n_iterations):

            rand_idxs = np.random.permutation(range(n_files))
            n_batches = n_files // self.batch_size

            for batch_i in xrange(n_batches):
                idxs_i = rand_idxs[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
                imgs_batch = ops.load_imgbatch(dataset_files[idxs_i], color=False)
                #imgs_batch = ops.load_voxelbatch(dataset_files[idxs_i])
                batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_size])

                dloss_fake = self.D_fake.eval(session=self.session, feed_dict={self.z: batch_z, self.train_flag: False})
                dloss_real = self.D_real.eval(session=self.session, feed_dict={self.images: imgs_batch, self.train_flag: False})
                gloss = self.G_loss.eval(session=self.session, feed_dict={self.z: batch_z, self.images: imgs_batch, self.train_flag: False})

                train_discriminator = True

                margin = 0.8
                dacc_real = np.mean(dloss_real)
                dacc_fake = np.mean(np.ones_like(dloss_fake) - dloss_fake)
                dacc = (dacc_real + dacc_fake)*0.5
                #print np.mean(dloss_real)
                #print np.mean(dloss_fake)
                if dacc > margin:
                    train_discriminator = False
                #if dloss_fake > 1.0-margin or dloss_real > 1.0-margin:
                #    train_generator = False
                #if train_discriminator is False and train_generator is False:
                #    train_generator = train_discriminator = True

                print "EPOCH[{}], BATCH[{}/{}]".format(epoch, batch_i, n_batches)
                print "Discriminator avg acc: {}".format(dacc)
                print "Discriminator real mean: {}".format(np.mean(dloss_real))
                print "Discriminator fake mean: {}".format(np.mean(dloss_fake))
                print "Generator Loss:{}".format(gloss)

                # Update discriminator
                if train_discriminator:
                    print "***Discriminator trained.***"
                    self.session.run(self.D_optim, feed_dict={self.images: imgs_batch, self.z: batch_z, self.train_flag: True})
                # Update generator
                #if dacc > 0.9:
                #    self.session.run(self.G_optim_classic, feed_dict={self.z: batch_z})
                #if dacc > margin + 1.0:
                self.session.run(self.G_optim_classic, feed_dict={self.z: batch_z, self.images: imgs_batch, self.train_flag: True})
                self.session.run(self.G_optim, feed_dict={self.z: batch_z, self.images: imgs_batch, self.train_flag: True})
                #self.session.run(self.G_optim, feed_dict={self.z: batch_z, self.images: imgs_batch, self.train_flag: True})

                if batch_i % 50 == 0:
                    rendered_images = self.G.eval(session=self.session, feed_dict={self.z: sample_z, self.images: imgs_batch, self.train_flag: False})
                    rendered_images = np.array(rendered_images)

                    voxels = self.voxels.eval(session=self.session, feed_dict={self.z: sample_z, self.images: imgs_batch, self.train_flag: False})
                    voxels = np.array(voxels)

                    create_folder("results/PrGAN{}".format(self.dataset_name))
                    ops.save_images(rendered_images, [8, 8],
                                    "results/PrGAN{}/{}.png".format(self.dataset_name, epoch*n_batches+batch_i))
                    ops.save_images(imgs_batch, [8, 8], "sanity_chairs.png")
                    ops.save_voxels(voxels, "results/PrGAN{}".format(self.dataset_name))

                    print "Saving checkpoint..."
                    create_folder('checkpoint/PrGAN{}'.format(self.dataset_name))
                    self.saver.save(self.session, 'checkpoint/PrGAN{}/model.ckpt'.format(self.dataset_name),
                                    global_step=training_step)
                    print "***CHECKPOINT SAVED***"
                training_step += 1

                self.history["generator"].append(gloss)
                self.history["discriminator_real"].append(dloss_real)
                self.history["discriminator_fake"].append(dloss_fake)

        np.save(os.path.join(self.logpath, "generator.npy"), np.array(self.history["generator"]))
        np.save(os.path.join(self.logpath, "discriminator_real.npy"), np.array(self.history["discriminator_real"]))
        np.save(os.path.join(self.logpath, "discriminator_fake.npy"), np.array(self.history["discriminator_fake"]))

    def discriminator(self, image, train, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        reshaped_img = tf.reshape(image, [self.batch_size, self.image_size[0], self.image_size[1], 1])
        h0 = ops.conv2d(reshaped_img, self.d_size, name='d_h0_conv')
        h0 = ops.lrelu(self.d_bn0(h0, train))
        h1 = ops.conv2d(h0, self.d_size*2, name='d_h1_conv')
        h1 = ops.lrelu(self.d_bn1(h1, train))
        h2 = ops.conv2d(h1, self.d_size*4, name='d_h2_conv')
        h2 = ops.lrelu(self.d_bn2(h2, train))
        h3 = ops.conv2d(h2, self.d_size*8, name='d_h3_conv')
        h3_tensor = ops.lrelu(self.d_bn3(h3, train))
        h3 = tf.reshape(h3_tensor, [self.batch_size, -1])
        h4 = ops.linear(h3, h3.get_shape()[1], 1, scope='d_h5_lin')

        return tf.nn.sigmoid(h4), h4, h3_tensor

    def generator(self, z_enc, train):
        with tf.variable_scope('gan'):
            base_filters = self.d_size
            h0 = ops.linear(z_enc[:, 0:(self.z_size-1)], self.z_size-1, 4*4*4*base_filters, scope='g_f0')
            h0 = tf.reshape(h0, [self.batch_size, 4, 4, 4, base_filters])
            h0 = tf.nn.relu(self.g_bn0(h0, train))
            h1 = ops.deconv3d(h0, [self.batch_size, 8, 8, 8, base_filters/2], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train))
            h2 = ops.deconv3d(h1, [self.batch_size, 16, 16, 16, base_filters/4], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train))
            h3 = ops.deconv3d(h2, [self.batch_size, 32, 32, 32, 1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train))
            h4 = ops.deconv3d(h3, [self.batch_size, 64, 64, 64, 1], name='g_h4')
            h4 = tf.nn.sigmoid(h4)
            self.voxels = tf.reshape(h4, [64, 64, 64, 64])
            v = z_enc[:, self.z_size-1]

            rendered_imgs = []
            for i in xrange(self.batch_size):
                img = ops.project(ops.transform_volume(self.voxels[i], ops.rot_matrix(v[i]), size=self.image_size[0]),
                        self.tau)
                rendered_imgs.append(img)

            self.final_imgs = tf.reshape(tf.pack(rendered_imgs), [64, 64, 64, 1])
        return self.final_imgs

    def sample (self, n_batches):
        self.session.run(tf.initialize_all_variables())
        self.load()
        all_voxels = []
        all_imgs = []
        all_zs = []
        for i in xrange(n_batches):
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_size])
            all_zs.append(batch_z)
            voxels = self.voxels.eval(session=self.session,
                                      feed_dict={self.z: batch_z, self.train_flag: False})
            imgs = self.final_imgs.eval(session=self.session,
                                      feed_dict={self.z: batch_z, self.train_flag: False})
            all_voxels.append(np.array(voxels))
            all_imgs.append(np.array(imgs))
        all_voxels = np.concatenate(all_voxels, axis=0)
        all_imgs = np.concatenate(all_imgs, axis=0)
        all_zs = np.vstack(all_zs)
        print all_voxels.shape
        np.save("results/PrGAN{}".format(self.dataset_name), all_zs)
        ops.save_voxels(all_voxels, "results/PrGAN{}".format(self.dataset_name))
        ops.save_separate_images(all_imgs, "results/PrGAN{}".format(self.dataset_name))

    def test (self, encs):
        self.session.run(tf.initialize_all_variables())
        self.load()
        z = np.load(encs)
        voxels = self.voxels.eval(session=self.session,
                                    feed_dict={self.z: z, self.train_flag: False})
        imgs = self.final_imgs.eval(session=self.session,
                                    feed_dict={self.z: z, self.train_flag: False})
        create_folder("results/PrGAN{}".format(self.dataset_name))
        ops.save_voxels(voxels, "results/PrGAN{}".format(self.dataset_name))
        ops.save_separate_images(imgs, "results/PrGAN{}".format(self.dataset_name))


def main():
    args = parser.parse_args()
    rgan = PrGAN(n_iterations=args.epochs, 
            batch_size=args.batch_size, 
            image_size=(args.image_size, args.image_size),
            dataset=args.dataset)
    if args.train:
        rgan.train()
    elif args.encoding != "None":
        rgan.test(args.encoding)
    else:
        rgan.sample(1)


if __name__ == '__main__':
    main()
