import tensorflow as tf
import numpy as np
import ops
import glob
import os
import argparse

parser = argparse.ArgumentParser(description='This program trains a PrGAN model.')
parser.add_argument("-e", "--epochs", type=int, help="Number training epochs.", default=50)
parser.add_argument("-bs", "--batch_size", type=int, help="Minibatch size.", default=64)
parser.add_argument("-d", "--dataset", type=str,
                    help="Dataset name. There must be a folder insde of the data folder with the same name.",
                    default="chairs_canonical")
parser.add_argument("--train", dest='train', action='store_true')
parser.set_defaults(train=False)

def create_folder(path):
    if os.path.exists(path):
        return
    else:
        print "Folder {} not found. Creating one...".format(path)
        os.makedirs(path)
        print "Done."

class InversePrGAN:

    def __init__(self, sess=tf.Session(), image_size=(32, 32), z_size=201,
                 n_iterations=50, dataset="None", batch_size=64, lrate=0.002, d_size=64):

        self.image_size = image_size
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lrate = lrate
        self.session = sess
        self.base_dim = 512
        self.d_size = 32
        self.z_size = z_size
        self.tau = 1
        self.dataset_name = dataset
        #self.alpha = tf.constant(1e-6)
        self.alpha = tf.constant(0.0)
        self.beta = tf.constant(0.0)
        self.size = image_size[0]
        self.logpath = "log"

        self.d_bn0 = ops.BatchNormalization([256], 'd_bn0')
        #self.d_bn0 = ops.BatchNormalization([self.d_size], 'd_bn0')
        self.d_bn1 = ops.BatchNormalization([self.d_size*2], 'd_bn1')
        self.d_bn2 = ops.BatchNormalization([self.d_size*4], 'd_bn2')

        self.history = {}
        self.history["generator"] = []
        self.history["discriminator_real"] = []
        self.history["discriminator_fake"] = []

        with tf.variable_scope('inverse_prgan'):
            self.images = tf.placeholder(tf.float32, shape=[batch_size, image_size[0], image_size[1], 1],
                                         name='final_image')
            self.encodings = tf.placeholder(tf.float32, shape=[batch_size, self.z_size], name="encoding")

        self.train_flag = tf.placeholder(tf.bool)

        with tf.variable_scope('inverse_prgan'):
            self.z = self.encode(self.images, self.train_flag)
            self.loss = ops.l2(self.encodings, self.z)
            self.optimizer = tf.train.AdamOptimizer(1e-5, beta1=0.9).minimize(self.loss)

            self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

    def load(self):
        ckpt_folder = "checkpoint"
        ckpt = tf.train.get_checkpoint_state(os.path.join(ckpt_folder, "InversePrGAN"+self.dataset_name))
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
        dataset_files.sort(key=ops.alphanum_key)
        dataset_files = np.array(dataset_files)
        dataset_encodings = np.load("data/"+self.dataset_name+"/encondings.npy")

        n_files = dataset_files.shape[0]
        training_step = 0
        test_idxs = np.random.permutation(range(n_files))[0:self.batch_size]
        test_imgs = ops.load_imgbatch(dataset_files[test_idxs], color=False)
        test_encs = dataset_encodings[test_idxs, :]

        self.session.run(tf.initialize_all_variables())
        self.load()
        for epoch in xrange(self.n_iterations):

            rand_idxs = np.random.permutation(range(n_files))
            n_batches = n_files // self.batch_size

            for batch_i in xrange(n_batches):
                idxs_i = rand_idxs[batch_i * self.batch_size: (batch_i + 1) * self.batch_size]
                imgs_batch = ops.load_imgbatch(dataset_files[idxs_i], color=False)
                self.session.run(self.optimizer, feed_dict={self.images: imgs_batch,
                                                            self.encodings: dataset_encodings[idxs_i, :],
                                                            self.train_flag: True})
                training_step += 1
                current_loss = self.session.run(self.loss, feed_dict={self.images: test_imgs,
                                                            self.encodings: test_encs,
                                                            self.train_flag: False})
                print "Epoch {}/{}, Batch {}/{}, Loss {}".format(epoch + 1, self.n_iterations,
                                                                 batch_i + 1, n_batches, current_loss)
                # Save checkpoint
                if training_step % 1000 == 0:
                    if not os.path.exists("checkpoint"):
                        print "Checkpoint folder not found. reating one..."
                        os.makedirs("checkpoint")
                        print "Done."
                    create_folder('checkpoint/InversePrGAN{}'.format(self.dataset_name))
                    self.saver.save(self.session, 'checkpoint/InversePrGAN{}/model.ckpt'.format(self.dataset_name),
                                    global_step=training_step)

    def encode(self, image, train, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        reshaped_img = tf.reshape(image, [self.batch_size, -1])
        h1 = ops.linear(reshaped_img, reshaped_img.get_shape()[1], 512, scope='enc_h1')
        h1 = ops.lrelu(h1)
        h2 = ops.linear(h1, h1.get_shape()[1], 512, scope='enc_h2')
        h3 = ops.linear(h2, h2.get_shape()[1], self.z_size, scope='enc_h3')

#        reshaped_img = tf.reshape(image, [self.batch_size, self.image_size[0], self.image_size[1], 1])
#
#        h0 = ops.conv2d(reshaped_img, self.d_size, name='d_h0_conv')
#        h0 = ops.lrelu(self.d_bn0(h0, train))
#        h1 = ops.conv2d(h0, self.d_size*2, name='d_h1_conv')
#        h1 = ops.lrelu(self.d_bn1(h1, train))
#        h2 = ops.conv2d(h1, self.d_size*4, name='d_h2_conv')
#        h2 = ops.lrelu(self.d_bn2(h2, train))
#        h2 = tf.nn.max_pool(h2, [1, 4, 4, 1], [1, 1, 1, 1], padding='SAME')
#        h2 = tf.reshape(h2, [self.batch_size, -1])
#        h3 = ops.linear(h2, h2.get_shape()[1], self.z_size, scope='d_h5_lin')

        return h3/self.z_size

    def test(self, path):
        self.load()
        imgs_path = glob.glob(path)
        imgs_path.sort(key=ops.alphanum_key)
        imgs_path = np.array(imgs_path)
        imgs_batch = ops.load_imgbatch(imgs_path[range(8*337, 8*337+64)], color=False)
        encs = self.z.eval(session=self.session, feed_dict={self.images: imgs_batch, self.train_flag: False})
        np.save("inverse_encs.npy", encs)

def main():
    args = parser.parse_args()
    rgan = InversePrGAN(n_iterations=args.epochs, batch_size=args.batch_size, dataset=args.dataset)
    if args.train:
        rgan.train()
    else:
        rgan.test("data/chairs_blur/*.png")


if __name__ == '__main__':
    main()
