import numpy as np
import tensorflow as tf
from RenderNet import RenderNet

class RenderGAN:

    def __init__(self, sess=tf.Session(), image_size=(64,64), z_size=10,
                 n_iterations=50, batch_size=64, lrate=0.001):

        self.image_size = image_size
        self.n_pixels = self.image_size[0] * self.image_size[1] * 3  # number of channels
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lrate = lrate
        self.session = sess
        self.base_dim = 512

        self.rendernet = RenderNet(sess=self.session, image_size=self.image_size,
                                   batch_size=self.batch_size)

def main():
    rgan = RenderGAN()
    rgan.rendernet.test()

if __name__ == '__main__':
    main()
