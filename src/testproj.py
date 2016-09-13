import matplotlib.image as mpimg
import numpy as np

np.set_printoptions(threshold=np.inf)

imgcoord = np.array(range(0, 32*32))
imgcoord.reshape((32, 32))
print imgcoord

data = np.zeros((32, 32, 32))
data[:, 0, 0] = 1
data[2, :, 2] = 1

lr = np.zeros((32*32*32, 32*32))
for i in range(32):
    for j in range(32):
        lrvx = np.zeros((32, 32, 32))
        lrvx[:, i, j] = 1
        lr[:, 32*i + j] = lrvx.flatten()

rl = np.zeros((32*32*32, 32*32))
for i in range(32):
    for j in range(32):
        rlvx = np.zeros((32, 32, 32))
        rlvx[:, i, 31-j] = 1
        rl[:, 32*i + j] = rlvx.flatten()

fb = np.zeros((32*32*32, 32*32))
for i in range(32):
    for j in range(32):
        fbvx = np.zeros((32, 32, 32))
        fbvx[j, i, :] = 1
        fb[:, 32*i + j] = fbvx.flatten()

bf = np.zeros((32*32*32, 32*32))
for i in range(32):
    for j in range(32):
        bfvx = np.zeros((32, 32, 32))
        bfvx[31-j, i, :] = 1
        bf[:, 32*i + j] = bfvx.flatten()

res = 1 - np.exp(-1000*np.dot(data.flatten(), lr))
np.save('data/lr.npy', lr)
mpimg.imsave("proj_lr.png", res.reshape(32, 32))

res = 1 - np.exp(-1000*np.dot(data.flatten(), rl))
np.save('data/rl.npy', rl)
mpimg.imsave("proj_rl.png", res.reshape(32, 32))

res = 1 - np.exp(-1000*np.dot(data.flatten(), fb))
np.save('data/fb.npy', fb)
mpimg.imsave("proj_fb.png", res.reshape(32, 32))

res = 1 - np.exp(-1000*np.dot(data.flatten(), bf))
np.save('data/bf.npy', bf)
mpimg.imsave("proj_bf.png", res.reshape(32, 32))
