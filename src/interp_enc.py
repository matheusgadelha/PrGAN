import numpy as np

z1 = np.load("results/PrGANsairplaneV.npy")[14, :]
z2 = np.load("results/PrGANsairplaneV.npy")[1, :]
#z2 = np.load("results/PrGANsairplaneV.npy")[61, :]
#z1 = np.random.uniform(-1, 1, 201)
#z2 = np.random.uniform(-1, 1, 201)

results = []
steps = 64

for i in range(steps):
    t = i/float(steps-1)
    v = t*z1 + (1.0-t)*z2
    results.append(v.copy())

results = np.vstack(results)
print results[:, 0]
np.save("interp_encs.npy", results)
