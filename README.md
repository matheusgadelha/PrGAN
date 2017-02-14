# PrGAN

Dependencies
-------------

The following dependencies can be installed through pip or your favorite package manager (apt, pacmam, etc)

* Matplotlib
* Numpy
* Scipy

```
sudo pip install matplotlib numpy scipy
```

Our code also relies on Tensorflow. The easiest way to install it is probably using a [pip package
(https://www.tensorflow.org/get_started/os_setup#pip_installation).
If you want to use the GPU version (highly recommended), you also need to install [CUDA](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn).

Dataset
-------

PrGAN training data consists of a set of grayscale images stored inside the data folder. This folder needs to be created in the
same path where src folder is located (the main PrGAN directory). Inside the data data folder, you should place the images
of your dataset in separate folders and name them accordingly. For example, if you want to use a dataset with airplane images 
you should save those images inside ```data/airplane```. You can use different names for the folders, but the folder name is
the name of the dataset.

The dataset is simply a collection of grayscale png files. There is no special naming convetion or annotation that needs to be
followed.

Usage
-----

Before executing PrGAN, you should create a ```results``` folder where the results will be stored.
If you want to train the PrGAN on, for example, the airplane dataset during 25 epochs, you can just execute (from the main folder):
```
python src/PrGAN -d airplane --train -e 25
```
Results will be saved at each 50 iterations.
You can have a good idea about the generated geometry by taking a look at the image files created.
However, the geometry is saved in a series of ```volume.npy``` files.
Only the last batch is saved.
You can use a script called ```create_cubes.py``` to generate ```.obj``` files from the ```.npy```s:
```
python src/create_cubes.py "results/PrGANairplane/*.npy" 0.5
```
