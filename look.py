import numpy as np
import os

data_dir = "/dev/shm/SD/config3/test.npy"
data = np.load(data_dir)

print(data.shape)