import numpy as np
import os

data_dir = "/home/mingze/Experiments/baseline/Multi-STL/work_dirs/test/outputs/out_data.npy"
data = np.load(data_dir)

print(data.shape)