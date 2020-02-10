import os, sys
import math
import random
import numpy as np
import time
import pickle
import json
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

default_cube_size       = 64
default_cell_size       = 8
default_channel_num     = 11
default_stride          = 32
default_voxel_size      = 1.
default_density_cutoff  = 2.

default_params = dict(
    voxel_size      = default_voxel_size,
    density_cutoff  = default_density_cutoff,
    cube_size       = default_cube_size,
    cell_size       = default_cell_size,
    channel_num     = default_channel_num,
    stride          = default_stride,
    seed            = 0,
    rotation        = True,
    shuffle         = True,
    cost_lambda     = 5.,
    cost_gamma      = 1e-5,
    batch_size      = 8,
    minibatch_size  = 32,
    epoch_num       = 200,
    threads         = 1,
    threads_desc    = 8,
)

def update_params_predict(params):
    params["stride"] = params["cube_size"]
    # params["minibatch_size"] *= 2
    return params

default_rotation_axes = [
    [ 0.79465447,   0.57735027,  0.18759247],
    [-0.30353100,   0.93417236,  0.18759247],
    [-0.98224695,  -0.00000000,  0.18759247],
    [-0.30353100,  -0.93417236,  0.18759247],
    [ 0.79465447,  -0.57735027,  0.18759247],
    [ 0.49112347,   0.35682209,  0.79465447],
    [-0.18759247,   0.57735027,  0.79465447],
    [-0.60706200,  -0.00000000,  0.79465447],
    [-0.18759247,  -0.57735027,  0.79465447],
    [ 0.49112347,  -0.35682209,  0.79465447],
]

class Timer:
    def __init__(self):
        self.start = time.time()

    def s_(self):
        return time.time() - self.start
    def m_(self):
        return self.s_() / 60
    def h_(self):
        return self.m_() / 60

    def s(self):
        return int(math.floor(self.s_() % 60))
    def m(self):
        return int(math.floor(self.m_() % 60))
    def h(self):
        return int(math.floor(self.h_()))

    def __call__(self):
        return self.s_()

    def __str__(self):
        s = "{:3d}:{:02d}:{:02d}".format(self.h(), self.m(), self.s())
        return s

    def str_l(self):
        s = "{:3d}:{:02d}:{:02.3f}".format(self.h(), self.m(), self.s_() % 60)
        return s

def save(obj, filename):
    with open(filename, "wb") as file:
        return pickle.dump(obj, file)

def load(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)

def save_params(params, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(params, file, ensure_ascii=False, indent=4)

def load_params(filename):
    with open(filename, "r") as file:
        return json.load(file)

class Logger:
    def __init__(self, filename, clear=True, **args):
        self.filename = filename
        if clear:
            self.clear()
        self.written = 0
        self.init(**args)

    def init(self, **args):
        pass

    def clear(self):
        # with open(self.filename, "w") as file:
        #     pass
        if os.path.isfile(self.filename):
            os.remove(self.filename)

    def to_str(self, s, format_s=False):
        res = ""
        if type(s) == str:
            if format_s:
                res = "{:10s}".format(s)
            else:
                res = s
        elif type(s) in [int, np.int8, np.int16, np.int32, np.int64]:
            res = "{:5d}".format(s)
        elif type(s) in [float, np.float32, np.float64, np.float16]:
            res = "{:8.3f}".format(s)
        elif type(s) == list:
            for v in s:
                res += self.to_str(v, format_s=True) + ""
        return res

    def write(self, s, end="\n", mode="a"):
        with open(self.filename, mode) as file:
            file.write(self.to_str(s) + end)
        self.written += 1
        return

def read_set(filename):
    with open(filename, "r") as file:
        path = file.readline().rstrip()
        train_list = file.readline().rstrip().split()
        test_list  = file.readline().rstrip().split()
    if not os.path.isdir(path):
        path = os.path.join(os.path.dirname(filename), path)
    return path, train_list, test_list

def write_set(path, train_list, test_list, filename):
    with open(filename, "w") as file:
        file.write(path + "\n")
        file.write(" ".join(train_list) + "\n")
        file.write(" ".join(test_list) + "\n")
    return

def read_dataset(filename):
    dataset = []
    with open(filename, "r") as file:
        for line in file:
            dataset.append(line.rstrip())
    return dataset

def get_simple_splits(list_, test_size=0.1, seed=0):
    random.seed(seed)
    random.shuffle(list_)
    train_split = list_[:int(len(list_) * (1. - test_size))]
    test_split  = list_[len(train_split):]
    return train_split, test_split

def rotation_matrix(theta, phi, psi):
    u = [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)]
    c, s, C = np.cos(psi), np.sin(psi), 1 - np.cos(psi)
    m = np.array([
        [u[0] * u[0] * C + c,           u[0] * u[1] * C - u[2] * s, u[0] * u[2] * C + u[1] *s],
        [u[1] * u[0] * C + u[2] * s,    u[1] * u[1] * C + c,        u[1] * u[2] * C - u[0] *s],
        [u[2] * u[0] * C - u[1] * s,    u[2] * u[1] * C + u[0] * s, u[2] * u[2] * C + c]
    ])
    return m

def moving_average(a, n=10):
    res = np.zeros_like(a)
    for i in range(len(a)):
        res[i] = np.mean(a[max(i - n, 0) : i + n], axis=0)
    return res

def distance(p1, p2):
    return np.linalg.norm(p1[:3] - p2[:3])

# import seaborn as sns
# channel_num = 11
# h, l, s     = 0.64, 0.6, 1.
# colors_all  = sns.hls_palette(channel_num, h=h, l=l, s=s)
# color_g     = sns.hls_palette(channel_num, h=h, l=l, s=0)

colors_all_ = [
    [0.200, 0.328, 1.000],
    [0.508, 0.200, 1.000],
    [0.945, 0.200, 1.000],
    [1.000, 0.200, 0.619],
    [1.000, 0.217, 0.200],
    [1.000, 0.654, 0.200],
    [0.910, 1.000, 0.200],
    [0.473, 1.000, 0.200],
    [0.200, 1.000, 0.363],
    [0.200, 1.000, 0.799],
    [0.200, 0.764, 1.000],
]
color_g = [0.6, 0.6, 0.6]

color_indices   = [10, 5, 3, 7, 1, 4, 9, 0, 6, 2, 8]
colors_channels = colors_all_[:]
colors_all      = [colors_all_[i] for i in color_indices] + [color_g]
colors          = colors_all[1:]