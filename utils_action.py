# import scipy.ndimage.interpolation as inter
from scipy.spatial.distance import cdist
from scipy.signal import medfilt
import scipy
import numpy as np
import random


class Config:
    def __init__(self):
        self.frame_l = 40
        self.joint_n = 17
        self.joint_d = 2
        self.clc_num = 2
        self.feat_d = 136
        self.filters = 16


def zoom(p, target_l=64, joints_num=25, joints_dim=3):
    l = p.shape[0]
    p_new = np.empty([target_l, joints_num, joints_dim])
    for m in range(joints_num):
        for n in range(joints_dim):
            p[:, m, n] = medfilt(p[:, m, n], 3)
            # p_new[:, m, n] = inter.zoom(p[:, m, n], target_l / l)[:target_l]
            p_new[:, m, n] = scipy.ndimage.zoom(p[:, m, n], target_l / l)[:target_l]
            
    return p_new


def sampling_frame(p, C):
    full_l = p.shape[0]  # full length
    if random.uniform(0, 1) < 0.5:  # aligment sampling
        valid_l = np.round(np.random.uniform(0.85, 1) * full_l)
        s = random.randint(0, full_l - int(valid_l))
        e = s + valid_l  # sample end point
        p = p[int(s):int(e), :, :]
    else:  # without aligment sampling
        valid_l = np.round(np.random.uniform(0.9, 1) * full_l)
        index = np.sort(np.random.choice(
            range(0, full_l), int(valid_l), replace=False))
        p = p[index, :, :]
    p = zoom(p, C.frame_l, C.joint_n, C.joint_d)
    return p


def norm_scale(x):
    return (x - np.mean(x)) / np.mean(x)


def get_CG(p, C):
    M = []
    iu = np.triu_indices(C.joint_n, 1, C.joint_n)
    for f in range(C.frame_l):
        d_m = cdist(p[f], p[f], "euclidean")
        d_m = d_m[iu]
        M.append(d_m)
    M = np.stack(M)
    M = norm_scale(M)
    return M

