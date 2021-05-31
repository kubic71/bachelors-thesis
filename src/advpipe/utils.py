from os import path
from munch import Munch
import inspect
import yaml
import matplotlib.pyplot as plt
import numpy as np


def show_img(np_img):
    plt.imshow(np_img, interpolation="bicubic")
    plt.show()



def l_inf(np_img1, np_img2):
    return np.max(np.abs(np_img1 - np_img2))

def clip_linf(orig_img, pertubed_img, epsilon=0.05):
    min_boundary = np.clip(orig_img - epsilon*np.ones_like(orig_img), 0, 1)
    max_boundary = np.clip(orig_img + epsilon*np.ones_like(orig_img), 0, 1)
    return np.clip(pertubed_img, min_boundary, max_boundary)




def load_yaml(yaml_filename):
    with open(yaml_filename, 'r') as stream:
        try:
            return Munch.fromDict(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)


def rel_to_abs_path(relative_path):
    """convert caller's relative path to absolute path"""
    callers_path = inspect.stack()[1].filename
    return path.normpath(
        path.join(path.dirname(path.abspath(callers_path)), relative_path))


def get_abs_module_path():
    return rel_to_abs_path(".")
