from typing import TypeVar
import tensorflow as tf
import torch
import numpy as np
import eagerpy as ep


# TypeVar that includes different native tensor types
TensorTypeVar = TypeVar("TensorTypeVar", torch.Tensor, tf.Tensor, np.ndarray, ep.Tensor)
