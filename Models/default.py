import os
import numpy as np
from yacs.config import CfgNode as CN
_C = CN()
_C.TRAIN = CN()
_C.TRAIN.weight_decay = 1e-4
_C.TRAIN.lr_drop = 40#
_C.TRAIN.device = "cuda"
_C.TRAIN.seed = 42