import sys, os, itertools, json
# from munkres import Munkres
from os.path import join as joinpath
import numpy as np
import torch, time, traceback
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

torch.manual_seed(2022)
np.random.seed(2022)

ROOT_PATH = os.path.abspath(os.curdir)# joinpath(os.path.dirname(os.path.abspath(__file__)), "..", "..", "../")

# Distances constants

JACCARD = 0
TANIMOTO = 1

# SOM Constants

NEW_INLIERS = 0
OLD_INLIERS = 1

PCA=0
RND=1
GRID = 2

# Pif models constants

LINE = "line"
CIRCLE = "circle"
PLANE = "plane"

SOM = "som"
AE = "ae"
SUBS = "subsampling"

# Sampling constants
UNIFORM = "uniform"
LOCALIZED = "localized"