import sys
import os
import itertools
import json
# from munkres import Munkres
from os.path import join as joinpath
import numpy as np
import torch
import time
import traceback
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import seaborn as sns
sns.set()
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.serif": ["Palatino"]})
plt.rcParams.update(
    {"axes.facecolor": (0.91764706, 0.91764706, 0.94901961, 0.35)})

torch.manual_seed(2022)
np.random.seed(2022)

# joinpath(os.path.dirname(os.path.abspath(__file__)), "..", "..", "../")
ROOT_PATH = os.path.abspath(os.curdir)

# Distances constants

JACCARD = 0
TANIMOTO = 1

# SOM Constants

NEW_INLIERS = 0
OLD_INLIERS = 1

PCA = 0
RND = 1
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
