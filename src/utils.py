# For logging, seeding, and reproducibility.
import os
import random

import joblib
import numpy as np


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_model(model, path):
    joblib.dump(model, path)
