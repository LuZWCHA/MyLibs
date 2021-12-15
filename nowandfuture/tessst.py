import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf, splprep
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import Axes3D
import util.preproccess as prp
a = np.ones((3, 3, 3))
a = prp.center_pad(a, (7, 6, 8))
print(a)
a = prp.center_crop(a, (3, 3, 3))
print(a)
