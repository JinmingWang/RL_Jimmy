import numpy as np
import matplotlib.pyplot as plt
from typing import *

def drawFunction(func: Callable[[float], float], x_min: float=-2, x_max: float=2) -> None:
    x = np.linspace(x_min, x_max, num=100)
    y = func(x)
    plt.plot(x, y)


def drawFucntionWithVariance(mean_func, var_func, x_min=-2, x_max=2) -> None:
    x = np.linspace(x_min, x_max, num=100)
    y = mean_func(x)

    var = var_func(x)
    #print(x, y, var)
    plt.vlines(x, y-var, y+var, alpha=0.5)
    plt.plot(x, y)