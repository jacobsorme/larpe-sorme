import re
import numpy as np
import argparse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os

def findNumber(string):
    n = re.findall("\d+\.?\d+", string) # Find eg "100" or "0.12"
    return float(n[0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--param',default="lamb",type=str)
    parser.add_argument('--result',default=0,type=int) # What data to plot
    args = parser.parse_args()

    # Find what to call the z-axis
    result = "score" if args.result==1 else "last100mean"
    result = "max_q_mean" if args.result==0 else result


    files = os.listdir("dat")
    files = list(filter(lambda f: args.param in f, files))
    files = sorted(files, key=lambda f1: findNumber(f1))

    surf = np.zeros((len(files),1000))
    x = np.zeros(len(files))
    i = 0
    for f in files:
        dat = np.load("dat/"+f)
        surf[i,:] = dat[args.result,:].T
        x[i] = findNumber(f) # The value of the parameter
        i += 1

    #x = x[::-1]
    y = np.arange(1000)
    x,y = np.meshgrid(x,y)
    ax = plt.figure().gca(projection='3d')
    ax.view_init(55,-140)
    ax.plot_surface(x, y,surf.T, cmap=cm.viridis)
    ax.set_xlabel(args.param)
    ax.set_ylabel("episode")
    ax.set_zlabel(result)


    #plt.legend(files)
    plt.show()
