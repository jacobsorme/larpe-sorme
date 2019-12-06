import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os

files = os.listdir("dat")
files.sort()
print(files)
surf = np.zeros((len(files),1000))
i = 0
for f in files:
    dat = np.load("dat/"+f)
    surf[i,:] = dat[2,:].T
    i += 1

x = np.arange(len(files)) # fix this
y = np.arange(1000)
x,y = np.meshgrid(x,y)
ax = plt.figure().gca(projection='3d')
ax.view_init(30,-140)
ax.plot_surface(x, y,surf.T, cmap=cm.viridis)
ax.set_xlabel("lambda")
ax.set_ylabel("episode")
ax.set_zlabel("last100mean")


#plt.legend(files)
plt.show()
