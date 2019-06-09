import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import glob

########## plots the 2D contours in 3D ###########
## JUST STARTED TO CODE!! ###

contourdir = "./contour"
mycontours = glob.glob(contourdir+'/*.jpg')
#pts3x = np.empty((29))
#pts3y = np.empty((29))

fig = plt.figure(figsize=(30,30))
ax = fig.add_subplot(2,2,1,projection='3d')

for idx, fname in enumerate(mycontours):
    img = cv2.imread(fname)
    pts3x = np.array([])
    pts3y = np.array([])
    for i in range(1023):
        for k in range(1023):
            if img[i][k][0] != 255:
                np.append(pts3x,i)
                np.append(pts3y,k)
    print(len(pts3x))
    print(len(pts3y))
    ax.scatter(pts3x, pts3y, zs=idx, zdir='y', s=20, c='b')

plt.show()