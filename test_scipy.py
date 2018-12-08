# from  scipy.misc import imread,imsave,imresize
# img_path='/Users/wywy/Desktop/bg.jpg'
# img=imread(img_path)
# print(img.shape)
import numpy as np


from scipy import special


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
def drumhead_height(n, k, distance, angle, t):
   kth_zero = special.jn_zeros(n, k)[-1]
   return np.cos(t) * np.cos(n*angle) * special.jn(n, distance*kth_zero)
theta = np.r_[0:2*np.pi:50j]
radius = np.r_[0:1:50j]
x = np.array([r * np.cos(theta) for r in radius])
print(x)
y = np.array([r * np.sin(theta) for r in radius])
print(y)
z = np.array([drumhead_height(1, 1, r, theta, 0.5) for r in radius])
print(z)



fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


