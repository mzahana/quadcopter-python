from draw_quad import QuadPlot
from spatialmath import SE3, SO3

import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

# Define World frame
# origin
O_W = [0,0,0]
# translation
W = SE3.Trans(O_W)
# Plot world frame
W.plot( frame='W', color="green", length=3, axes=ax)

quadFrame = SE3(1,2,3) * SE3.Rx(30, unit='deg')*SE3.Ry(45, unit='deg')
quadFrame.plot(frame='B', axes=ax, length=0.5)

q = QuadPlot(ax=ax, arm_l=0.45)
q.update(quadFrame.R, quadFrame.t)

plt.show()