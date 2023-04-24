import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from PIL import Image
import os
GoodShot = "D:\\CLPU2_DATA\\CCD_LANEX\\20220520\\ccd180_data_2022-05-20-151913-0036.tif"
Background = "D:\\CLPU2_DATA\\CCD_LANEX\\20220520\\ccd180_data_2022-05-20-151913-0038.tif"

if os.path.exists(GoodShot) and os.path.exists(Background):
    plt.imshow(Image.open(GoodShot))

crop = [[1700,570],[1870,640]]

x_loc = [13,73]
p_loc = [1770,1146]

x_px = np.linspace(crop[0][0],crop[1][0],1+crop[1][0]-crop[0][0],endpoint=True)

m = (x_loc[0]-x_loc[1])/(p_loc[0]-p_loc[1])
c = x_loc[1] - (m*p_loc[1])

x_mm = m*x_px+c

e_x = ratio_of_quads( m*x_px+c,*popt)

fig,ax = plt.subplots()
ax.plot(x_px, e_x)
ax.set_ylabel("MEV")
ax.set_xlabel("px")


import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from scipy.signal import convolve

GoodShot = "D:\\CLPU2_DATA\\CCD_LANEX\\20220520\\ccd180_data_2022-05-20-151913-0036.tif"
Background = "D:\\CLPU2_DATA\\CCD_LANEX\\20220520\\ccd180_data_2022-05-20-151913-0038.tif"

x_fils = [3,5,53,5,3]

if os.path.exists(GoodShot) and os.path.exists(Background):
    shot = np.array(Image.open(GoodShot))[crop[0][1]:crop[1][1],crop[0][0]:crop[1][0]]
    print(shot.shape)
    back = np.array(Image.open(Background))[crop[0][1]:crop[1][1],crop[0][0]:crop[1][0]]
    print(shot.shape)
    for f in x_fils:  # filter xrays
        shot = filterImage(shot, f)
        back = filterImage(back, f)
    
    kernel = norm_gaus2d_ary(1, 3, 1, 3)
    
    shot = convolve(shot, kernel, mode='same')
    back = convolve(back, kernel, mode='same')
    
    
    sub = np.abs(np.subtract(shot, back))
    
    if sub.min() < 0:
        sub = sub - sub.min()
    
    sub = sub - np.percentile(sub,25)
    sub = np.clip(sub,0,sub.max())
    
    fig,ax = plt.subplots()
    ax.imshow(sub, vmin = 0, vmax = sub.max())
    
    labs = ["{:.0f}".format(ratio_of_quads((m*(crop[0][0]+loc))+c,*popt)) for loc in ax.get_xticks()]
    ax.set_xticklabels(labs)
    ax.set_yticks([])
    ax.set_xlabel("Energy in MeV")
    
