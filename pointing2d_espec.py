import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import convolve
from PIL import Image
from pointing2d_backfiltlib import filterImage, walkDir, norm_gaus2d_ary
import os
import pointing2d_fit as fit
import csv

folder = "D:\\Bunker C\\Lanex\\E_Spec_test_data\\042023_2a\\synced_with_spectrometer\\"
files = walkDir(folder)

saving = True



exportDir = "{}\\EXPORTED".format(folder)  # name the subdirectory to export to
if saving and not os.path.exists(exportDir):  # check if it exists
    os.mkdir(exportDir)  # create it if not

GoodShot = files[8]
Background = files[-1]

threshold = 70

plotfit = False

x_fils = [3,5,53,5,3]

kernel = norm_gaus2d_ary(5, 3, 9, 3)

"""
CALIBRATION
"""
#  Fit results:
"""
a = 11.5
b = 24170
c = 32519703
d = -2880
e = 2150460
"""

a = 328.6443391832576
b = -294684.42463444325
c = 68563690.461126
d = 785.6130708255362
e = 167767.36631428992
def ratio_of_quads(x,a,b,c,d,e):
    n = (a*x*x)+(b*x) + c
    d = (x*x) + (d*x) + e 
    y = n/d
    return(y)



bin = 2

l_m = -0.18287037037037038
l_c = 240.49305555555554

analyse = [
    [files[8],[[0,250],[800,450]]],
    [files[22],[[0,250],[800,450]]],
    [files[23],[[0,470],[800,670]]],
    [files[24],[[0,250],[800,450]]],
    [files[26],[[0,370],[800,570]]],
    [files[28],[[0,450],[800,650]]],
    [files[29],[[0,450],[800,650]]]
]

fmodel = fit.setup_double_2d_gauss_model()

for file, crop in analyse:
    x_px = np.linspace(crop[0][0],crop[1][0],1+crop[1][0]-crop[0][0],endpoint=True)[::bin]
    e_x = ratio_of_quads(l_m*x_px+l_c,a,b,c,d,e)
    name = file[:-5].split('\\')[-1]

    shot = np.array(Image.open(file))[crop[0][1]:crop[1][1],crop[0][0]:crop[1][0]]
    back = np.array(Image.open(Background))[crop[0][1]:crop[1][1],crop[0][0]:crop[1][0]]
    n,m = shot.shape
    if bin !=1:
        shot = shot.reshape(int(n/bin), bin, int(m/bin), bin).sum(3).sum(1)
        back = back.reshape(int(n/bin), bin, int(m/bin), bin).sum(3).sum(1)
        n,m = shot.shape
        
    for f in x_fils:  # filter xrays
        shot = filterImage(shot, f)
        back = filterImage(back, f)
    
    shot = convolve(shot, kernel, mode='same')
    back = convolve(back, kernel, mode='same')
    
    sub = np.abs(np.subtract(shot, back))
    
    if sub.min() < 0:
        sub = sub - sub.min()
    
    sub = sub - np.percentile(sub,threshold)
    sub = np.clip(sub,0,sub.max())


    x = range(m)
    y = range(n)

    x2, y2 = np.meshgrid(x, y)

    result = fit.fit_double_gauss2d_lm(x2, y2, sub, fmodel)
    fitted = fmodel.func(x2, y2, **result.best_values)

    fig,ax = plt.subplots(2,1,sharex=True,tight_layout = True)
    fig.suptitle(name)
    ax[0].imshow(sub, vmin = 0, vmax = sub.max())
    if plotfit:
        ax[0].contour(x,
                        y,
                        fitted,
                        2,
                        colors='black',
                        extent=(0, n, 0, m),
                        linewidths=0.8)
    ax[0].set_yticks([])
    ax[1].plot(sub.sum(0))
    labs = ["{:.0f}".format(e_x[int(loc)]) for loc in ax[0].get_xticks()] # ratio_of_quads(((crop[0][0]+loc)),a,b,c,d,e)) for loc in ax.get_xticks()]
    ax[1].set_xticklabels(labs)
    ax[1].grid('both','both')
    ax[1].set_xlabel("Energy in MeV")
    ax[1].set_ylabel("Charge (arb. units)")
    ax[0].set_title("total charge (arb. units) = {:.1e}".format(sub.sum(1).sum(0)))
    ax[0].sharex(ax[1])
    if saving:
        saveplot = "{}\\{}_plot".format(exportDir, name)
        savefile = "{}\\{}_lineout.csv".format(exportDir, name)
        fig.savefig(saveplot)
        with open(savefile, 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(["E","Q"])
            for i, q in enumerate(sub.sum(0)):
                writer.writerow([e_x[i],q])

        plt.close(fig)  
        
    else:
        fig.show()
        input("done?")