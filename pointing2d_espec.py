import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import convolve
from PIL import Image
from pointing2d_backfiltlib import filterImage, walkDir, norm_gaus2d_ary
import os
import pointing2d_fit as fit
import csv


folder = "D:\\Bunker C\\2022\\Dec 05\\Pointing Lanex\\Run005\\"
#folder = "D:\\Bunker C\\Lanex\\E_Spec_test_data\\042023_2a\\synced_with_spectrometer\\"

#files = walkDir(folder)

saving = False


exportDir = "{}\\EXPORTED".format(folder)  # name the subdirectory to export to
if saving and not os.path.exists(exportDir):  # check if it exists
    os.mkdir(exportDir)  # create it if not

#GoodShot = files[8]
#Background = files[-1]

threshold = 70

plotfit = True

x_fils = [3,5,53,5,3]

kernel = norm_gaus2d_ary(11, 3, 17, 3)

"""
CALIBRATION
"""
def ratio_of_quads(x,a,b,c,d,e):
    n = (a*x*x)+(b*x) + c
    d = (x*x) + (d*x) + e 
    y = n/d
    return(y)

# espec calibration lookup
# { 'DAY' : [a,b,c,d,e] }
espec_coeffs = {
    '22-12-05' : [19.5487,58230.3,7.48844e7,-3014.01,2.38075e6],# [11.5, 24170, 32519703, -2880, 2150460], ## ???? im doubtful of this calibration
    '23-03-23' :[16.82203932056203,44381.734282360354,50860414.825793914,-2919.0964588725824,2204294.821475454],
    '23-03-24' :[16.82203932056203,44381.734282360354,50860414.825793914,-2919.0964588725824,2204294.821475454],
    '23-04-20' :[9.2080820556058758,35563.53137486182,4.48789354385e7,-2933.2162484668197,2281787.5593547565],
    '23-04-26' :[9.280269716645035,36492.34932605621,48405689.11574387,-3352.4915825373205,2961512.5164166116],
    '23-04-27' :[9.280351087922435,26752.558352428678,25734721.443078351683,-2345.4430,1453419.190032153],
    '23-04-28' :[9.280351087922435,26752.558352428678,25734721.443078351683,-2345.4430,1453419.190032153]
}



plotcalibs = True
if plotcalibs:
    X = np.linspace(1,1000,1000)

    fig,ax = plt.subplots(1,1)
    for day in espec_coeffs.keys():
        a,b,c,d,e = espec_coeffs[day]
        ax.plot(X,ratio_of_quads(X,a,b,c,d,e),label=day)
    ax.grid("both","both")
    ax.legend(loc='best')

    ax.set_xlabel("Pixel")
    ax.set_ylabel("Energy")
    ax.set_title(f"Espec Calibrations")
    fig.show()

a,b,c,d,e = espec_coeffs['22-12-05']

bin = 1

analyse = {
    "D:\\Bunker C\\2022\\Dec 05\\Pointing Lanex\\Run005\\HE\\Run005-12052022164625-530.tiff":[[250,450],[800,750]],
    "D:\\Bunker C\\2022\\Dec 05\\Pointing Lanex\\Run005\\HE\\Run005-12052022171508-2240.tiff":[[200,450],[650,750]]
}
Background = "D:\\Bunker C\\2022\\Dec 05\\Pointing Lanex\\Run005\\HE\\Run005-12052022163736-1.tiff"

fmodel = fit.setup_double_2d_gauss_model()

for file in analyse.keys():
    crop = analyse[file]

    print(f"cropping {file.split('/')[-1]}  to \n x  [{crop[0][0]}   to  {crop[1][0]}] \n y  [{crop[0][1]}   to  {crop[1][1]}]")

    x_px = np.linspace(crop[0][0],crop[1][0],crop[1][0]-crop[0][0],endpoint=True)[::bin]

    e_x = ratio_of_quads(x_px,a,b,c,d,e)
    """  
    fig_c,ax_c = plt.subplots(1,1)
    ax_c.plot(x_px,e_x)
    fig_c.show()
    input("cont...")
    """
    name = file[:-5].split('\\')[-1]

    shot = np.array(Image.open(file))[crop[0][1]:crop[1][1],crop[0][0]:crop[1][0]]

    print(shot.shape)

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

    fig,ax = plt.subplots(2,1,sharex=True,tight_layout = True,figsize = (9,12), height_ratios = [n/m,0.6])
    fig.suptitle(name)
    ax[0].imshow(sub, vmin = 0, vmax = sub.max())#,extent = (e_x[0],e_x[-1],sub.shape[1],0))
    if plotfit:
        ax[0].contour(x,
                        y,
                        fitted,
                        2,
                        colors='black',
                        extent= (0,n,0,m),#(e_x[0],e_x[-1],0,sub.shape[1]),
                        linewidths=0.8)
    ax[0].set_yticks([])
    ax[1].plot(sub.sum(0))
    ax[1].grid('both','both')
    labs = ["{:.2f}".format(e_x[i]) for i in range(0,m,50)]
    locs = [i for i in range(0,m,50)]
    ax[1].set_xticks(locs,labs)
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
                writer.writerow([ratio_of_quads((((crop[0][0]+i))),a,b,c,d,e),q])

        plt.close(fig)  
        
    else:
        fig.show()
        
if not saving:
    input("done?")