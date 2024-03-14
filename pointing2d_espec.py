import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import convolve2d, savgol_filter
from PIL import Image
from pointing2d_backfiltlib import filterImage, walkDir, norm_gaus2d_ary
from pointing2d_lib import load_points_from_file, points_to_roi
import os
import pointing2d_fit as fit
import csv


folder = "D:\\Bunker C\\2022\\Dec 05\\Pointing Lanex\\Run005\\HE\\"

saving = True

exportDir = "{}\\EXPORTED".format(folder)  # name the subdirectory to export to
if saving and not os.path.exists(exportDir):  # check if it exists
    os.mkdir(exportDir)  # create it if not


points_file =  f"{folder}points_data.txt"
analyse = {k: points_to_roi(v,w=1288,h=964,x_pad=150,y_pad=100) for k, v in load_points_from_file(points_file).items() if len(v) > 0} #is not None}

print(f"found {len(analyse)} rois")

Background = "D:\\Bunker C\\2022\\Dec 05\\Pointing Lanex\\Run005\\HE\\Run005-12052022163736-1.tiff"


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


make_back = True

x_fils = [3,5,53,5,3] # [3,5,3]
kernel = norm_gaus2d_ary(37, 3, 57, 3)

a,b,c,d,e = espec_coeffs['22-12-05']

bin = 1

make_back = True

fmodel = fit.setup_double_2d_gauss_model()

peaks = []

plotfit = False

for i,file in enumerate(list(analyse.keys())):
    print(f"loop {i+1}")
    
    padding = 50

    crop = analyse[file]
    
    x_px = np.linspace(crop[0][0],crop[1][0]+padding,crop[1][0]+padding-crop[0][0],endpoint=True)[::bin]
    
    e_x = ratio_of_quads(x_px,a,b,c,d,e)

    name = file[:-5].split('\\')[-1].split('/')[-1]

    full = np.array(Image.open(file))
    full_back = np.array(Image.open(Background))

    if bin !=1:
        full = full.reshape(int(h/bin), bin, int(w/bin), bin).sum(3).sum(1)
        full_back = full_back.reshape(int(h/bin), bin, int(w/bin), bin).sum(3).sum(1)

    for f in x_fils:  # filter xrays
        full = filterImage(full, f)
        full_back = filterImage(full_back, f)

    h,w = full.shape

    flag_l = flag_r = False
    if crop[1][0] + padding > w:
        print("ROI goes close to the right edge of the image")
        flag_r = True
    if crop[1][1] + padding > h:
        print("ROI goes close to the bottom edge of the image")
        flag_l = True
    
    pad_r = min(crop[1][0] + padding, w)
    pad_l = min(crop[1][1] + padding, h)

    pad_back = full_back[crop[0][1]:pad_l,crop[0][0]:pad_r]
    pad_sig = full[crop[0][1]:pad_l,crop[0][0]:pad_r]

    smothed_shot = convolve2d(pad_sig, kernel, boundary='wrap', mode='same')
    smothed_back = convolve2d(pad_back, kernel, boundary='wrap', mode='same')

    sub = np.subtract(smothed_shot, smothed_back)
    sub = sub - sub.min()

    if make_back:
        win_len = 31
        ord = 3
        v_slice = savgol_filter(np.mean(sub[:,-padding:-1],1),win_len,ord)
        sub = sub - v_slice[:,np.newaxis]   
    sub = sub[:-padding,:-padding]

    w,h = sub.shape

    ssum = sub.sum(0)

    if plotfit and False:
        x2, y2 = np.meshgrid(x, y)

        result = fit.fit_double_gauss2d_lm(x2, y2, sub, fmodel)
        fitted = fmodel.func(x2, y2, **result.best_values)

    if plotfit and False:
        ax[0].contour(x,
                        y,
                        fitted,
                        2,
                        colors='black',
                        extent= (0,n,0,m),#(e_x[0],e_x[-1],0,sub.shape[1]),
                        linewidths=0.8)
    fig,ax = plt.subplots(2,1,sharex=True,tight_layout = True,figsize = (9,12), height_ratios = [w/h,0.6])
    fig.suptitle(name)
    ax[0].imshow(sub, vmin = 0, vmax = sub.max()) #,extent = (e_x[0],e_x[-1],sub.shape[1],0))
    if plotfit and False:
        ax[0].contour(x,
                        y,
                        fitted,
                        2,
                        colors='black',
                        extent= (0,n,0,m),#(e_x[0],e_x[-1],0,sub.shape[1]),
                        linewidths=0.8)
    ax[0].set_yticks([])
    ssum = sub.sum(0)
    ax[1].plot(ssum)
    ax[1].grid('both','both')
    labs = ["{:.2f}".format(e_x[i]) for i in range(0,w,50)]
    locs = [i for i in range(0,w,50)]
    ax[1].set_xticks(locs,labs)
    ax[1].set_xlabel("Energy in MeV")
    ax[1].set_ylabel("Charge (arb. units)")
    ax[0].set_title("total charge (arb. units) = {:.1e}".format(sub.sum(1).sum(0)))
    ax[0].sharex(ax[1])
    
    peaks.append(e_x[np.argmax(ssum)])
    
    if saving:
        saveplot = "{}\\{}_plot".format(exportDir, name.split("\\")[-1].split("/")[-1].split('.')[0])
        savefile = "{}\\{}_lineout.csv".format(exportDir, name.split("\\")[-1].split("/")[-1].split('.')[0])
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

plt.hist(peaks,4)
plt.show()
