import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import convolve2d, savgol_filter
from PIL import Image
from pointing2d_backfiltlib import filterImage, walkDir, norm_gaus2d_ary
from pointing2d_lib import load_dict_from_file, points_to_roi
import os
import pointing2d_fit as fit
import csv
from tqdm import tqdm

import diplib as dip

points_file = rois_file = None
#"""
folder = "D:/Bunker C/Undulator_Masters_Project/DATA" 
Background = f"{folder}/BACKGROUND-03232023140207-4.tiff" 
days = ['23-03-23',
        '23-03-24',
        '23-04-20',
        '23-04-26',
        '23-04-27',
        '23-04-28']
rois_file = f"{folder}/roi_data.txt"
"""

folder  = "D:\\Bunker C\\2022\\Dec 05\\Pointing Lanex\\Run005\\HE"
Background = "D:\\Bunker C\\2022\\Dec 05\\Pointing Lanex\\Run005\\HE\\Run005-12052022163736-1.tiff"
days = ['22-12-05']
points_file = f"{folder}/points_data.txt"
#"""

overwrite = True
saving = True
bin = 1
make_back = True
plotfit = False #True

x_fils = [3,5,53,5,3] # [3,5,3]
kernel_sigmas = [21,31]

#kernel = norm_gaus2d_ary(37, 3, 57, 3)

if points_file is not None:
    analyse = {k: points_to_roi(v,w=1288,h=964,x_pad=150,y_pad=100) for k, v in load_dict_from_file(points_file).items() if len(v) > 0}
elif rois_file is not None:
    analyse = load_dict_from_file(rois_file)
else:
    raise ValueError("No Files To Analyse")

print(f"found {len(analyse)} rois")

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
    '22-12-05' : [19.5487,58230.3,7.48844e7,-3014.01,2.38075e6],
    '23-03-23' :[16.82203932056203,44381.734282360354,50860414.825793914,-2919.0964588725824,2204294.821475454],
    '23-03-24' :[16.82203932056203,44381.734282360354,50860414.825793914,-2919.0964588725824,2204294.821475454],
    '23-04-20' :[9.2080820556058758,35563.53137486182,4.48789354385e7,-2933.2162484668197,2281787.5593547565],
    '23-04-26' :[9.280269716645035,36492.34932605621,48405689.11574387,-3352.4915825373205,2961512.5164166116],
    '23-04-27' :[9.280351087922435,26752.558352428678,25734721.443078351683,-2345.4430,1453419.190032153],
    '23-04-28' :[9.280351087922435,26752.558352428678,25734721.443078351683,-2345.4430,1453419.190032153]
}

fmodel = fit.setup_double_2d_gauss_model()

if plotfit:
    if 'y' not in input("Potting initialised with Fitting ON \nthis is EXTREEMLY SLOW \nconfirm? Y/N: ").lower():
        plotfit = False

for day in days:

    if saving and not os.path.exists(f"{folder}/EXPORTED/"):  # check if it exists
        os.mkdir(f"{folder}/EXPORTED/")  # create it if not


    print(f"{day} : ")    
        
    peaks = []
    a,b,c,d,e = espec_coeffs[day]

    daystr = '20'+''.join(day.split('-'))

    if day == '22-12-05':
        day_keys = analyse
    else:
        day_keys = [f for f in list(analyse.keys()) if daystr in f and analyse[f] is not None]
    
    for i, file in enumerate(tqdm(day_keys)):

        run = file.split('_')[-1].split('-')[0]   
    
        exportDir = f"{folder}/EXPORTED/{day}/{run}"  # name the subdirectory to export to
        if saving and not os.path.exists(exportDir):  # check if it exists
            os.mkdir(exportDir)  # create it if not

        name = file[:-5].split('\\')[-1].split('/')[-1]

        saveplot = "{}\\{}_plot.png".format(exportDir, name.split("\\")[-1].split("/")[-1].split('.')[0])

        if saving and os.path.exists(saveplot):
            if not overwrite:
                continue
            else:
                pass

        padding = 50

        crop = analyse[file]
        
        x_px = np.linspace(crop[0][0],crop[1][0]+padding,crop[1][0]+padding-crop[0][0],endpoint=True)[::bin]
        
        e_x = ratio_of_quads(x_px,a,b,c,d,e)

        full = np.array(Image.open(file))
        full_back = np.array(Image.open(Background))

        if bin !=1:
            full = full.reshape(int(h/bin), bin, int(w/bin), bin).sum(3).sum(1)
            full_back = full_back.reshape(int(h/bin), bin, int(w/bin), bin).sum(3).sum(1)

        for f in x_fils:  # filter xrays
            full = filterImage(full, f)
            full_back = filterImage(full_back, f)

        h,w = full.shape

        pad_r = min(crop[1][0] + padding, w)
        pad_l = min(crop[1][1] + padding, h)

        pad_back = full_back[crop[0][1]:pad_l,crop[0][0]:pad_r]
        pad_sig = full[crop[0][1]:pad_l,crop[0][0]:pad_r]
        
        smothed_shot = dip.Gauss(pad_sig, sigmas=kernel_sigmas[::-1])
        smothed_back = dip.Gauss(pad_back,sigmas=kernel_sigmas[::-1])
            
        sub = np.subtract(smothed_shot, smothed_back)
        sub = sub - sub.min()

        if make_back:
            win_len = 31
            ord = 3
            v_slice = savgol_filter(np.mean(sub[:,-padding:-1],1),win_len,ord)
            sub = sub - v_slice[:,np.newaxis]   
        sub = sub[:-padding,:-padding]

        c_h,c_w = sub.shape

        x = np.linspace(0,c_w,c_w)
        y = np.linspace(0,c_h,c_h)

        ssum = sub.sum(0)

        if plotfit:
            x2, y2 = np.meshgrid(x, y)

            result = fit.fit_double_gauss2d_lm(x2, y2, sub, fmodel)
            fitted = fmodel.func(x2, y2, **result.best_values)

    
        fig,ax = plt.subplot_mosaic([['im','hb'],['vb','.']], 
                                    figsize = (12,9),
                                    width_ratios = [0.6, 0.4], 
                                    height_ratios = [0.7,0.3])

        fig.suptitle(name)
        ax['im'].imshow(sub, vmin = 0, vmax = sub.max()) #, extent = (e_x[0],e_x[-1],sub.shape[1],0))
        
        x1 = max(0,crop[0][0])
        y1 = max(0,crop[0][1])
        y2 = y1 + sub.shape[0]
        x2 = x1 + sub.shape[1]

        ax['im'].set_xticks([0, sub.shape[1] - 1])
        ax['im'].set_xticklabels([x1,x2])
        ax['im'].set_yticks([0, sub.shape[0] - 1])
        ax['im'].set_yticklabels([y1,y2])
        
        if plotfit:
            ax['im'].contour(x,
                            y,
                            fitted,
                            2,
                            colors='black',
                            extent= (0,c_w,0,c_h),#(e_x[0],e_x[-1],0,sub.shape[1]),
                            linewidths=0.8)

        ssum = sub.sum(0)
        hsum = sub.sum(1)

        ax['hb'].plot(hsum,range(y1,y2),'b')
        ax['hb'].set_ylabel('Pixel')
        ax['hb'].set_title("Fully Horizontally Binned")
        ax['hb'].grid('both','both') 
        ax['hb'].set_xlabel("Charge (arb. units)")
        ax['hb'].invert_yaxis()

        ax['vb'].plot(ssum)
        ax['vb'].grid('both','both')
        labs = ["{:.0f}".format(e_x[j]) for j in range(0,c_w,50)]
        locs = [j for j in range(0,c_w,50)]
        ax['vb'].set_xticks(locs,labs)
        ax['vb'].set_xlabel("Energy in MeV")
        ax['vb'].set_ylabel("Charge (arb. units)")
        ax['im'].set_title("total charge (arb. units) = {:.1e}".format(sub.sum(1).sum(0)))
        ax['vb'].set_title("Fully Vertically Binned")
        
        fig.tight_layout()
        peaks.append(e_x[np.argmax(ssum)])
        
        if saving:
            savefile = "{}\\{}_lineout.csv".format(exportDir, name.split("\\")[-1].split("/")[-1].split('.')[0])
            fig.savefig(saveplot,dpi=600)
            with open(savefile, 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(["E","Q"])
                for i, q in enumerate(sub.sum(0)):
                    writer.writerow([ratio_of_quads((((crop[0][0]+i))),a,b,c,d,e),q])

            plt.close(fig)  
            
        else:
            fig.show()
            input("Done")


        histfig, histax = plt.subplots(2,1,sharex=True)
        histfig.suptitle("Calibration and Histogram for {} with {} shots".format(day,len(day_keys)))
        histax[0].plot(e_x, x_px)
        histax[0].set_title("Calibration")
        histax[0].set_ylabel("Pixel")
        histax[1].set_title("Distribution")
        histax[1].set_xlabel("Energy")
        histax[1].set_ylabel("Counts")
        histax[1].hist(peaks,12,density=True)
        if saving:
            savehist = "{}\\EXPORTED\\{}_hist".format(folder, day)
            histfig.savefig(savehist)
            plt.close(histfig)  
            
        else:
            fig.show()
    

if not saving:
    input("done?")

plt.show()
