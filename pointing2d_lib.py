# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:21:22 2022

@author: willo

A generic library of supporting functions for pointing 2d

"""
import numpy as np
import os, sys, json
import PIL
from scipy.stats import norm
from scipy.signal import convolve
import pointing2d_settings as settings
import pointing2d_backfiltlib as backfilt
import pointing2d_perspective as perspective
import pointing2d_fit as fit
from pointing2d_fit import lm_double_gaus2d # Ineeded for loading models from file 
import lmfit as lm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import rc
from tqdm import tqdm
from scipy.optimize import curve_fit
import pickle

font = {'family': 'Corbel', 'weight': 'normal', 'size': 14}

rc('font', **font)


def get_background(backgroundPath = None):
    """
    Returns the data array from the specified background image file 

    Returns
    -------
    backgroundData : np.array() or 0
        thedata array for the background that will be subtracted from the images

    """
    if backgroundPath is not None:
        background = backgroundPath
    else:
       background = settings.background 
    if background is not None:
        kernel = eval(settings.kernel)
        backgroundData = np.array(PIL.Image.open(background))
        for f in settings.filters:
            backgroundData = backfilt.filterImage(backgroundData, f)

        backgroundData = convolve(backgroundData, kernel, mode='same')
        noise_scale = np.percentile(backgroundData, settings.background_clip)
        backgroundData = np.clip(backgroundData - noise_scale, 0, np.inf)

    else:
        backgroundData = None
    return(backgroundData)


def check_calibration_transformation(exportDir, src, dst):
    """
    A small function for sanity checking the generated transformation on the iluminated image of the lanex
     
    Parameters
    ----------
    exportDir : str
        the directory in which the output should be saved if saving
    src : np.array[,float32]
        source points in pixel coordinates
    dst : np.array[,float32]
        destination points in theta, phi

    Returns
    -------
    None
    """
    pixelData = np.array(
        np.array(PIL.Image.open(settings.pointingCalibrationImage)))
    saveDir = None
    if settings.saving:
        saveDir = exportDir
    perspective.check_transformation(pixelData, src, dst, settings.units,
                                     settings.resolution,
                                     settings.zoom_radius, saveDir)

def integrate_gausians(x2, y2, result, src, dst):
    """
    A function that generates data from the plots the fitted gausians and transforms them back into the camera frame to integrate
    TODO: UNUSED
     
    Parameters
    ----------
    x2 : np.array
        a 2d array of x coordinates as returned from numpy.from meshgrid
    y2 : np.array
        a 2d array of y coordinates as returned from numpy.from meshgrid
    result : ModelResult
        fit returned from LMFIT
        see https://lmfit.github.io/lmfit-py/model.html#lmfit.model.ModelResult
    src : np.array[,float32]
        source points in pixel coordinates
    dst : np.array[,float32]
        destination points in theta, phi

    Returns
    -------
    Integral: array[float]
        the integrals of the two isolated gaussians
    """
    
    gaussians = [] # [ [amplitude, offset, xo, yo, theta, sigma_x, sigma_y] ] 
    integrals = [] # [ float ]

    if result.model.name == "Model(lm_gaus2d)":
        # decompose the two gaussians
        gauss = [result.best_values["amplitude"],0,result.best_values["xo"],result.best_values["yo"],result.best_values["sigma_x"],result.best_values["sigma_y"],result.best_values["theta"]]

    elif result.model.name == "Model(lm_double_gaus2d)":
        # zero the offset
        gaussians.append( [result.best_values["amplitude_1"],0,result.best_values["xo_1"],result.best_values["yo_1"],result.best_values["sigma_x_1"],result.best_values["sigma_y_1"],result.best_values["theta_1"]] )
        gaussians.append( [result.best_values["amplitude_2"],0,result.best_values["xo_2"],result.best_values["yo_2"],result.best_values["sigma_x_2"],result.best_values["sigma_y_2"],result.best_values["theta_2"]] )

    else:
        print("Sorry Integration of Gausian Bunches not Implimented for the Model : {}".format(result.model.name))
        integrals.append(0)
    
    if len(gaussians) != 0:
        for gaus in gaussians:
            integrals.append(gaus[0]*gaus[4]*gaus[5]*np.sqrt(np.pi * 2))
    return(integrals)

def generate_stats(exportDir, src, dst, backgroundData=None):
    fmodel = fit.setup_double_2d_gauss_model()

    tifFiles = backfilt.walkDir(settings.targetDir)

    kernel = eval(settings.kernel)

    stats = []

    if settings.verbose: print(f"Found {len(tifFiles)} Tiff Files")

    for file in tqdm(tifFiles[settings.start:settings.stop:settings.decimate]):
        name = file[:-5].split('\\')[-1].split('/')[-1]
        
        savefile = "{}\\{}_data".format(exportDir, name)

        pixelData = np.array(PIL.Image.open(file))

        if os.path.exists(f"{savefile}") and not settings.overwrite:
            print(f"found fitresults for {savefile}")
            result = lm.model.load_modelresult(f"{savefile}")

        else:
            if settings.verbose: print(f"using {len(settings.filters)} x-ray filters")
            for f in settings.filters:
                pixel_data = backfilt.filterImage(pixelData, f)
            if settings.verbose: print(f"convolving with {settings.kernel}")
            pixelData = convolve(pixelData, kernel, mode='same')

            if backgroundData is not None:
                if settings.verbose: print(f"Subtracting Backgrounds")
                if not backgroundData.shape == pixelData.shape:
                    print("Background missmatch")
                else:
                    A = pixelData.sum()
                    B = backgroundData.sum()
                    pixelData = np.subtract(
                        pixelData,
                        np.multiply(backgroundData, (settings.background_scale * A / B)))

            noise_scale = np.percentile(pixelData[30:-1, 1:-35], settings.background_clip)
            pixelData = np.clip(pixelData - noise_scale, 0, np.inf)

            transformed, axis = perspective.TransformToThetaPhi(
                pixelData, np.array(src, np.float32),
                np.array(dst, np.float32))

            zoom_x_lims = [
                max(0,int(axis[1] - (settings.zoom_radius * settings.resolution))),
                int(axis[1] + (settings.zoom_radius * settings.resolution))
            ]
            zoom_y_lims = [
                int(axis[0] + (settings.zoom_radius * settings.resolution)),
                max(0,int(axis[0] - (settings.zoom_radius * settings.resolution)))
            ]

            roi = np.array(transformed[zoom_x_lims[0]:zoom_x_lims[1],zoom_y_lims[1]:zoom_y_lims[0]])

            if settings.plotBackgroundSubtraction:
                roi_pre = roi

            for region in settings.ignore_regions:
                try:
                    roi[region[0][0]:region[1][0],
                        region[0][1]:region[1][1]] = np.percentile(roi, 10)
                except (IndexError):
                    print("ignore region {} is invalid".format(region))
            for f in settings.filters:
                roi = backfilt.filterImage(roi, f)

            if np.max(roi) > settings.ignore_ptvs_below * np.mean(roi):

                if settings.plotBackgroundSubtraction:
                    cbpad = 0
                    cbscale = 0.6
                    rfig, rax = plt.subplots(1, 2, figsize=(12, 6))
                    im1 = rax[0].imshow(roi_pre,
                                        vmin=np.min(roi),
                                        vmax=np.max(roi))
                    cax1 = rfig.colorbar(im1,
                                            ax=rax[0],
                                            pad=cbpad,
                                            shrink=cbscale,
                                            location='right')

                    im2 = rax[1].imshow(roi,
                                        vmin=np.min(roi),
                                        vmax=np.max(roi))
                    cax2 = rfig.colorbar(im2,
                                            ax=rax[1],
                                            pad=cbpad,
                                            shrink=cbscale,
                                            location='right')

                x = np.linspace(-settings.zoom_radius, settings.zoom_radius, roi.shape[1])
                y = np.linspace(-settings.zoom_radius, settings.zoom_radius, roi.shape[0])

                x2, y2 = np.meshgrid(x, y)

                result = fit.fit_double_gauss2d_lm(x2, y2, roi, fmodel)

            else:
                print("I don't think there are electrons in this image: {}".format(file))
                print("max = {}".format(np.max(roi)))
                print("mean = {}".format(np.mean(roi)))
                print("med = {}".format(np.median(roi)))
                result = None
                x2 = None
                y2 = None
                roi = None
        if result is not None:

            fitted = fmodel.func(x2, y2, **result.best_values)

            stats.append(
                [result.rsquared, result.best_values, result.covar])

            fig, ax = plt.subplots(1, 1,figsize=(6,6))

            im = ax.imshow(roi,
                        cmap=plt.cm.jet,
                        origin='lower',
                        extent=(x.min(), x.max(), y.min(), y.max()))
            ax.contour(x,
                        y,
                        fitted,
                        4,
                        colors='black',
                        extent=(x.min(), x.max(), y.min(), y.max()),
                        linewidths=0.3)
         
            fig.colorbar(im,ax=ax)

            V_C = 2 * np.pi * result.best_values['amplitude_1'] * result.best_values['sigma_x_1'] * result.best_values['sigma_y_1']
            V_B = 2 * np.pi * result.best_values['amplitude_2'] * result.best_values['sigma_x_2'] * result.best_values['sigma_y_2']
 
            ax.set_title(f"\n BUNCH: [A:{result.best_values['amplitude_2']:.1f},"+r" $\sigma_x$"+f":{result.best_values['sigma_x_2']:.2f},"+r" $\sigma_y$"+f":{result.best_values['sigma_y_2']:.2f}] \n Integral {V_B:.0f} at ["+r"$\theta_x$"+f":{result.best_values['xo_1']:.1f},"+r"$\theta_y$"+f":{result.best_values['yo_1']:.1f}]" +
                         f"\n CLOUD: [A:{result.best_values['amplitude_1']:.1f},"+r" $\sigma_x$"+f":{result.best_values['sigma_x_1']:.2f},"+r" $\sigma_y$"+f":{result.best_values['sigma_y_1']:.2f}] \n Integral {V_C:.0f} at ["+r"$\theta_x$"+f":{result.best_values['xo_2']:.1f},"+r"$\theta_y$"+f":{result.best_values['yo_2']:.1f}]")
            
            ax.set_xlabel(r"$\theta_x$")
            ax.set_ylabel(r"$\theta_y$")

            name = file[:-5].split('\\')[-1]

            fig.tight_layout()

            if settings.saving:
                saveplot = "{}\\{}_plot".format(exportDir, name)
                lm.model.save_modelresult(result, savefile)
                fig.savefig(saveplot, dpi=600)
                plt.close(fig)
            else:
                fig.show()
                settings.blockingPlot = True

    if settings.saving:
        with open("{}\\stats.pickle".format(exportDir), 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return(stats)

def generate_report(stats, exportDir):
    report_figures = []
    ##
    #   Pointing:
    #
    u_x = np.array([stats[i][1]['xo_2'] for i in range(len(stats))])
    u_y = np.array([stats[i][1]['yo_2'] for i in range(len(stats))])

    report_figures.append([plt.figure(figsize = (9,9), dpi=360), 'pointing']) 
    report_figures[-1][0].set_tight_layout(True)

    gs_hist = report_figures[-1][0].add_gridspec(2,
                                                 2,
                                                 width_ratios=(4, 1),
                                                 height_ratios=(1, 4),
                                                 left=0.1,
                                                 right=0.9,
                                                 bottom=0.1,
                                                 top=0.9,
                                                 wspace=0.05,
                                                 hspace=0.05)

    report_figures[-1][0].set_size_inches(9,9)
    
    nbins = int(max(np.sqrt(len(stats)), 15))
    pAx = report_figures[-1][0].add_subplot(gs_hist[1, 0], )    
    
    pAx.minorticks_on()
    uux = np.mean(u_x)
    uuy = np.mean(u_y)
    
    u_0_x = u_x-uux
    u_0_y = u_y-uuy 
    pAx.hist2d(u_0_x, u_0_y, nbins, [[-settings.zoom_radius, settings.zoom_radius ], [-settings.zoom_radius, settings.zoom_radius]])
    pAx.set_aspect('equal')
    #pAx.add_patch(Circle((0,0),6,ec='red',fill=False))
 
    xlabels = [item.get_text() if item.get_text() != '0' else r'$\mu_x$' for item in pAx.get_xticklabels()]
    pAx.set_xticklabels(xlabels)

    ylabels = [item.get_text() if item.get_text() != '0' else r'$\mu_y$' for item in pAx.get_yticklabels()]
    pAx.set_yticklabels(ylabels)

    pax_histx = report_figures[-1][0].add_subplot(gs_hist[0, 0], sharex=pAx)
    pax_histy = report_figures[-1][0].add_subplot(gs_hist[1, 1], sharey=pAx)

    pax_text =  report_figures[-1][0].add_subplot(gs_hist[0, 1])
    pax_text.axis('off')  # Turn off axis for the text subplot

    pax_histx.axes.get_xaxis().set_visible(False)
    pax_histy.axes.get_yaxis().set_visible(False)

    xcount, xbins,_ = pax_histx.hist(
        u_0_x,
        nbins,
        density = True,
        color='green',
    )

    xbins=xbins[:-1]+(xbins[1]-xbins[0])/2

    ycount, ybins,_ = pax_histy.hist(
        u_0_y, 
        nbins, 
        density = True,
        color='green', 
        orientation='horizontal'
    )
    ybins = ybins[:-1] + (ybins[1] - ybins[0]) / 2
    xmin, xmax = pax_histx.get_xlim()
    ymin, ymax = pax_histy.get_ylim()

    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)

    def gauss (x, A, s):
        return A*np.exp(-(x/s)**2)

    try:
        poptx, pcovx = curve_fit(gauss, xbins, xcount, maxfev = 6000)
    except RuntimeError as E:
        print("WARNING", E)
        popt_x = [np.sqrt(2*np.pi*u_x.var()),np.sqrt(u_x.var())]
    try:
        popty, pcovy = curve_fit(gauss, ybins, ycount)
    except RuntimeError as E:
        print("WARNING", E)
        popt_x = [np.sqrt(2*np.pi*u_y.var()),np.sqrt(u_y.var())]


    px = gauss(x,*poptx)
    py = gauss(y,*popty)

    pax_histx.plot(x, px, 'k', linewidth=2)
    pax_histy.plot(py, y, 'k', linewidth=2)

    xlab = r"$\theta_x$ [mRad]"
    ylab = r"$\theta_y$ [mRad]"

    rep_str = "$\\mu_x$ = {:.2f},\n$\\sigma_x$ = {:.2f}\n\n$\\mu_y$ = {:.2f},\n$\\sigma_y$ = {:.2f}".format(*poptx, *popty)
    pax_text.text(0.1, 0.6, rep_str, fontsize=10, verticalalignment='top', family='monospace')

    pAx.set_xlabel(xlab)
    pAx.set_ylabel(ylab)
    report_figures[-1][0].set_tight_layout(True)
    
        
    ##
    #   Bunch Emittence:
    #
    s_x = np.array([stats[i][1]['sigma_x_2'] for i in range(len(stats))])
    s_y = np.array([stats[i][1]['sigma_y_2'] for i in range(len(stats))])
    th = np.array([stats[i][1]['theta_2'] for i in range(len(stats))])
    
    report_figures.append([plt.figure(figsize=(9, 12)), 'emittence_b'])
    maj_ax = report_figures[-1][0].add_subplot(311)
    min_ax = report_figures[-1][0].add_subplot(312)
    th_ax = report_figures[-1][0].add_subplot(313)
    report_figures[-1][0].suptitle("Analysis of the Bunch Sizes")
    report_figures[-1][0].set_tight_layout(True)


    maj_ax.hist(s_x, nbins)
    mxaxlab = "$\sigma_{major}$"+"[mRad] mean:{:.2f}  s.d.:{:.2f}".format(s_x.mean(), np.sqrt(s_x.var()))
    maj_ax.set_xlabel(mxaxlab)
    min_ax.hist(s_y, nbins)
    minxaxlab = "$\sigma_{minor}$"+"[mRad] mean:{:.2f}  s.d:{:.2f}".format(s_y.mean(), np.sqrt(s_y.var()))
    min_ax.set_xlabel(minxaxlab)


    th = ((th+90)%180)-90 # Center around 0
    th_ax.hist(th, nbins)
    th_ax.set_xlabel("$\phi [^\circ]$ (from x to major axis) mean:{:.2f}  s.d.:{:.2f}".format(
        th.mean(), np.sqrt(th.var())))

    ##
    #   Bunch Twist:
    #  

    report_figures.append((plt.figure(figsize=(9, 12)), 'twist'))
    th_scat3d_ax = report_figures[-1][0].add_subplot(projection='3d')
    th_scat3d_ax.scatter(s_x, s_y, th)
    th_scat3d_ax.set_xlabel("Major")
    th_scat3d_ax.set_ylabel("Minor")
    th_scat3d_ax.set_zlabel("Theta")



        ##
    #   Cloud Emittence:
    #
    s_x = np.log10(np.array([stats[i][1]['sigma_x_1'] for i in range(len(stats))]))
    s_y = np.log10(np.array([stats[i][1]['sigma_y_1'] for i in range(len(stats))]))
    th = np.array([stats[i][1]['theta_1'] for i in range(len(stats))])
    
    report_figures.append([plt.figure(figsize=(9, 12)), 'emittence_c'])
    maj_ax = report_figures[-1][0].add_subplot(311)
    min_ax = report_figures[-1][0].add_subplot(312)
    th_ax = report_figures[-1][0].add_subplot(313)
    report_figures[-1][0].suptitle("Analysis of the Cloud Sizes")
    report_figures[-1][0].set_tight_layout(True)


    maj_ax.hist(s_x, nbins)
    mxaxlab = "$log_{10}(\sigma_{major})$"+"[log(mRad)] mean:{:.2f}  s.d.:{:.2f}".format(s_x.mean(), np.sqrt(s_x.var()))
    maj_ax.set_xlabel(mxaxlab)
    min_ax.hist(s_y, nbins)
    minxaxlab = "$log_{10}(\sigma_{minor})$"+"[log(mRad)] mean:{:.2f}  s.d:{:.2f}".format(s_y.mean(), np.sqrt(s_y.var()))
    min_ax.set_xlabel(minxaxlab)


    th = ((th+90)%180)-90 # Center around 0
    th_ax.hist(th, nbins)
    th_ax.set_xlabel("$\phi [^\circ]$ (from x to major axis) mean:{:.2f}  s.d.:{:.2f}".format(
        th.mean(), np.sqrt(th.var())))



    ##
    #   Bunch Amplitude and Contrast:
    #  
    def gaussian_volume(A, S_x, S_y):
        return A * S_x * S_y * 2 * np.pi
    
    ratio = []
    V_1 = []
    V_2 = []
    for i, shot in enumerate(stats):
        V1 = gaussian_volume(shot[1]['amplitude_1'],shot[1]['sigma_x_1'],shot[1]['sigma_y_1'])
        V2 = gaussian_volume(shot[1]['amplitude_2'],shot[1]['sigma_x_2'],shot[1]['sigma_y_2'])
        if V1>0 and V2>0:
            V_1.append(np.log10(V1))
            V_2.append(np.log10(V2))
            if V1/V2 < 100:
                ratio.append(V1/V2)

    report_figures.append((plt.figure(figsize=(9, 12)), 'Charge Ratio'))

    V1ax = report_figures[-1][0].add_subplot(3,1,1)
    V1ax.hist(V_1,bins=30)
    V1ax.set_title("Log_10 of the volume under the wider gaussian g_1")

    V2ax = report_figures[-1][0].add_subplot(3,1,2)
    V2ax.hist(V_2,bins=30)
    V2ax.set_title("Log_10 of the volume under the thinner gaussian g_2")

    Rax = report_figures[-1][0].add_subplot(3,1,3)
    Rax.hist(ratio,bins=30)
    #Rax.set_xlim(0,100)
    Rax.set_title("ratio of Volumes where less than 500")


    for fig in report_figures:
        if settings.saving:
            fig[0].savefig("{}\\{}_fig".format(exportDir, fig[1]),dpi=600)
        else:
            fig[0].show()
            settings.blockingPlot = True

def save_u16_to_tiff(imDatIn, size, tiff_filename, norm = True):
    """
    ## https://blog.itsayellow.com/technical/saving-16-bit-tiff-images-with-pillow-in-python/# ##
     
    Since Pillow has poor support for 16-bit TIFF, we make our own
    save function to properly save a 16-bit TIFF.
    """
    # IF NORMALISING, RESCALE IMAE TO FILL 16 BIT 

    if norm:
        imDatIn = (imDatIn/imDatIn.max()) * (2**16 - 1)

    u16in = np.uint16(imDatIn)

    # write 16-bit TIFF image

    # PIL interprets mode 'I;16' as "uint16, little-endian"
    img_out = PIL.Image.new('I;16', size)

    outpil = u16in.astype(u16in.dtype.newbyteorder("<")).tobytes()
    
    img_out.frombytes(outpil)
    img_out.save(tiff_filename)

def points_to_roi(points,w,h,x_pad=150,y_pad=100):
    if len(points.items())==0:
        raise ValueError("Expects dict of points (int,int,bool) (x,y,good/bad) ")

    x_min = y_min = 10000
    x_max = y_max = 0

    for [lab,[x,y,good]] in points.items():
        #x = point[1][0]
        #y = point[1][1]
        #good = point[1][2]
        x_min = max(min(x_min,(x-(1.5*x_pad) if good else x-(3*x_pad))),0) # extra low E padding 
        x_max = min(max(x_max,(x+x_pad if good else x+(2*x_pad))),w)
        y_min = max(min(y_min,(y-y_pad if good else y-(2*y_pad))),0)
        y_max = min(max(y_max,(y+y_pad if good else y+(2*y_pad))),h)

    return([[x_min,y_min],[x_max,y_max]])

def load_dict_from_file(filepath):
    points_dict = {}
    # Read the contents of the file
    with open(filepath, "r") as file:
        # Skip the header
        header_skipped = False
        for line in file:
            # Remove leading and trailing whitespace, including newline characters
            line = line.strip()
            if not header_skipped:
                if line.endswith(" = {"):
                    header_skipped = True
                continue
            
            # Check if the line is not empty
            if line and line != "}":
                # Split the line into key and value using ":" as delimiter
                try:
                    lineparts = line.split(":")
                    key = ":".join(lineparts[0:2]) # C:/path 
                    value = ":".join(lineparts[2:])
                except ValueError as e:
                    raise ValueError("file contains line \n\t" + line) from e
                # Remove leading and trailing whitespace from key and value
                key = key.strip("'")
                value = eval(value.strip())
                # Add key-value pair to the dictionary
                points_dict[key] = value
            elif line == "}":
                # End of the dictionary
                break
    return points_dict

def update_user_settings(input_deck_path=None):
    # If no input_deck_path is provided, check sys.argv for it
    if input_deck_path is None:
        if len(sys.argv) > 1:
            input_deck_path = sys.argv[1]
        else:
            print("Warning: No input deck file provided,\nRunning Example Input")
            print("Usage: python pointing2d_main.py <input_deck_path>")
            input_deck_path = "./EXAMPLES/LPL_Settings.json"

    maindir = os.getcwd()

    input_deck_path = os.path.abspath(input_deck_path)

    print(f"running with {input_deck_path=}")

    os.chdir(os.path.dirname(input_deck_path)) # allow for relative paths in input deck
    print(os.getcwd())
    # Load the input_deck (assuming it's a JSON file for this example)
    with open(input_deck_path, 'r') as f:
        input_deck = json.load(f)
    
    # Iterate over the input settings and apply them to the settings module
    for setting_name, setting_value in input_deck.items():
        if hasattr(settings, setting_name):
            if isinstance(setting_value,str):
                if setting_value == '.' or setting_value.startswith(".\\") or setting_value.startswith("..\\"):
                    setting_value = os.path.abspath(setting_value)
            setattr(settings, setting_name, setting_value)

        else:
            print(f"Warning: {setting_name} not found in settings.")

    os.chdir(maindir)

if __name__ == "__main__":
    print("this is not the main file")