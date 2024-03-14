# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:21:22 2022

@author: willo

A generic library of supporting functions for __main__

"""
import numpy as np
import os
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
        backgroundData = 0
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

    for file in tifFiles[settings.start:settings.stop:settings.decimate]:
        name = file[:-5].split('\\')[-1]
        
        savefile = "{}\\{}_data".format(exportDir, name)

        pixelData = np.array(PIL.Image.open(file))

        for f in settings.filters:
            pixel_data = backfilt.filterImage(pixelData, f)
        pixeldata = convolve(pixelData, kernel, mode='same')

        if backgroundData is not None:
            if not backgroundData.shape == pixelData.shape:
                print("Background missmatch")
            else:
                A = pixelData.sum()
                B = backgroundData.sum()
                pixelData = np.subtract(
                    pixelData,
                    np.multiply(backgroundData, (settings.background_scale * A / B)))

        noise_scale = np.percentile(pixelData[300:-1, 1:-350], 60)
        pixelData = np.clip(pixelData - noise_scale, 0, np.inf)

        transformed, axis = perspective.TransformToThetaPhi(
            pixelData, np.array(src, np.float32),
            np.array(dst, np.float32))

        zoom_x_lims = [
            int(axis[1] - (settings.zoom_radius * settings.resolution)),
            int(axis[1] + (settings.zoom_radius * settings.resolution))
        ]
        zoom_y_lims = [
            int(axis[0] + (settings.zoom_radius * settings.resolution)),
            int(axis[0] - (settings.zoom_radius * settings.resolution))
        ]

        roi = np.array(transformed[zoom_x_lims[0]:zoom_x_lims[1],
                                    zoom_y_lims[1]:zoom_y_lims[0]])

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
            print("I don't think there are electrons in this image: {}".
                    format(file))
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

            fig, ax = plt.subplots(1, 1)

            ax.imshow(roi,
                        cmap=plt.cm.jet,
                        origin='lower',
                        extent=(x.min(), x.max(), y.min(), y.max()))
            ax.contour(x,
                        y,
                        fitted,
                        2,
                        colors='black',
                        extent=(x.min(), x.max(), y.min(), y.max()),
                        linewidths=0.8)
            
            smaller = '1'
            if result.best_values["sigma_x_1"] > result.best_values["sigma_x_2"]:
                smaller = '2'

            bunch_charge = (result.best_values["amplitude_{}".format(smaller)]*result.best_values["sigma_x_{}".format(smaller)]*result.best_values["sigma_y_{}".format(smaller)])

            ax.set_title('Charge of :{:.1f} [arb. Units] \n'.format(bunch_charge) + r'at $\theta$ = {:.1f}, $\phi$ = {:.1f}'.format(result.best_values["xo_{}".format(smaller)],result.best_values["yo_{}".format(smaller)]))
            name = file[:-5].split('\\')[-1]

            if settings.saving:
                saveplot = "{}\\{}_plot".format(exportDir, name)
                lm.model.save_modelresult(result, savefile)
                fig.savefig(saveplot)
                plt.close(fig)
            else:
                fig.show()
                settings.blockingPlot = True

        
    if settings.saving:
        np.save("{}\\stats".format(exportDir),
                stats,
                allow_pickle=True,
                fix_imports=True)

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

    nbins = int(max(len(stats) / 10, 10))
    pAx = report_figures[-1][0].add_subplot(gs_hist[1, 0], )
    pAx.minorticks_on()
    uux = np.mean(u_x)
    uuy = np.mean(u_y)
    
    sx = np.sqrt(u_x.var())
    sy = np.sqrt(u_y.var())

    u_0_x = u_x-uux
    u_0_y = u_y-uuy 
    pAx.hist2d(u_0_x, u_0_y, nbins, [[-25, 25], [-25, 25]]) 
    pAx.set_aspect('equal')
    pAx.add_patch(Circle((0,0),6,ec='red',fill=False))
 
    xlabels = [item.get_text() if item.get_text() != '0' else r'$\mu_x$' for item in pAx.get_xticklabels()]
    pAx.set_xticklabels(xlabels)

    ylabels = [item.get_text() if item.get_text() != '0' else r'$\mu_y$' for item in pAx.get_yticklabels()]
    pAx.set_yticklabels(ylabels)

    pax_histx = report_figures[-1][0].add_subplot(gs_hist[0, 0], sharex=pAx)
    pax_histy = report_figures[-1][0].add_subplot(gs_hist[1, 1], sharey=pAx)


    pax_histx.axes.get_xaxis().set_visible(False)
    pax_histy.axes.get_yaxis().set_visible(False)

    pax_histx.hist(
        u_0_x,
        nbins,
        density = True,
        color='green',
    )
    pax_histy.hist(
        u_0_y, 
        nbins, 
        density = True,
        color='green', 
        orientation='horizontal'
    )

    xmin, xmax = pax_histx.get_xlim()
    ymin, ymax = pax_histy.get_ylim()
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    px = norm.pdf(x, 0, sx)
    py = norm.pdf(y, 0, sy)

    pax_histx.plot(x, px, 'k', linewidth=2)
    pax_histy.plot(py, y, 'k', linewidth=2)

    xlab = "x location [/mRad]\n $\\mu_x$ = {:.2f},  $\\sigma_x$ = {:.2f}".format(uux,sx)
    ylab = "y location [/mRad]\n $\\mu_y$ = {:.2f},  $\\sigma_y$ = {:.2f}".format(uuy,sy)
    pAx.set_xlabel(xlab)
    pAx.set_ylabel(ylab)
    report_figures[-1][0].set_tight_layout(True)
    
        
    ##
    #   Bunch Emittence:
    #
    s_x = np.array([stats[i][1]['sigma_x_2'] for i in range(len(stats))])
    s_y = np.array([stats[i][1]['sigma_y_2'] for i in range(len(stats))])
    th = np.array([stats[i][1]['theta_2'] for i in range(len(stats))])
    
    report_figures.append([plt.figure(figsize=(9, 12)), 'emittence'])
    maj_ax = report_figures[-1][0].add_subplot(311)
    min_ax = report_figures[-1][0].add_subplot(312)
    th_ax = report_figures[-1][0].add_subplot(313)
    report_figures[-1][0].suptitle("Analysis of the Bunch Sizes")
    report_figures[-1][0].set_tight_layout(True)
    maj_ax.hist(s_x, nbins)
    maj_ax.set_xlabel("Major axis size (sigma) [/mRad] mean:{:.2f}  s.d.:{:.2f}".format(
        s_x.mean(), np.sqrt(s_x.var())))
    min_ax.hist(s_y, nbins)
    min_ax.set_xlabel("Minor axis size (sigma) [/mRad] mean:{:.2f}  s.d:{:.2f}".format(
        s_y.mean(), np.sqrt(s_y.var())))
    th_ax.hist(th, nbins)
    th_ax.set_xlabel("Theta [/deg] mean:{:.2f}  s.d.:{:.2f}".format(
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
    #   Bunch Charge:
    #   TODO: currently in arbitrary units

    amp = np.array([stats[i][1]['amplitude_2'] for i in range(len(stats))])  

    report_figures.append((plt.figure(figsize=(9, 12)), 'charge'))
    amp_scat3d_ax = report_figures[-1][0].add_subplot(projection='3d')
    amp_scat3d_ax.scatter(s_x, s_y, amp)
    amp_scat3d_ax.set_xlabel("Major")
    amp_scat3d_ax.set_ylabel("Minor")
    amp_scat3d_ax.set_zlabel("Amplitude")    

    emit = np.sqrt(np.add(np.power(s_x,2),np.power(s_y,2)))
    point = np.sqrt(np.add(np.power(u_x,2),np.power(u_y,2)))

    report_figures.append((plt.figure(figsize=(9, 12)), 'charge'))
    amp_evp_scat3d_ax = report_figures[-1][0].add_subplot(projection='3d')
    amp_evp_scat3d_ax.scatter(point, emit, amp)
    amp_evp_scat3d_ax.set_xlabel("Pointing")
    amp_evp_scat3d_ax.set_ylabel("Emittence")
    amp_evp_scat3d_ax.set_zlabel("Amplitude")    


    for fig in report_figures:
        if settings.saving:
            fig[0].savefig("{}\\{}_fig".format(exportDir, fig[1]))
        else:
            fig[0].draw()
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

def load_points_from_file(filepath):
    points_dict = {}
    # Read the contents of the file
    with open(filepath, "r") as file:
        # Skip the header
        header_skipped = False
        for line in file:
            # Remove leading and trailing whitespace, including newline characters
            line = line.strip()
            if not header_skipped:
                if line == "points_dict = {":
                    header_skipped = True
                continue
            
            # Check if the line is not empty
            if line and line != "}":
                # Split the line into key and value using ":" as delimiter
                try:
                    key, value = line.split(": {")
                except ValueError as e:
                    raise ValueError("file contains line \n\t" + line.split(": {")) from e
                # Remove leading and trailing whitespace from key and value
                key = key.strip("'")
                value = '{' + value
                value = eval(value.strip())
                # Add key-value pair to the dictionary
                points_dict[key] = value
            elif line == "}":
                # End of the dictionary
                break
    return points_dict


if __name__ == "__main__":
    print("this is not the main file")