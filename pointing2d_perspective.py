# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:21:22 2022

@author: willo

A library of functions to handle perspectoive transformations of the images to analyse the pointing 

"""
import os
import PIL
import numpy as np
import math as m
import matplotlib.pyplot as plt
import matplotlib.transforms as trans
import matplotlib.ticker as tick
from cv2 import getPerspectiveTransform, perspectiveTransform, warpPerspective
import pointing2d_settings as settings
import pointing2d_fit as fit
from scipy.interpolate import bisplrep, bisplev

def ceil2(num):
    """
    a simple function for rounding up to a power of 2

    Parameters:
    ---------
    num : float
        the number to round
    
    Returns :
    ---------
    num : float
        the next power of 2    
    """
    return(np.pow(ceil(np.log2(num)),2))


def getTransform(pixelDataShape,src,dst):
    """
    generates or loads a perspective transformation from known points
    and a weights array to prescale the original image

    WARN: this projects the points to a flat plane not a sphere and is unsuitable for large angles
          this is also not integral preserving ):

    Parameters
    ----------
    src : np.array[,float32]
        source points in pixel coordinates
    dst : np.array[,float32]
        destination points in theta, phi
    
    Returns
    ----------
    warp_transform : np.array
        transformation matrix
    
    """
        
    if settings.transformation is not None:
        if settings.verbose: print("Loading existing transformation and normalisation matricies ... ", end = '')
        warp_transform = settings.transformation[0]
        weights = settings.transformation[1]
        if settings.verbose: print("Done")
    else: 
        if settings.verbose: print("calculating transformation ... ", end = '')
        warp_transform = getPerspectiveTransform(np.array(src,np.float32), np.array(dst,np.float32))
        if settings.verbose: print("Done")

        if settings.verbose: print("calculating weights ... ", end = '')
        weights = GenerateWeightArray(pixelDataShape, warp_transform)
        if settings.verbose: print("Done")

        if settings.verbose: print("Recording to temp ... ", end = '')
        settings.transformation = [warp_transform, weights]
        if settings.verbose: print("Done")
    return warp_transform, weights 


def TransformToThetaPhi(pixelData, src, dst, weighted = True):
    """
    generates a perspective transformation from known points and applys it to the image

    Parameters
    ----------
    pixelData : 2d array
        the image data
    src : np.array[,float32]
        source points in pixel coordinates
    dst : np.array[,float32]
        destination points in theta, phi

    Returns
    -------
    out_im : 2d array
        the warped image data
    axis : tuple(float,float)
        the pixel coordinates of the axis given this transformation
    """

    assert len(src) == 4
    assert len(dst) == 4

    ## padding around the output points:

    lpad = 1.01
    tpad = 1.01
    rpad = 1.2
    bpad = 0.92

    dstmin = [tpad * min(dst[:, 0]), lpad * min(dst[:, 1])]
    dst = np.float32([[d[0] - dstmin[0], d[1] - dstmin[1]] for d in dst])

    dstmax = np.array([max(dst[:, 0]) * bpad, max(dst[:, 1]) * rpad], int)
    axis = np.float32([0 - dstmin[0], 0 - dstmin[1]])

    warp_transform, weights = getTransform(pixelData.shape,src,dst)

    pixelData = np.multiply(pixelData,weights)

    if settings.verbose: print("Transforming Image")
    out_im = warpPerspective(pixelData, warp_transform, (dstmax[1], dstmax[0]))
    if settings.verbose: print("Done")

    return (out_im, axis)


def GenerateWeightArray(pixelDataShape, warp_transform, plotting = False):
    """
    Generates an that contains a weight value for each pixel of the input image to preserve integrals after the perspective warp

    Parameters
    ----------
    pixelData: 2d array
        raw image data
    src : np.array[,float32]
        source points in pixel coordinates
    dst : np.array[,float32]
        destination points in theta, phi

    Returns
    -------
    weights : 2d array
        the grid of pixel weights
    """
    num_x_samples = 200
    num_y_samples = 160

    x_len = pixelDataShape[0]
    y_len = pixelDataShape[1]

    x_start = x_len/(2*num_x_samples)
    x_stop = x_len - x_start

    y_start = y_len/(2*num_y_samples)
    y_stop = y_len - y_start

    sample_x = np.linspace(x_start,x_stop,num_x_samples)     
    sample_y = np.linspace(y_start,y_stop,num_y_samples) 
    
    delta = min([x_start,y_start])
    
    samples = []

    for y in sample_y:
        for x in sample_x:
            corner1 = np.float32(np.array([[[x-delta,y-delta]]]))
            corner2 = np.float32(np.array([[[x+delta,y+delta]]]))
            c1_maped = perspectiveTransform(corner1,warp_transform)
            c2_maped = perspectiveTransform(corner2,warp_transform)
            z = (4*delta*delta)/(abs(c2_maped[0,0,0]-c1_maped[0,0,0])*abs(c2_maped[0,0,1]-c1_maped[0,0,1]))

            samples.append(z)

    px, py = np.meshgrid(sample_x, sample_y)

    full_x = np.linspace(0,x_len-1,x_len,endpoint=True,dtype=int)    
    full_y = np.linspace(0,y_len-1,y_len,endpoint=True,dtype=int)
    
    surf_interpolator = bisplrep(px, py, samples)
    weights = bisplev(full_x, full_y, surf_interpolator)


    fymesh, fxmesh = np.meshgrid(full_y, full_x)

    if plotting:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(px,py,samples,'rx',label = "Samples")
        ax.plot_surface(fxmesh, fymesh, weights, label = "interpolated")
        ax.set_title("Sampled and interpolated weights for each pixel")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Integration Weight")
        fig.show()
        settings.blockingPlot = True

    return weights

def TransformFromThetaPhi(bunchData,src,dst):
    """
    generates a perspective transformation from known points and deapplys it to the fitted data to reconstruct an image

    WARN: this projects the points to a flat plane not a sphere and is unsuitable for large angles
    TODO: this is here to invert the original transformation so that we can integrate and photon count, T'T needs to be integral preserving 
            ROI IS NOT IMPLIMENTED

    Parameters
    ----------
    bunchData: 2d array
        the flattened data in theta phi
    src : np.array[,float32]
        source points in pixel coordinates
    dst : np.array[,float32]
        destination points in theta, phi

    Returns
    -------
    out_im : 2d array
        the warped image data in the original camera perspective
    """

    assert len(src) == 4
    assert len(dst) == 4

    ## padding around the output points:

    lpad = 1.01
    tpad = 1.01
    rpad = 1.2
    bpad = 0.92

    dstmin = [tpad * min(dst[:, 0]), lpad * min(dst[:, 1])]
    dst = np.float32([[d[0] - dstmin[0], d[1] - dstmin[1]] for d in dst])

    dstmax = np.array([max(dst[:, 0]) * bpad, max(dst[:, 1]) * rpad], int)
    axis = np.float32([0 - dstmin[0], 0 - dstmin[1]])

    warp_transform, weights = getTransform(bunchData,src,dst) 

    inv_warp_transform = np.linalg.inv(warp_transform)

    srcmax = np.array( [ceil2(max(src[:, 0])), ceil2(max(src[:, 1]))],int)

    if settings.verbose: print("Transforming Image")
    out_im = warpPerspective(bunchData, inv_warp_transform, (dstmax[1], dstmax[0]))
    if settings.verbose: print("Done")

    return (out_im)

def cart2sph(x, y, z):
    """
    generate r,theta,phi from x,y,z

    this is a generic spgerical coordinate refactor

    Parameters
    ----------
    x : float:
        the horizontal offset of the point from the viewing (laser) axis
    y : float:
        the vertical offset of the point from the viewing (laser) axis
    z : float:
        the depth along the viewing (laser) axis to the point

    Returns
    -------
    r : float
        radius of the point
    t : float
        theta of the point
    p : float
        phi of the point
    """
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)  # r
    elev = m.atan2(z, m.sqrt(XsqPlusYsq))  # theta
    az = m.atan2(y, x)  # phi
    return r, elev, az

def src_dst_from_known_points(known_points, units, resolution,lanex_onAx_dist, lanex_theta, lanex_inPlane_dist, lanex_height, lanex_vertical_offset):
    """
    generates src, dst arrays for the perspective transform

    Parameters
    ----------
    known_points : dict [int,int,int]
        A 4 element dict with each corner eg 'TR' being a list of [rulermark,p_x,p_y]
    units: int
        units/radian (typically 1000)
    resolution: int
        pixels/unit (typically 10)
    lanex_onAx_dist: [float] 
        mm distance to the lanex plane in the axis of the laser
    lanex_theta: [float] 
        deg angle of the normal of the lanex plane to the laser  
        # WARN: this assumes that the lanex plane is vertical and only rotates about z
    lanex_inPlane_dist: [float] 
        mm distance of the edge of the lanex (0mm ruler mark) 
        from the axis in the plane of the lanex -ve implies that the laser axes intersects the lanex
    lanex_height: [float] 
        mm height of the lanex from the plane of the laser 
    lanex_vertical_offset: [float]
        mm height of the center of the lanex wrt. the laser level

    Returns
    -------
    src : np.array[,float32]
        list of source points in pixel x,y
    dst : np.array[,float32]
        list of destination points in theta phi [rad/units]

    """
    if settings.verbose:
        print("Generating transformation coords from known points: ")

    order = ['TR','BR','BL','TL']

    src = []
    dst = []

    cosT = np.cos(np.radians(lanex_theta))
    sinT = np.sin(np.radians(lanex_theta))

    for corner in order:
        if settings.verbose:
            print("{} Corner".format(corner))
        src.append([known_points[corner][1],known_points[corner][2]])

        x = - cosT * (lanex_inPlane_dist + known_points[corner][0])
        z = lanex_onAx_dist - sinT * (lanex_inPlane_dist + known_points[corner][0])

        if 'T' in corner:
            y = lanex_vertical_offset + (0.5 * lanex_height)
        else:
            y = lanex_vertical_offset - (0.5 * lanex_height)
        r, p, t = cart2sph(z,x,y)
        if settings.verbose:
            print("x:{}, y:{}, z:{}".format(x,y,z))
            print("r:{}, theta:{}, phi:{}".format(r,t,p))
        dst.append([units * resolution * t, units * resolution * p])
    if settings.verbose:
        print("src: {}\ndst: {}".format(src,dst))

    return(np.array(src), np.array(dst))

def check_integration(pixelData,
                      src,
                      dst,
                      units,
                      resolution,
                      zoom_radius,
                      saveDir=None):
    """
    a function for checking the integral preservation of a generated bunch-like gaussian under the transformation generated from the known points

    Parameters
    ----------
    pixelData : str 
        the absolute path to the calibration image to be tested
    src: np.array([int,int])
        src points in pixel value, start top right, clockwise 
    dst: np.array([t,p])
        destination points in theta phi
    units: int
        units/radian (typically 1000)
    resolution: int
        pixels/unit (typically 10)
    zoom_radius: int
        number of units to plot out from origin (laser/z axis)
    save_dir: str
        absolute path to saving directory or None if not saving
    """

    # Generate test image:
    xary = np.linspace(0, pixelData.shape[0], pixelData.shape[0], endpoint=True)
    yary = np.linspace(0, pixelData.shape[1], pixelData.shape[1], endpoint=True)
    sx = 120
    sy  = 130
    th = 35
    untransformed_data = np.array([[fit.lm_gaus2d(x_v, y_v, amplitude=1, offset=0, xo=0.5*pixelData.shape[0], yo=0.5*pixelData.shape[1], theta=th, sigma_x=sx, sigma_y=sy) for y_v in yary] for x_v in xary])
 
    # transform it:
    transformed, axis = TransformToThetaPhi(untransformed_data,
                                                        np.array(src, np.float32),
                                                        np.array(dst, np.float32))

    # transform and plot:
    fig, ax = check_transformation(untransformed_data,
                         src,dst,units,resolution,zoom_radius,saveDir)

    # compare integrals:
    print("untransformed:\n\t size: {}\n\t sum: {} \n\t ave : {}".format(untransformed_data.shape, np.sum(untransformed_data),np.sum(untransformed_data)/(untransformed_data.shape[0]*untransformed_data.shape[1])))
    print("transformed:\n\t size: {}\n\t sum: {} \n\t ave : {}".format(transformed.shape, np.sum(transformed),np.sum(transformed)/(transformed.shape[0]*transformed.shape[1])))

    print("ratios:\n\tsize: {}\n\t sum: {}".format(transformed.shape[0]*transformed.shape[1]/(untransformed_data.shape[0]*untransformed_data.shape[1]), np.sum(transformed)/np.sum(untransformed_data)))

    fig.suptitle("Integration Ratio : {}".format(np.sum(transformed)/np.sum(untransformed_data)))

    if not settings.saving:
        settings.blockingPlot = True

def check_transformation(pixelData,
                         src,
                         dst,
                         units,
                         resolution,
                         zoom_radius,
                         saveDir=None):
    """
    a function for checking the transformation generated from the known points by plotting the transformed image 

    Parameters
    ----------
    pixelData : str 
        the absolute path to the calibration image to be tested
    src: np.array([int,int])
        src points in pixel value, start top right, clockwise 
    dst: np.array([t,p])
        destination points in theta phi
    units: int
        units/radian (typically 1000)
    resolution: int
        pixels/unit (typically 10)
    zoom_radius: int
        number of units to plot out from origin (laser/z axis)
    save_dir: str
        absolute path to saving directory or None if not saving
    
    Returns
    ----------
    fig : mpl Figure
    ax : mpl Axis Array

    """
    if settings.verbose: print("checking transformation")
    transformed, axis = TransformToThetaPhi(pixelData,
                                            np.array(src, np.float32),
                                            np.array(dst, np.float32))

    fig, ax = plt.subplots(1, 3, figsize=(18, 8))

    x_tick_every = 10
    y_tick_every = 10

    xticks = []
    xlabels = []
    for i in range(transformed.shape[1]):
        if abs(np.floor(i - axis[0])) % (x_tick_every * resolution) == 0:
            xticks.append(i)
            lab = int(np.floor(((i - axis[0]) / resolution)))
            labtext = "{}".format(lab) if not (lab == -10 or lab == 0) else ""
            xlabels.append(labtext)

    yticks = []
    ylabels = []

    for i in range(transformed.shape[0]):
        if abs(np.floor(i - axis[1])) % (y_tick_every * resolution) == 0:
            lab = int(np.floor(((i - axis[1]) / resolution)))
            labtext = "{}".format(-lab) if not lab == 10 else ""
            ylabels.append(labtext)
            yticks.append(i)


    raw_on = 0
    ax[raw_on].imshow(pixelData)
    ax[raw_on].set_title("Raw Data")


    warp_on = 1
    ax[warp_on].imshow(transformed)
    ax[warp_on].set_title("Transformed to Theta-Phi")
    ax[warp_on].spines['left'].set_position(('data', axis[0]))
    ax[warp_on].spines['bottom'].set_position(('data', axis[1] + 1))
    ax[warp_on].set_xticks(xticks, labels=xlabels, rotation=90)
    ax[warp_on].set_yticks(yticks, labels=ylabels, rotation=0)

    for label in ax[warp_on].yaxis.get_ticklabels():
        if label.get_text() == "0":
            dx = 0
            dy = -0.15
            offset = trans.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)

    ax[warp_on].set_xlabel(r'$\theta \quad rad^{{{}}}$'.format(
        int(-np.log10(units))))
    ax[warp_on].xaxis.set_label_coords(
        0.15, 1.2 * (1 - (axis[1] / transformed.shape[0])))

    ax[warp_on].set_ylabel(r'$\phi \quad rad^{{{}}}$'.format(
        int(-np.log10(units))))
    ax[warp_on].yaxis.set_label_coords(0.85 * axis[0] / transformed.shape[1],
                                       0.15)

    z_xticks = []
    z_xlabels = []
    for i in range(transformed.shape[1]):
        if abs(np.floor(i - axis[0])) % (x_tick_every * (resolution / 2)) == 0:
            lab = int(np.floor(((i - axis[0]) / (resolution))))
            labtext = "{}".format(lab) if not (lab == -5 or lab == 0) else ""
            z_xlabels.append(labtext)
            z_xticks.append(i)

    z_yticks = []
    z_ylabels = []

    for i in range(transformed.shape[0]):
        if abs(np.floor(i - axis[1])) % (y_tick_every * (resolution / 2)) == 0:
            lab = int(np.floor(((i - axis[1]) / (resolution))))
            labtext = "{}".format(-lab) if not lab == 5 else ""
            z_ylabels.append(labtext)
            z_yticks.append(i)

    zoom_on = 2

    zoom_x_lims = [
        int(axis[0] - (zoom_radius * resolution)),
        int(axis[0] + (zoom_radius * resolution))
    ]
    zoom_y_lims = [
        int(axis[1] + (zoom_radius * resolution)),
        int(axis[1] - (zoom_radius * resolution))
    ]
    roi = transformed[zoom_x_lims[0]:zoom_x_lims[1],
                      zoom_y_lims[1]:zoom_y_lims[0]]

    ax[zoom_on].imshow(transformed, vmin=roi.min(),
                       vmax=roi.max())  #, extent=[*zoom_x_lims,*zoom_y_lims])
    ax[zoom_on].autoscale(False)
    ax[zoom_on].set_title(r"Zoomed to ${{{}}}rad^{{{}}}$".format(
        zoom_radius, int(-np.log10(units))))
    ax[zoom_on].spines['left'].set_position(('data', axis[0]))
    ax[zoom_on].spines['bottom'].set_position(('data', axis[1] + 1))
    ax[zoom_on].set_xticks(z_xticks, labels=z_xlabels, rotation=90)
    ax[zoom_on].set_yticks(z_yticks, labels=z_ylabels, rotation=0)

    for label in ax[zoom_on].yaxis.get_ticklabels():
        if label.get_text() == "0":
            dx = 0
            dy = -0.15
            offset = trans.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)

    ax[zoom_on].set_ylim(*zoom_y_lims)
    ax[zoom_on].set_xlim(*zoom_x_lims)

    ax[zoom_on].set_xlabel(r'$\theta \quad rad^{{{}}}$'.format(
        int(-np.log10(units))))
    ax[zoom_on].xaxis.set_label_coords(0.15, 0.58)

    ax[zoom_on].set_ylabel(r'$\phi \quad rad^{{{}}}$'.format(
        int(-np.log10(units))))
    ax[zoom_on].yaxis.set_label_coords(0.4, 0.15)

    if saveDir is not None:
        saveplot = "{}\\transformation".format(saveDir)
        fig.savefig(saveplot)
    else:        
        fig.show()
        settings.blockingPlot = True

    return(fig,ax)

if __name__ == "__main__":
    src, dst = src_dst_from_known_points(settings.known_points, 
                                         settings.units,
                                         settings.resolution,
                                         settings.lanex_onAx_dist,
                                         settings.lanex_theta,
                                         settings.lanex_inPlane_dist,
                                         settings.lanex_height,
                                         settings.lanex_vertical_offset)
    
    pixelData = np.array(PIL.Image.open(settings.pointingCalibrationImage))

    check_transformation(pixelData,
                         src,
                         dst,
                         settings.units,
                         settings.resolution,
                         settings.zoom_radius,
                         saveDir=None)
    
    check_integration(pixelData,
                         src,
                         dst,
                         settings.units,
                         settings.resolution,
                         settings.zoom_radius,
                         saveDir=None)

    if settings.blockingPlot:
            input("press RETURN key to continue ...")  # this is here to stop plots from closing immediatly if you are not saving them
