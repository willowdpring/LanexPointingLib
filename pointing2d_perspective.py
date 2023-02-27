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


def getTransform(src,dst):
    """
    generates or loads a perspective transformation from known points

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
        warp_transform = settings.transformation
        if settings.verbose: print("Done")
    else: 
        if settings.verbose: print("calculating transformation ... ", end = '')
        warp_transform = getPerspectiveTransform(np.array(src,np.float32), np.array(dst,np.float32))
        if settings.verbose: print("Done")

        if settings.verbose: print("Recording to temp ... ", end = '')
        settings.transformation = warp_transform
        if settings.verbose: print("Done")
    return warp_transform 


def TransformToThetaPhi(pixelData, src, dst):
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

    warp_transform = getTransform(src,dst)

    if settings.verbose: print("Transforming Image")
    out_im = warpPerspective(pixelData, warp_transform, (dstmax[1], dstmax[0]))
    if settings.verbose: print("Done")

    return (out_im, axis)


def GenerateWeightMatrix(pixelData, src, dst):
    """
    Generates a matrix that contains a weight value for each pixel of the input image to preserve integrals after the perspective warp

    TODO: NOT IMPLIMENTED

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
    num_x_samples = 10
    num_y_samples = 8

    x_len = pixelData.shape[0]
    y_len = pixelData.shape[1]

    x_start = x_len/(2*num_x_samples)
    x_stop = x_len - x_start

    y_start = y_len/(2*num_y_samples)
    y_stop = y_len - y_start

    sample_x = np.linspace(x_start,x_stop,num_x_samples)     
    sample_y = np.linspace(y_start,y_stop,num_y_samples) 
    
    delta = min([x_start,y_start])

    warp_transfrom = getTransform(src, dst)
    
    samples = []

    for y in sample_y:
        for x in sample_x:
            corner1 = np.float32(np.array([[[x-delta,y-delta]]]))
            corner2 = np.float32(np.array([[[x+delta,y+delta]]]))
            c1_maped = perspectiveTransform(corner1,warp_transfrom)
            c2_maped = perspectiveTransform(corner2,warp_transfrom)
            z = (4*delta*delta)/(abs(c2_maped[0,0,0]-c1_maped[0,0,0])*abs(c2_maped[0,0,1]-c1_maped[0,0,1]))

            samples.append(z)

    px,py = np.meshgrid(sample_x,sample_y)

    full_x = np.linspace(0,pixelData.shape[0]-1,pixelData.shape[0],endpoint=True,dtype=int)
    full_y = np.linspace(0,pixelData.shape[1]-1,pixelData.shape[1],endpoint=True,dtype=int)    
    
    surf_interpolator = bisplrep(px, py, samples)
    weights = bisplev(full_x, full_y, surf_interpolator)

    fymesh,fxmesh = np.meshgrid(full_y,full_x)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(px,py,samples)
    ax.plot_surface(fxmesh,fymesh,weights)
    fig.show()
    input("done?")

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

    if settings.transformation is not None:
        if settings.verbose: print("Loading existing transformation ... ", end = '')
        warp_transform = settings.transformation
        if settings.verbose: print("Done")
    else: 
        if settings.verbose: print("calculating transformation ... ", end = '')
        warp_transform = getPerspectiveTransform(src, dst)
        if settings.verbose: print("Done")

        if settings.verbose: print("Recording to temp ... ", end = '')
        settings.transformation = warp_transform
        if settings.verbose: print("Done")

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

def src_dst_from_PIX_XYZ(known_points, units, resolution):
    """
    generates src, dst arrays for the perspective transform

    Parameters
    ----------
    known_points : np.array([[int,int],[float,float,float],str]) 
        A 4 element array with each element being a list of [p_x,p_y], [x,y,z], and comment string 
    units: int
        units/radian (typically 1000)
    resolution: int
        pixels/unit (typically 10)

    Returns
    -------
    src : np.array[,float32]
        list of source points in pixel x,y
    dst : np.array[,float32]
        list of destination points in theta phi [rad/units]

    """
    assert len(known_points) == 4

    src = []
    dst = []

    for point in known_points:
        try:
            src.append(point[0])
            r, t, p = cart2sph(*point[1][::-1])
            dst.append([units * resolution * t, units * resolution * p])
        except (IndexError):
            print("ERROR ON {}".format(point[2]))

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
    x_array = np.ones(pixelData.shape[0])
    y_array = np.ones(pixelData.shape[1])


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
    """
    if settings.verbose: print("checking transformation")
    transformed, axis = TransformToThetaPhi(pixelData,
                                            np.array(src, np.float32),
                                            np.array(dst, np.float32))

    fig, ax = plt.subplots(3, 1, figsize=(18, 8))

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

    warp_on = 0
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

    zoom_on = 1

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


if __name__ == "__main__":
    src, dst = src_dst_from_PIX_XYZ(settings.known_points, settings.units,
                                    settings.resolution)
    
    pixelData = np.array(PIL.Image.open(settings.pointingCalibrationImage))


    GenerateWeightMatrix(pixelData, src, dst)



"""
    
    check_transformation(pixelData,
                         src,
                         dst,
                         settings.units,
                         settings.resolution,
                         settings.zoom_radius,
                         saveDir=None)
"""