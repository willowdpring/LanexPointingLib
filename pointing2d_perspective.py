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
from cv2 import getPerspectiveTransform, warpPerspective
import pointing2d_settings as settings


def normalise_transformation(in_shape,warp_transform,dstmax):
    """
    TODO: this still needs alot of work

        it may be better to invert the transformation and generate an inverted fit then integrate that?
        inverted_trans = np.linalg.pinv(trans)

    generates various test arrays and runs them throught the transformation to generate an array of normalisation weights for integral preservation  
    
    Parameters
    ----------
    in_shape
        the size/shape of the arrays that will be transformed i.e. (raw camera data as an np.array).shape
    warp_transform
        the cv2 transformation generated with getPerspectiveTransform
    dstmax
        the padded (ultimate) destination points in theta, phi

    Returns
    -------
    transformation_scale
        the array of pixel weights to integral preserve the transformation
        

"""

    transformation_scale = np.zeros(in_shape)
    for p_x, col in enumerate(transformation_scale):
        for p_y, pixel in enumerate(col):
            dummy = np.zeros_like(transformation_scale)
            dummy[p_x,p_y] = 1e8
            result = np.sum(warpPerspective(dummy,warp_transform,(dstmax[1],dstmax[0])))
            if result == 0:
                transformation_scale[p_x,p_y] = 0
            else:
                print("{} for x:{}, y:{}".format(result/1e8,p_x,p_y))
                transformation_scale[p_x,p_y] = result/1e8
    return(transformation_scale)


def TransformToThetaPhi(pixelData, src, dst):
    """
    generates a perspective transformation from known points and applys it to the image

    WARN: this projects the points to a flat plane not a sphere and is unsuitable for large angles
    TODO: this is also not integral preserving ):


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

    if settings.transformation is not None:
        if settings.verbose: print("Loading existing transformation and normalisation matricies ... ", end = '')
        warp_transform = settings.transformation[0]
        transformation_scale = settings.transformation[1]
        if settings.verbose: print("Done")
    else: 
        if settings.verbose: print("calculating transformation ... ", end = '')
        warp_transform = getPerspectiveTransform(src, dst)
        if settings.verbose: print("Done")

        if settings.verbose: print("calculating normalisations ... ", end = '')
        transfromation_scale = normalise_transformation(pixelData.shape,warp_transform,dstmax)
        if settings.verbose: print("Done")
        if settings.verbose: print("Recording to temp ... ", end = '')
        settings.transformation = [warp_transform, transformation_scale]
        if settings.verbose: print("Done")

    if settings.verbose: print("Transforming Image")
    out_im = warpPerspective(pixelData, warp_transform, (dstmax[1], dstmax[0]))
    if settings.verbose: print("Done")

    return (out_im, axis)


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
    pixelData = np.array(
        np.array(PIL.Image.open(settings.pointingCalibrationImage)))
    check_transformation(pixelData,
                         src,
                         dst,
                         settings.units,
                         settings.resolution,
                         settings.zoom_radius,
                         saveDir=None)
