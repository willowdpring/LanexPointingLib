# -*- coding: utf-8 -*-
"""
A library for filtering X-Ray spots and subtracting backgrounds from images
designed for use in LWFA experiments

Created on Thu Dec 15 15:21:22 2022

@author: willowdpring@gmail.com

A library of functions for filtering xrays (abundant in lwfa experiments,) smothing noise and subtracting backgrounds 

"""
import os
import PIL
import numpy as np
import matplotlib.pyplot as plt
from numba import cfunc, carray
from numba.types import CPointer, intc, intp, float64, voidptr
from scipy import LowLevelCallable, ndimage
import pointing2d_settings as settings

# There are a series of functions for generating gaussian kernels:


def norm_gaus_ary(s, n=3):
    # generates a normalised gaussian array length with sigma s
    # shift the center to the middle of the range and back for a modulo
    l = int(n * s)
    g = np.zeros(2 * l + 1)
    for i in range(2 * l + 1):
        x = i
        a = 1 / (s * np.sqrt(2 * np.pi))
        g[i] = a * np.exp(-((x - ((n - 1) * s))**2) / (s**2))
    return (g)


def norm_gaus2d_ary(s1, n1=3, s2=1, n2=3):
    g1 = norm_gaus_ary(s1, n1)
    g2 = norm_gaus_ary(s2, n2)
    g2d = np.ndarray((len(g1), len(g2)))
    for i, v1 in enumerate(g1):
        for j, v2 in enumerate(g2):
            g2d[i, j] = (v1 * v2)
    g2d = g2d / g2d.sum()
    return (g2d)


# a series of functions for x-ray subtraction
# X-Rays typically present as single saturated pixels


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def nb_fil3(values_ptr, len_values, result, data):
    """
    A sliding window to set bright pixels to the average of the pixels around them in a 3x3 excluding the center

    should only be called by filterImage()

    |0|1|2|
    |3|X|5|
    |6|7|8|
    """
    skip = 4
    values = carray(values_ptr, (len_values, ), dtype=float64)
    tot = 0.0
    lim = 25.0
    for i, v in enumerate(values):
        if i != skip:
            tot = tot + v
    av = tot / (len_values - 1)
    if values[skip] - av > lim:
        result[0] = av
    else:
        result[0] = values[skip]
    return (1)


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def nb_fil5(values_ptr, len_values, result, data):
    """
    A sliding window to set bright pixels to the average of the pixels around them in a 5x5 excluding the center

    should only be called by filterImage()

    | 0| 1| 2| 3| 4|
    | 5| 6| 7| 8| 9|
    |10|11| X|13|14|
    |15|16|17|18|19|
    |20|21|22|23|24|
    """

    skip = 12
    values = carray(values_ptr, (len_values, ), dtype=float64)
    tot = 0.0
    lim = 25.0
    for i, v in enumerate(values):
        if i != skip:
            tot = tot + v
    av = tot / (len_values - 1)
    if values[skip] - av > lim:
        result[0] = av
    else:
        result[0] = values[skip]
    return (1)


@cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
def nb_fil53(values_ptr, len_values, result, data):
    """
    A sliding window to set bright pixels to the average of the pixels around them in a 5x5 excluding the center 3x3

    should only be called by filterImage()

    | 0| 1| 2| 3| 4|
    | 5| X| X| X| 9|
    |10| X| X| X|14|
    |15| X| X| X|19|
    |20|21|22|23|24|
    """

    cen = 12
    skip = [6, 7, 8, 11, 12, 13, 16, 17, 18]
    values = carray(values_ptr, (len_values, ), dtype=float64)
    tot = 0.0
    lim = 15.0
    for i, v in enumerate(values):
        if i not in skip:
            tot = tot + v
    av = tot / (len_values - len(skip))
    if values[cen] - av > lim:
        result[0] = av
    else:
        result[0] = values[cen]
    return (1)


def filterImage(image_data, f):
    """
    filters image_data for single bright pixels indicitive of xrays incident on the chip
    by applying a sliding window to set bright pixels to the average of the pixels around them in a 5x5 or 3x3

    Parameters
    ----------
    image_data : 2d array
        the image data to scan
    f : int accepts 3,5,53
        a selector of the sliding window size
        3 -> 3x3 
        5 -> 5x5
        53 -> 5x5 excluding 3x3 

    Returns
    -------
    filtered
        a 2d array of filtered image data

    """
    if f == 3:
        filtered = ndimage.generic_filter(
            image_data,
            LowLevelCallable(
                nb_fil3.ctypes,
                signature="int (double *, npy_intp, double *, void *)"),
            size=3)
    elif f == 5:
        filtered = ndimage.generic_filter(
            image_data,
            LowLevelCallable(
                nb_fil5.ctypes,
                signature="int (double *, npy_intp, double *, void *)"),
            size=5)
    elif f == 53:
        filtered = ndimage.generic_filter(
            image_data,
            LowLevelCallable(
                nb_fil53.ctypes,
                signature="int (double *, npy_intp, double *, void *)"),
            size=5)
    else:
        filtered = image_data
    return (filtered)


def walkDir(targetDir,
            select_strings=['.tiff', '.tif'],
            ignore_strings=['BACKGROUND']):
    """
    walkes the directory tree of targetdir and lists all the files containing select_strings but not ignore_strings

    Parameters
    ----------
    targetDir : str
        the path to the root directory for the search

    select_strings : np.array(str) optional 
        necessary substrings for a file to be included 
        default : ['.tiff','.tif']


    ignore_strings : np.array(str) optional 
        unwanted substrings that imply the file should be ignored eg the name of a subdirectory
        default : ['BACKGROUND']

    Returns
    -------
    files : list(str)
        a list of absolute paths to files that match the parameters

    """
    assert os.path.isdir(targetDir), "Target Directory is not valid \n {}".format(targetDir)
    out_files = []
    for root, dirs, files in os.walk(targetDir):
        if ignore_strings == None:
            for file in files:    
                if any([f in file for f in select_strings]):
                    out_files.append(os.path.join(root, file))
        else:
            for file in files:    
                if any([f in file for f in select_strings]) and not any([f in os.path.join(root, file) for f in ignore_strings]):
                    out_files.append(os.path.join(root, file))
        
    return (out_files)


def mashBackground(tifFiles):
    """
    Backgrounds files are generated from all tiff files in the list 
    by combining along z using a max() and a mean() method  

    Parameters
    ----------
    tiffFiles : list(str)
        a list of tiff files to be flattened

    Returns
    -------
    avg : 2d numpy array
        the average value of each pixel
    highest : 2d numpy array
        the maximum value of each pixel

    """
    assert len(tifFiles) > 1
    tot = None
    highest = None
    for file in tifFiles:
        raw = np.array(PIL.Image.open(file), np.float32)
        proc = raw
        if tot is not None:
            tot += proc
            highest = np.maximum(highest, proc)
        else:
            tot = proc
            highest = proc

    avg = np.divide(tot, len(tifFiles))
    return ([avg, highest])


def generateAndPlotBackgrounds(
        targetDir="D:/CLPU2_DATA/XRAYCCD/20220519/BACKGROUND/",
        x_fils=[3],
        save=True,
        plot=True):
    """
    Generates and saves background .tiff files from all tiff files in target directory
    in an ./EXPORTED/ subdirectory

    Parameters
    ----------
    targetDir : str
        the absolute path to the folder containing the tiff files to be combined

    x_fils : list(int) 3|5|53
        xray filters to apply filter

    save : bool
        if True save files

    plot : bool
        if True plot images and outputs 

    """
    av_low = 0
    av_hi = 9000
    max_s = 1.985

    tifFiles = walkDir(targetDir,ignore_strings=None)

    N = len(tifFiles)
    assert N > 0

    for file in tifFiles:
        raw = np.array(PIL.Image.open(file))
        proc = raw
        for f in x_fils:  # filter xrays
            proc = filterImage(proc, f)
        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 12))
            ax.imshow(proc, vmin=av_low, vmax=av_hi)
        else:
            ax.imshow(proc)
        ax.set_title(file)

    backs = mashBackground(tifFiles)

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(12, 12))

        ax[0].imshow(backs[0], vmin=av_low, vmax=av_hi)
        ax[0].set_title("avg")
        ax[1].imshow(backs[1], vmin=max_s * av_low, vmax=max_s * av_hi)
        ax[1].set_title("max")

    if save:
        # Export
        exportDir = "{}\\EXPORTED".format(
            targetDir)  # name the subdirectory to export to
        if not os.path.exists(exportDir):  # check if it exists
            os.mkdir(exportDir)  # create it if not

        exportPathAv = "{}\\AVG_BAK.tiff".format(exportDir)
        exportPathMax = "{}\\MAX_BAK.tiff".format(exportDir)
        PIL.Image.fromarray(backs[0]).save(exportPathAv, format='tiff')
        PIL.Image.fromarray(backs[1]).save(exportPathMax, format='tiff')


if __name__ == "__main__":
    background_dir = settings.background_dir
    generateAndPlotBackgrounds(background_dir)