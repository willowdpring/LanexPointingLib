# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:21:22 2022

@author: willo

Configuration of the runtime settings for the program e.g. working directories, backgrounds, etc.

"""
import os
from pointing2d_backfiltlib import norm_gaus2d_ary

""" The first settings are for the target directory, the script will try to analyse every *.tiff* and *.tif* file in this directory and subdirectorys;
"""
verbose = True

targetDir = "C:\\Users\\BunkerC-User\\Documents\\LanexPointingTEST\\230223\\Run001"  #[str] the target root directory saves will go in ./EXPORTED

start = 1  # [int] the first file to analyse

stop = -1  # [int] the last file to analyse

decimate = 1  # [int] the step size

saving = False  # [bool] if True we will save the numpy arrays and png's of the resulting contor plots?

overwrite = False  # [bool] if True we will overwrite existing save data (in the ./EXPORTED/ directory on a file by file basis)
""" Backgrounds are generated from user selected files in a seperate directory 
    by compressing all the tiff files in that directory along z using a max() and a mean() method  
"""
background_dir = "{}\\BACKGROUND".format(targetDir)  # [str] the directory to generate

generate_background_files = True  # [bool] if True we will generate /EXPORTED/MAX_BAK.tiff and /EXPORTED/AVG_BAK.tiff in the background folder

background = "{}\\EXPORTED\\MAX_BAK.tiff".format(background_dir)  # [str] the file to use as a background

background_clip = 60  # [int] the percentile below which the background data is ignored

background_scale = 1.1  #[float] a multiplicative factor applied to the background data

plotBackgroundSubtraction = True  # [bool] if True we will generate a plot before and after subtracting the backgtround to check

kernel = norm_gaus2d_ary(
    60, 3, 60, 3
)  # [2d array] The convolution kernel applied before background subtraction

filters = [
    3, 3, 3, 3, 53, 53, 53, 53
]  # [list[3|5|53]] the sequence of x-ray filters to apply (see backfiltlib.py)

ignore_regions = [
    [[213, 543], [223, 553]], [[20, 495], [120, 555]], [[105, 585], [115, 599]]
]  # [array([[[x1,y1],[x2,y2]]])] list of paird x,y, coordinates for rectangles to be ignored # for user defined regions

ignore_ptvs_below = 10  # the peak to mean value ratio above which the image is accepted as having electrons
""" The transformation is generated from four known points, 
    pixel values in the first and [x,y,z] in the second  with the laser along z and the target at origin
    [theta, phi] coordinates can be used if the toggle is set to true 

"""
units = 1000  # /radian

resolution = 10  # pixels/unit

zoom_radius = 30  # the radius of the analysis box

pointingCalibrationImage = "C:\\Users\\BunkerC-User\\Documents\\LanexPointingTEST\\230220\\Lanex_out.tiff"

dh = 10  # a nudge to vertical offset of the lanex in mm
dx = 12  # a nudge to horizontal offset of the lanex in mm

known_points = [
    #[ [X,Y] , [X,Y,Z], 'note']
    [[891, 91], [14.27 + dx, -91 + dh, 1303.46], "top right"],
    [[873, 882], [14.27 + dx, 91 + dh, 1303.46], "bottom right"],
    [[25, 23], [-181.83 + dx, -91 + dh, 1076.43], "top left (23cm along)"],
    [[346, 950], [-90.38 + dx, 91 + dh, 1182.38], "bottom left 16cm along"]
]

transformation = None # this is a placeholder for a variable that will contain the transformation and normalisation matricies

in_theta_phi = False  # [bool] if True the known points are given in spherical coords (without radius)

checkTransformation = True  # [bool] if True a plot will be generated to check the generated transformation


def assert_reasonable():
    ers = False
    msg = "Settings are NOT reasonable: \n  detected errors:\n"
    try:
        assert os.path.exists(targetDir), 'No Target Directory'
    except AssertionError as e:
        msg += "\t{}\n".format(e)
        ers = True
    if background is not None:
        try:
            assert os.path.exists(background), 'Background file doesn\'t exist'
        except AssertionError as e:
            msg += "\t{}\n".format(e)
            ers = True
        try:
            assert background_clip < 100 and  background_clip >= 0, 'background_clip should be in range 0-100'
        except AssertionError as e:
            msg += "\t{}\n".format(e)
            ers = True

    if generate_background_files:
        try:
            assert os.path.exists(background_dir), 'No Target Background Directory for Generating new Averages'
        except AssertionError as e:
            msg += "\t{}\n".format(e)
            ers = True

    if checkTransformation:
        try:
            assert os.path.exists(pointingCalibrationImage), 'No Calibration Image to Check'
        except AssertionError as e:
            msg += "\t{}\n".format(e)
            ers = True

    if ers == 0:
        return(True)
    else:
        msg += "please edit pointing2d_settings.py and try again"
        print(msg)
        return(False)


if __name__ == "__main__":
    assert_reasonable()
