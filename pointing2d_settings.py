# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:21:22 2022

@author: willo

Configuration of the runtime settings for the program e.g. working directories, backgrounds, etc.


These are the default settings. 
Please pass user settings as a json to main


"""
import os

""" The first settings are for the target directory, the script will try to analyse every *.tiff* and *.tif* file in this directory and subdirectorys;
"""
verbose = True #True # [bool] this will toggle printouts in many functions, set to True to enabe logging to terminal and assist debugging 

targetDir = "../example/" #[str] the target root directory saves will go in ./EXPORTED

start = 0  # [int] the first file to analyse

stop = 1 # [int] the last file to analyse -1 for all

decimate = 1  # [int] the step size

saving = False  # [bool] if True we will save the numpy arrays and png's of the resulting contor plots?

overwrite = False  # [bool] if True we will overwrite existing save data (in the ./EXPORTED/ directory on a file by file basis)

""" Backgrounds are generated from user selected files in a seperate directory 
    by compressing all the tiff files in that directory along z using a max() and a mean() method  
"""
background_dir = "{}\\BACKGROUND".format(targetDir)  # [str] the directory to generate

generate_background_files = False # True  # [bool] if True we will generate /EXPORTED/MAX_BAK.tiff and /EXPORTED/AVG_BAK.tiff in the background folder

background = None # "{}\\EXPORTED\\AVG_BAK.tiff".format(background_dir)  # [str] the file to use as a background

background_clip = 1  # [int] the percentile below which the background data is ignored

background_scale = 1 #[float] a multiplicative factor applied to the background data

plotBackgroundSubtraction = True  # [bool] if True we will generate a plot before and after subtracting the backgtround to check

kernel = "backfilt.norm_gaus2d_ary(12, 3, 12, 3)"  # [string] when evaluated this should make the The convolution kernel applied before background subtraction

filters = [
    3, 3, 3, 3, 53, 53, 53, 53
]  # [list[3|5|53]] the sequence of x-ray filters to apply (see backfiltlib.py)

ignore_regions = []  # [array([[[x1,y1],[x2,y2]]])] list of paird x,y, coordinates for rectangles to be ignored # for user defined regions
"""
[
    [[213, 543], [223, 553]], [[20, 495], [120, 555]], [[105, 585], [115, 599]]
]
"""

ignore_ptvs_below = 12  # the peak to mean value ratio above which the image is accepted as having electrons
""" The transformation is generated from four known points, 
    pixel values in the first and [x,y,z] in the second  with the laser along z and the target at origin
    [theta, phi] coordinates can be used if the toggle is set to true 

"""
units = 1000  # units/radian

resolution = 10  # pixels/unit

zoom_radius = 8  # the radius of the analysis box

#pointingCalibrationImage = "C:\\Users\\BunkerC-User\\Documents\\LanexPointingTEST\\230220\\Lanex_in.tiff"
pointingCalibrationImage = "C:\\Users\\willo\\Documents\\Baris_Lanex\\light-reference.tiff "# "C:/Users/willo/Documents/BunkerC/LanexBeamProfile/HighE_LanexIN.tiff"

dh = 0  # a nudge to vertical offset of the lanex in mm
dx = 1.5  # a nudge to horizontal offset of the lanex in mm


"""
           Lanex
               \ 
laser:   theta  \ 
--->o__________/_\_______
    |            |\ \ < inPlane
    |<-- onAx -->| 

"""

# measurements are taken in horizontal plane of the laser and the heights of the top and bottom corners of the lanex are calculated 

lanex_onAx_dist = 1000 # 1150 # [float] mm distance to the lanex plane in the axis of the laser

lanex_theta = 0 # 45 # [float] deg angle of the normal of the lanex plane to the laser  
                 # WARN: this assumes that the lanex plane is vertical and only rotates about z

lanex_inPlane_dist = 0 # -50 # [float] mm distance of the edge of the lanex (0mm ruler mark) 
                         # from the axis in the plane of the lanex -ve implies that the laser axes intersects the lanex
                                         
lanex_height = 50 # 180 # [float] mm height of the lanex screen

lanex_vertical_offset = 0 # [float] mm height of the center plane of the lanex from the plane of the laser 

# Known points is a dict of four lanex corners as keys 
# and 3 element arrays as entries: [mm Mark on ruler, pixel X coord, pixel Y coord]
"""
# these are for Feb \\230220\\Lanex_in.tiff
known_points = {'TR': [0,1180,130],  # TR - Top Right 
                'BL': [280,213,942], # BL - Bottom Left
                'BR': [0,1159,877],  # BR - Bottom right
                'TL': [290,95,79]    # TL - Top Left
                }

# these are for December \\
known_points = {'TR': [0,891, 91],  # TR - Top Right 
                'BL': [160,346, 950], # BL - Bottom Left
                'BR': [0,873, 882],  # BR - Bottom right
                'TL': [230,25,23]    # TL - Top Left
                }

"""

known_points = [
                [[129.500,187.500], [79.000,176.000], [61.500,343.000], [129.500,281.000]],
                [[20.705,17.029],[13.406,14.304], [9.381,22.318], [20.015,22.058]]
                ]

"""

(Px,PY) -> (Lx,Ly)

[
    #[ [P_X, P_Y] , [X,Y,Z], 'note']
    [[891, 91], [14.27 + dx, -91 + dh, 1303.46], "top right - 0 on the ruler"],
    [[873, 882], [14.27 + dx, 91 + dh, 1303.46], "bottom right - 0 on the ruler"],
    [[25, 23], [-181.83 + dx, -91 + dh, 1076.43], "top left - 23cm along"],
    [[346, 950], [-90.38 + dx, 91 + dh, 1182.38], "bottom left - 16cm along"]
]
"""

transformation = None # this is a placeholder for a variable that will contain the transformation and normalisation matricies during runttime

in_theta_phi = False  # [bool] if True the known points are given in spherical coords (without radius)

checkTransformation = True  # [bool] if True a plot will be generated to check the generated transformation

blockingPlot = False   # [bool] this is here to stop plots from closing immediatly if you are not saving them

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
            assert os.path.exists(os.path.abspath(pointingCalibrationImage)), "No Calibration Image to Check at {}\n".format(os.path.abspath(pointingCalibrationImage))
        except AssertionError as e:
            msg += "\t{}\n".format(e)
            ers = True

    if ers == 0:
        return(True)
    else:
        msg += "please edit the input JSON and try again"
        print(msg)
        return(False)


if __name__ == "__main__":
    assert_reasonable()
