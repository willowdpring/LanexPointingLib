# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:21:22 2022

@author: willo
"""
import numpy as np
import os
import PIL
import pointing2d_settings as settings
import pointing2d_perspective as perspective
import pointing2d_fit as fit
import pointing2d_lib


def main():

    exportDir = "{}\\EXPORTED".format(settings.targetDir)  # name the subdirectory to export to

    if settings.saving and not os.path.exists(exportDir):  # check if it exists
        os.mkdir(exportDir)  # create it if not

    src, dst = perspective.src_dst_from_PIX_XYZ(settings.known_points,
                                                settings.units,
                                                settings.resolution)

    if settings.checkTransformation:
        pixelData = np.array(
            np.array(PIL.Image.open(settings.pointingCalibrationImage)))
        saveDir = None
        if settings.saving:
            saveDir = exportDir
        perspective.check_transformation(pixelData, src, dst, settings.units,
                                         settings.resolution,
                                         settings.zoom_radius, saveDir)

    if settings.checkTransformation:
        pointing2d_lib.check_calibration_transformation(exportDir, src, dst)

    backgroundData = pointing2d_lib.get_background()

    if os.path.exists(
            "{}\\stats.npy".format(exportDir)) and not settings.overwrite:
        print("found existing stats file in export directory")
        stats = np.load("{}\\stats.npy".format(exportDir),
                        allow_pickle=True,
                        fix_imports=True)
    else:
        stats = pointing2d_lib.generate_stats(exportDir, src, dst, backgroundData)   

    report = pointing2d_lib.generate_report(stats, exportDir)

    input("close? : ")

def test_trans():
    src, dst = perspective.src_dst_from_PIX_XYZ(settings.known_points,
                                                settings.units,
                                                settings.resolution)
    pixelData = np.array(
            np.array(PIL.Image.open(settings.pointingCalibrationImage)))
        
    """
    if settings.checkTransformation:
        saveDir = None
        if settings.saving:
            saveDir = exportDir
        perspective.check_transformation(pixelData, src, dst, settings.units,
                                         settings.resolution,
                                         settings.zoom_radius, saveDir)
    """

    untransformed_data = np.ones_like(pixelData)
    
    """
    x_pixels = np.linspace(0, pixelData.shape[0] - 1, pixelData.shape[0], endpoint=True)
    y_pixels = np.linspace(0, pixelData.shape[1] - 1, pixelData.shape[1], endpoint=True)
    x_grid, y_grid = np.meshgrid(x_pixels, y_pixels)
    untransformed_data = fit.lm_gaus2d(x_grid, y_grid, amplitude=200, offset=20, xo=2 * pixelData.shape[0] / 3, yo=pixelData.shape[0] / 2, theta=0, sigma_x=20, sigma_y=30)
    """
    transformed, axis = perspective.TransformToThetaPhi(untransformed_data,
                                                        np.array(src, np.float32),
                                                        np.array(dst, np.float32))
    
    print("untransformed:\n\t size: {}\n\t sum: {} \n\t ave : {}".format(untransformed_data.shape, np.sum(untransformed_data),np.sum(untransformed_data)/(untransformed_data.shape[0]*untransformed_data.shape[1])))
    print("transformed:\n\t size: {}\n\t sum: {} \n\t ave : {}".format(transformed.shape, np.sum(transformed),np.sum(transformed)/(transformed.shape[0]*transformed.shape[1])))

    if not settings.saving:
        input("press RETURN key to continue ...") ## this is here to stop plots from closing immediatly if you are not saving them



if __name__ == "__main__":
    main()
    #test_trans()