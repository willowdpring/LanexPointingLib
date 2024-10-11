# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:21:22 2022

@author: willo

the __main__ file that should be run 

"""
import numpy as np
import os
import matplotlib.pyplot as plt
import pointing2d_settings as settings
import pointing2d_perspective as perspective
import pointing2d_lib
import pickle
import sys
import json

def main(input_deck_path=None):

    pointing2d_lib.update_user_settings(input_deck_path=input_deck_path)
    
    if settings.assert_reasonable():
        exportDir = "{}\\EXPORTED".format(settings.targetDir)  # name the subdirectory to export to

        if settings.saving and not os.path.exists(exportDir):  # check if it exists
            os.mkdir(exportDir)  # create it if not

        src, dst = perspective.src_dst_from_known_points(settings.known_points,
                                                    settings.units,
                                                    settings.resolution,
                                                    settings.lanex_onAx_dist, 
                                                    settings.lanex_theta, 
                                                    settings.lanex_inPlane_dist, 
                                                    settings.lanex_height,
                                                    settings.lanex_vertical_offset)

        pointing2d_lib.check_calibration_transformation(exportDir, src, dst)

        backgroundData = pointing2d_lib.get_background()

        if os.path.exists("{}\\stats.pickle".format(exportDir)) and not settings.overwrite:
            print("found existing pikle stats file in export directory")
            
            with open("{}\\stats.pickle".format(exportDir), 'rb') as handle:
                stats = pickle.load(handle)
        elif os.path.exists("{}\\stats.npy".format(exportDir)) and not settings.overwrite:
            print("found existing numpy stats file in export directory")

            stats = np.load("{}\\stats.npy".format(exportDir), allow_pickle=True)

        else:
            stats = pointing2d_lib.generate_stats(exportDir, src, dst, backgroundData)

        report = pointing2d_lib.generate_report(stats, exportDir)
        
        if settings.blockingPlot:
            plt.show()
            input("close? : ")
  

if __name__ == "__main__":
    main()
