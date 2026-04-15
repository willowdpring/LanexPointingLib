# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:21:22 2022

@author: willo

the __main__ file that should be run 

"""
import numpy as np
import os
import matplotlib.pyplot as plt
from pointing2d_settings import settings
import pointing2d_perspective as perspective
import pointing2d_lib
import pickle
import sys
import json
from pathlib import Path

def main(input_deck_path=None):
    
    settings.update_user_settings(input_deck_path=input_deck_path)
    
    if settings.assert_reasonable():
        export_path = Path(settings.targetDir) / "EXPORTED"  # name the subdirectory to export to

        export_path.mkdir(parents=True, exist_ok=True)

        src, dst = perspective.src_dst_from_known_points(settings.known_points,
                                                    settings.units,
                                                    settings.resolution,
                                                    settings.lanex_onAx_dist, 
                                                    settings.lanex_theta, 
                                                    settings.lanex_inPlane_dist, 
                                                    settings.lanex_height,
                                                    settings.lanex_vertical_offset)

        pointing2d_lib.check_calibration_transformation(str(export_path), src, dst)
        
        backgroundData = pointing2d_lib.get_background()

        stats_pickle = export_path / "stats.pickle"
        stats_npy = export_path / "stats.npy"

        if stats_pickle.exists() and not settings.overwrite:
            print("found existing pikle stats file in export directory")
            
            with open(stats_pickle, 'rb') as handle:
                 stats = pickle.load(handle)
        elif stats_npy.exists() and not settings.overwrite:
            print("found existing numpy stats file in export directory")

            stats = np.load(stats_npy, allow_pickle=True)

        else:
            stats = pointing2d_lib.generate_stats(export_path, src, dst, backgroundData)

        report = pointing2d_lib.generate_report(stats, export_path)
        
        if settings.blockingPlot:
            plt.show()
            input("close? : ")
  

if __name__ == "__main__":
    main()
