# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:21:22 2022

@author: willo

Configuration of the runtime settings for the program e.g. working directories, backgrounds, etc.

These are the default settings.
Please pass user settings as a json to main
"""

import json
import sys
import pathlib

# font = {'family': 'Corbel', 'weight': 'normal', 'size': 14}
# rc('font', **font)


class Settings:
    def __init__(self):

        # --- Target Directory ---
        self.verbose = True  # [bool] toggle printouts/logging to terminal
        self.targetDir = pathlib.Path(
            "../example/"
        )  # [path] root dir; saves go in ./EXPORTED
        self.shortlist = None  # [list|None] specific files to use from targetDir
        self.start = 0  # [int] first file to analyse
        self.stop = 1  # [int] last file to analyse, -1 for all
        self.decimate = 1  # [int] step size
        self.saving = False  # [bool] save numpy arrays and png contour plots
        self.overwrite = False  # [bool] overwrite existing save data in ./EXPORTED/

        # --- Backgrounds ---
        # Generated from user-selected files in a separate directory by compressing
        # all tiffs along z using max() and mean() methods
        self.background_dir = pathlib.Path(
            "BACKGROUND"
        )  # [path] subdir for background files
        self.generate_background_files = False  # [bool] generate MAX_BAK/AVG_BAK tiffs
        self.background = pathlib.Path(
            "/EXPORTED/AVG_BAK.tiff"
        )  # [path] background file
        self.background_clip = (
            1  # [int] percentile below which background data is ignored
        )
        self.background_scale = (
            1  # [float] multiplicative factor applied to background data
        )
        self.plotBackgroundSubtraction = (
            True  # [bool] plot before/after background subtraction
        )

        # --- Filtering ---
        self.kernel = "backfilt.norm_gaus2d_ary(12, 3, 12, 3)"  # [str] convolution kernel expression
        self.filters = [3, 3, 3, 3, 53, 53, 53, 53]  # [list] x-ray filter sequence
        self.ignore_regions = []  # [list] paired x,y rect coords to ignore, e.g:
        # [[[213,543],[223,553]], [[20,495],[120,555]]]
        self.ignore_ptvs_below = (
            12  # [float] peak-to-mean ratio threshold for electron acceptance
        )

        # --- Transformation / Geometry ---
        # Generated from four known points: pixel coords and [x,y,z] world coords
        # with laser along z and target at origin. [theta,phi] coords available if toggled.
        self.units = 1000  # [float] units/radian
        self.resolution = 10  # [float] pixels/unit
        self.zoom_radius = 8  # [float] radius of the analysis box
        self.pointingCalibrationImage = pathlib.Path("./EXAMPLES/AirLanex_330.tiff")
        self.dh = 0  # [float] nudge to vertical lanex offset (mm)
        self.dx = 0  # [float] nudge to horizontal lanex offset (mm)

        # --- Lanex Geometry ---
        #            Lanex
        #                \
        # laser:  theta   \
        # --->o__________/_\_______
        #     |            |\ \ < inPlane
        #     |<-- onAx -->|
        #
        # Measurements taken in horizontal plane of laser;
        # top/bottom corner heights are calculated from these.
        self.lanex_onAx_dist = (
            1000  # [float] mm distance to lanex plane along laser axis
        )
        self.lanex_theta = 0  # [float] deg angle of lanex normal to laser
        # WARN: assumes lanex is vertical, rotates about z only
        self.lanex_inPlane_dist = (
            0  # [float] mm distance of lanex edge (0mm mark) from axis
        )
        # in lanex plane; -ve means laser axis intersects lanex
        self.lanex_height = 50  # [float] mm height of lanex screen
        self.lanex_vertical_offset = (
            0  # [float] mm height of lanex centre plane from laser plane
        )

        # Known points: four lanex corners in pixel space then world space
        self.known_points = [
            [
                [129.500, 187.500],
                [79.000, 176.000],
                [61.500, 343.000],
                [129.500, 281.000],
            ],
            [[20.705, 17.029], [13.406, 14.304], [9.381, 22.318], [20.015, 22.058]],
        ]
        
        #   padding of the image around the known points wrt the axis (lpad, tpad, rpad, bpad) = settings.src_padding * zoom
        self.dst_padding = [1.1,1.1]
        self.dst_layout = (
            None  # placeholder for output image dims
        )       
        self.transformation = (
            None  # placeholder for transformation matrices
        )
        self.transformation_weights = (
            None  # placeholder for normalisation matrices
        )
        self.in_theta_phi = (
            False  # [bool] if True, known_points are in spherical coords
        )
        self.checkTransformation = True  # [bool] generate plot to check transformation
        self.blockingPlot = (
            False  # [bool] prevent plots closing immediately when not saving
        )

    def update_user_settings(self, input_deck_path=None):
        """Load settings from a JSON file, resolving relative paths against the JSON's directory."""
        if input_deck_path is None:
            if len(sys.argv) > 1:
                input_deck_path = sys.argv[1]
            else:
                print("Warning: No input deck file provided,\nRunning Example Input")
                print("Usage: python pointing2d_main.py <input_deck_path>")
                input_deck_path = "EXAMPLES/LPL_Settings.json"

        input_deck_path = pathlib.Path(input_deck_path).resolve()
        input_deck_dir = input_deck_path.parent

        print(f"running with {input_deck_path=}")

        with open(input_deck_path, "r") as f:
            input_deck = json.load(f)

        PATH_PREFIX = "path:"

        for setting_name, setting_value in input_deck.items():
            if setting_name.startswith("_"):
                continue
            if hasattr(self, setting_name):
                if isinstance(setting_value, str) and setting_value.startswith(
                    PATH_PREFIX
                ):
                    candidate = pathlib.Path(setting_value[len(PATH_PREFIX) :])
                    setting_value = (input_deck_dir / candidate).resolve()
                setattr(self, setting_name, setting_value)
            else:
                print(f"Warning: {setting_name} not found in settings.")

    def assert_reasonable(self):
        """Validate that the current settings point to real files/directories."""
        ers = False
        msg = "Settings are NOT reasonable: \n  detected errors:\n"

        try:
            assert self.targetDir.exists(), "No Target Directory"
        except AssertionError as e:
            msg += "\t{}\n".format(e)
            ers = True

        if self.background is not None:
            try:
                assert self.background.exists(), "Background file doesn't exist"
            except AssertionError as e:
                msg += "\t{}\n".format(e)
                ers = True
            try:
                assert (
                    0 <= self.background_clip < 100
                ), "background_clip should be in range 0-100"
            except AssertionError as e:
                msg += "\t{}\n".format(e)
                ers = True

        if self.generate_background_files:
            try:
                assert (
                    self.background_dir.exists()
                ), "No Target Background Directory for Generating new Averages"
            except AssertionError as e:
                msg += "\t{}\n".format(e)
                ers = True

        if self.checkTransformation:
            cal_path = self.pointingCalibrationImage.resolve()
            try:
                assert (
                    cal_path.exists()
                ), f"No Calibration Image to Check at {cal_path}\n"
            except AssertionError as e:
                msg += "\t{}\n".format(e)
                ers = True

        if not ers:
            return True
        else:
            msg += "please edit the input JSON and try again"
            print(msg)
            return False


settings = Settings()

if __name__ == "__main__":
    settings.assert_reasonable()
