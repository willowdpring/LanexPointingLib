import json
import PIL

import numpy as np

from scipy.optimize import curve_fit

from _settings_cache import settings

## hack to json encode numpy

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return super().default(obj)

json._default_encoder = NumpyEncoder()


def vprint(message, keep=False):
    if settings.verbose:
        if keep:
            print(message,end="")
        else:
            print(message)

def integrate_gausians(x2, y2, result, src, dst):
    """
    A function that generates data from the plots the fitted gausians and transforms them back into the camera frame to integrate
    TODO: UNUSED

    Parameters
    ----------
    x2 : np.array
        a 2d array of x coordinates as returned from numpy.from meshgrid
    y2 : np.array
        a 2d array of y coordinates as returned from numpy.from meshgrid
    result : ModelResult
        fit returned from LMFIT
        see https://lmfit.github.io/lmfit-py/model.html#lmfit.model.ModelResult
    src : np.array[,float32]
        source points in pixel coordinates
    dst : np.array[,float32]
        destination points in theta, phi

    Returns
    -------
    Integral: array[float]
        the integrals of the two isolated gaussians
    """

    gaussians = []  # [ [amplitude, offset, xo, yo, theta, sigma_x, sigma_y] ]
    integrals = []  # [ float ]

    if result.model.name == "Model(lm_gaus2d)":
        # decompose the two gaussians
        gauss = [
            result.best_values["amplitude"],
            0,
            result.best_values["xo"],
            result.best_values["yo"],
            result.best_values["sigma_x"],
            result.best_values["sigma_y"],
            result.best_values["theta"],
        ]

    elif result.model.name == "Model(lm_double_gaus2d)":
        # zero the offset
        gaussians.append(
            [
                result.best_values["amplitude_1"],
                0,
                result.best_values["xo_1"],
                result.best_values["yo_1"],
                result.best_values["sigma_x_1"],
                result.best_values["sigma_y_1"],
                result.best_values["theta_1"],
            ]
        )
        gaussians.append(
            [
                result.best_values["amplitude_2"],
                0,
                result.best_values["xo_2"],
                result.best_values["yo_2"],
                result.best_values["sigma_x_2"],
                result.best_values["sigma_y_2"],
                result.best_values["theta_2"],
            ]
        )

    else:
        print(
            "Sorry Integration of Gausian Bunches not Implimented for the Model : {}".format(
                result.model.name
            )
        )
        integrals.append(0)

    if len(gaussians) != 0:
        for gaus in gaussians:
            integrals.append(gaus[0] * gaus[4] * gaus[5] * np.sqrt(np.pi * 2))
    return integrals

def save_u16_to_tiff(imDatIn, size, tiff_filename, norm=False):
    """
    ## https://blog.itsayellow.com/technical/saving-16-bit-tiff-images-with-pillow-in-python/# ##

    Since Pillow has poor support for 16-bit TIFF, we make our own
    save function to properly save a 16-bit TIFF.
    """
    # IF NORMALISING, RESCALE IMAE TO FILL 16 BIT
    print(np.max(imDatIn))
    if norm:
        imDatIn = (imDatIn / imDatIn.max()) * (2**16 - 1)

    u16in = np.uint16(imDatIn)

    # write 16-bit TIFF image

    # PIL interprets mode 'I;16' as "uint16, little-endian"
    img_out = PIL.Image.new("I;16", size)

    outpil = u16in.astype(u16in.dtype.newbyteorder("<")).tobytes()

    img_out.frombytes(outpil)
    img_out.save(tiff_filename)

def points_to_roi(points, w, h, x_pad=150, y_pad=100):
    if len(points.items()) == 0:
        raise ValueError("Expects dict of points (int,int,bool) (x,y,good/bad) ")

    x_min = y_min = 10000
    x_max = y_max = 0

    for [lab, [x, y, good]] in points.items():
        # x = point[1][0]
        # y = point[1][1]
        # good = point[1][2]
        x_min = max(
            min(x_min, (x - (1.5 * x_pad) if good else x - (3 * x_pad))), 0
        )  # extra low E padding
        x_max = min(max(x_max, (x + x_pad if good else x + (2 * x_pad))), w)
        y_min = max(min(y_min, (y - y_pad if good else y - (2 * y_pad))), 0)
        y_max = min(max(y_max, (y + y_pad if good else y + (2 * y_pad))), h)

    return [[x_min, y_min], [x_max, y_max]]

def load_dict_from_file(filepath):
    points_dict = {}
    # Read the contents of the file
    with open(filepath, "r") as file:
        # Skip the header
        header_skipped = False
        for line in file:
            # Remove leading and trailing whitespace, including newline characters
            line = line.strip()
            if not header_skipped:
                if line.endswith(" = {"):
                    header_skipped = True
                continue

            # Check if the line is not empty
            if line and line != "}":
                # Split the line into key and value using ":" as delimiter
                try:
                    lineparts = line.split(":")
                    key = ":".join(lineparts[0:2])  # C:/path
                    value = ":".join(lineparts[2:])
                except ValueError as e:
                    raise ValueError("file contains line \n\t" + line) from e
                # Remove leading and trailing whitespace from key and value
                key = key.strip("'")
                value = eval(value.strip())
                # Add key-value pair to the dictionary
                points_dict[key] = value
            elif line == "}":
                # End of the dictionary
                break
    return points_dict

if __name__ == "__main__":
    print("this is not the main file")
