# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 15:21:22 2022

@author: willo

A library for fiting 2 dimensional gausians to 2d arrays

"""

import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt
import pointing2d_settings as settings

def lm_gaus2d(x, y, amplitude, offset, xo, yo, theta, sigma_x, sigma_y):
    """
    a 2d gaussian function 
    Parameters
    ----------
    x : float
        x coordinate of the point to sample
    y : float
        y coordinate of the point to sample
    amplitude : float
        the amplitude of the gaussian
    offset : float
        the background offset
    xo : float
        the x coordinate of the peak
    yo : float
        the y coordinate of the peak
    theta : float
        the rotation of sigma_x
    sigma_x : float
        the standard defiation in x
    sigma_y : float
        the standard deviation in y

    Returns
    -------
    intensity : float
        the amplitude of the gaussian at x,y

    """

    theta = np.radians(theta)
    sigx2 = sigma_x**2
    sigy2 = sigma_y**2
    a = np.cos(theta)**2 / (2 * sigx2) + np.sin(theta)**2 / (2 * sigy2)
    b = np.sin(theta)**2 / (2 * sigx2) + np.cos(theta)**2 / (2 * sigy2)
    c = np.sin(2 * theta) / (4 * sigx2) - np.sin(2 * theta) / (4 * sigy2)

    expo = -a * (x - xo)**2 - b * (y - yo)**2 - 2 * c * (x - xo) * (y - yo)
    return (amplitude * np.exp(expo) + offset)


def lm_double_gaus2d(x, y, amplitude_1, offset, xo_1, yo_1, theta_1, sigma_x_1,
                     sigma_y_1, amplitude_2, xo_2, yo_2, theta_2, sigma_x_2,
                     sigma_y_2):
    """
    a sum of two 2d gaussians 
    Parameters
    ----------
    x : float
        x coordinate of the point to sample
    y : float

    parameters: see lm_gaus2d() for specifics

    Returns
    -------
    intensity : float
        the amplitude of the gaussians at x,y

    """
    theta_1 = np.radians(theta_1)

    sigx2_1 = sigma_x_1**2
    sigy2_1 = sigma_y_1**2
    a_1 = np.cos(theta_1)**2 / (2 * sigx2_1) + np.sin(theta_1)**2 / (2 *
                                                                     sigy2_1)
    b_1 = np.sin(theta_1)**2 / (2 * sigx2_1) + np.cos(theta_1)**2 / (2 *
                                                                     sigy2_1)
    c_1 = np.sin(2 * theta_1) / (4 * sigx2_1) - np.sin(
        2 * theta_1) / (4 * sigy2_1)
    expo_1 = -a_1 * (x - xo_1)**2 - b_1 * (y - yo_1)**2 - 2 * c_1 * (
        x - xo_1) * (y - yo_1)
    g_1 = amplitude_1 * np.exp(expo_1)

    theta_2 = np.radians(theta_2)
    sigx2_2 = sigma_x_2**2
    sigy2_2 = sigma_y_2**2
    a_2 = np.cos(theta_2)**2 / (2 * sigx2_2) + np.sin(theta_2)**2 / (2 *
                                                                     sigy2_2)
    b_2 = np.sin(theta_2)**2 / (2 * sigx2_2) + np.cos(theta_2)**2 / (2 *
                                                                     sigy2_2)
    c_2 = np.sin(2 * theta_2) / (4 * sigx2_2) - np.sin(
        2 * theta_2) / (4 * sigy2_2)
    expo_2 = -a_2 * (x - xo_2)**2 - b_2 * (y - yo_2)**2 - 2 * c_2 * (
        x - xo_2) * (y - yo_2)
    g_2 = amplitude_2 * np.exp(expo_2)

    return (np.sum([offset, g_1, g_2]))


def gaus(x, a, ux, s, c):
    """
    a one dimensional gaussian

    Parameters
    ----------
    x : float 
        sample point
    a : float 
        amplitude
    ux : float
        mean
    s : float
        sigma
    c : float
        constant

    Returns
    ----------
    intensity : float
        the amplitude of the gaussians at x
    """
    g = a * np.exp(-((x - ux) * s)**2 / s**2) + c
    return (g)


def linear(x, a, c):
    """
    a linear function
    Parameters
    ----------
    x : float 
        sample point
    a : float 
        coeficint of x    
    c : float 
        constant

    Returns
    -------
    y : float
        sample at x
    """
    return (np.multiply(x, a) + c)


def getest2DGF(x, y, I):
    """
    estimates parameters for a 2d gaussian fit 

    SEE https://www.astro.rug.nl/~vogelaar/Gaussians2D/2dgaussians.html
        Given the meshgrid (Xg,Yg) in x and y and the image intensities on that grid in I
        then use the moments of your image to calculate constants a, b and c from which we can
        calculate the semi major axis length (1 sigma) and semi minor axis legth (1 sigma) and
        the angle between the major axis and the positive x axis measured counter clockwise.
        If values a, b and c do not comply to the conditions for an ellipse, then return None.
        The calling environment should check the return value   

    Parameters
    ----------
    x : np.meshgrid of x coordinates
    y : np.meshgrid of y coordinates 
    I : the image data to fit

    Returns
    ---------- A, of, x0, y0, sx, sy, theta
    A : float
        the estimated amplitude of the gaussian
    of : float
        the estimated background offset
    x0 : float
        the estimated x coordinate of the peak
    y0 : float
        the estimated y coordinate of the peak
    sx : float
        the estimated standard defiation in x
    sy : float
        the standard deviation in y
    theta : float
        the estimated rotation of sigma_x from x

    -------
    """
    M0 = I.sum()
    x0 = (x * I).sum() / M0
    y0 = (y * I).sum() / M0
    Mxx = (x * x * I).sum() / M0 - x0 * x0
    Myy = (y * y * I).sum() / M0 - y0 * y0
    Mxy = (x * y * I).sum() / M0 - x0 * y0
    D = 2 * (Mxx * Myy - Mxy * Mxy)
    a = Myy / D
    b = Mxx / D
    c = -Mxy / D
    if a * b - c * c < 0 or a <= 0 or b <= 0:
        return None

    # Find the area of one pixel expressed in grids to find amplitude A
    Nx = x[0].size
    Ny = y[:, 0].size
    dx = abs(x[0, 0] - x[0, -1]) / Nx
    dy = abs(y[0, 0] - y[-1, -1]) / Ny
    A = dx * dy * M0 * (a * b - c * c)**0.5 / np.pi

    p = ((a - b)**2 + 4 * c * c)**0.5
    theta = np.degrees(0.5 * np.arctan(2 * c / (a - b)))
    if a - b > 0:  # Not HW1 but the largest axis corresponds to theta.
        theta += 90.0
    if theta < 0:
        theta += 180

    # Major and minor axis lengths
    major = (1 / (a + b - p))**0.5
    minor = (1 / (a + b + p))**0.5
    sx = major
    sy = minor
    ofe = np.percentile(I, 20)
    return (A, ofe, x0, y0, sx, sy, theta)


def getestdbl2DGF(x, y, I):
    """
    estimates parameters for the summation of two a 2d gaussian fit 

    ASSUMPTION:
    the first gaussian (_1) is a large background signal
    the second gaussian (_2) is a small signal due to an electron bunch

    Parameters
    ----------
    x : np.meshgrid of x coordinates
    y : np.meshgrid of y coordinates 
    I : the image data to fit   
    Returns
    -------
    A1 ofe, x01, y01, sx1, sy1, theta1, A2, x02, y02

    A1 : float
        the estimated amplitude of the first gaussian
    ofe : float
        the estimated background offset
    x01 : float
        the estimated x coordinate of the peak of the first gaussian
    y01 : float
        the estimated y coordinate of the peak of the first gaussian
    sx1 : float
        the estimated standard defiation in x of the first gaussian
    sy1 : float
        the standard deviation in y of the first gaussian
    theta1 : float
        the estimated rotation of the first gaussian
    A2 : float
        the estimated amplitude of the first gaussian
    x02 : float
        the estimated x coordinate of the peak of the second gaussian
    y02 : float    
        the estimated y coordinate of the peak of the second gaussian
    """

    M0 = I.sum()
    x01 = (x * I).sum() / M0
    y01 = (y * I).sum() / M0
    Mxx1 = (x * x * I).sum() / M0 - x01 * x01
    Myy1 = (y * y * I).sum() / M0 - y01 * y01
    Mxy1 = (x * y * I).sum() / M0 - x01 * y01
    D1 = 2 * (Mxx1 * Myy1 - Mxy1 * Mxy1)
    a1 = Myy1 / D1
    b1 = Mxx1 / D1
    c1 = -Mxy1 / D1
    if a1 * b1 - c1 * c1 < 0 or a1 <= 0 or b1 <= 0:
        return None

    # Find the area of one pixel expressed in grids to find amplitude A
    Nx1 = x[0].size
    Ny1 = y[:, 0].size
    dx1 = abs(x[0, 0] - x[0, -1]) / Nx1
    dy1 = abs(y[0, 0] - y[-1, -1]) / Ny1
    A1 = dx1 * dy1 * M0 * (a1 * b1 - c1 * c1)**0.5 / np.pi

    p1 = ((a1 - b1)**2 + 4 * c1 * c1)**0.5
    theta1 = np.degrees(0.5 * np.arctan(2 * c1 / (a1 - b1)))
    if a1 - b1 > 0:  # Not HW1 but the largest axis corresponds to theta.
        theta1 += 90.0
    if theta1 < 0:
        theta1 += 180

    # Major and minor axis lengths
    major1 = (1 / (a1 + b1 - p1))**0.5
    minor1 = (1 / (a1 + b1 + p1))**0.5
    sx1 = major1
    sy1 = minor1

    xi, yi = np.unravel_index(I.argmax(), I.shape)

    x02 = x[xi, yi]
    y02 = y[xi, yi]
    ofe = np.percentile(I, 20)
    A2 = np.max(I) - ofe
    return A1, ofe, x01, y01, sx1, sy1, theta1, A2, x02, y02


def fit_gauss2d_lm(x2, y2, z, fmodel):
    """
    uses lmfit (https://lmfit.github.io/lmfit-py/) to fit a 2d gaussian to the data

    Parameters
    ----------
    x2 : np.meshgrid of x coordinates
    y2 : np.meshgrid of y coordinates 
    z : the image data to fit       
    fmodel : class model
        lmfit model generated from a 2d gaussian see setup_2d_gauss_model()

    Returns
    -------
    result : class ModelResult
        see https://lmfit.github.io/lmfit-py/model.html#lmfit.model.ModelResult
    """
    Ae, ofe, x0e, y0e, sxe, sye, thetae = getest2DGF(x2, y2, z)
    print("Found initial estimates: ", Ae, x0e, y0e, sxe, sye, thetae)

    result = fmodel.fit(z,
                        x=x2,
                        y=y2,
                        amplitude=Ae,
                        offset=ofe,
                        xo=x0e,
                        yo=y0e,
                        theta=thetae,
                        sigma_x=sxe,
                        sigma_y=sye)

    print(result.fit_report())

    return (result)


def fit_double_gauss2d_lm(x2, y2, z, fmodel):
    """
    uses lmfit (https://lmfit.github.io/lmfit-py/) to fit a double 2d gaussian to the data

    Parameters
    ----------
    x2 : np.meshgrid of x coordinates
    y2 : np.meshgrid of y coordinates 
    z : the image data to fit       
    fmodel : class Model
        lmfit model generated from two summed 2d gaussian see setup_double_2d_gauss_model()

    Returns
    -------
    result : class ModelResult
        see https://lmfit.github.io/lmfit-py/model.html#lmfit.model.ModelResult
    """
    A1e, ofe, x01e, y01e, sx1e, sy1e, theta1e, A2e, x02e, y02e = getestdbl2DGF(
        x2, y2, z)
    print("Found initial estimates: ")
    printvars = [A1e, ofe, x01e, y01e, sx1e, sy1e, theta1e, A2e, x02e, y02e]
    printnames = [
        'A1e', 'ofe', 'x01e', 'y01e', 'sx1e', 'sy1e', 'theta1e', 'A2e', 'x02e',
        'y02e'
    ]
    for i, v in enumerate(printvars):
        print("{}\t =\t {}".format(printnames[i], v))

    result = fmodel.fit(z,
                        x=x2,
                        y=y2,
                        amplitude_1=A1e,
                        offset=ofe,
                        xo_1=x01e,
                        yo_1=y01e,
                        sigma_x_1=sx1e,
                        sigma_y_1=sy1e,
                        theta_1=theta1e,
                        amplitude_2=A2e,
                        xo_2=x02e,
                        yo_2=y02e)

    print(result.fit_report())

    return (result)


def setup_2d_gauss_model():
    """
    setup an lmfit model based on the 2d gaussian (lm_gaus2d) with some limits 

    Returns
    -------
    fmodel : class Model

    """
    fmodel = lm.Model(lm_gaus2d, independent_vars=('x', 'y'))
    # 'amplitude', 'offset', 'xo', 'yo', 'theta', 'sigma_x', 'sigma_y'
    fmodel.set_param_hint('sigma_x', min=0.1, max=5)
    fmodel.set_param_hint('sigma_y', min=0.1, max=5)
    fmodel.set_param_hint('theta', value=0, min=0, max=180)
    return (fmodel)


def setup_double_2d_gauss_model():
    """
    setup an lmfit model based on summed 2d gaussian (lm_double_gaus2d) with some limits 

    Returns
    -------
    fmodel : class Model

    """
    fmodel = lm.Model(lm_double_gaus2d, independent_vars=('x', 'y'))
    # x, y, amplitude_1, offset, xo_1, yo_1, theta_1, sigma_x_1, sigma_y_1,
    # amplitude_2, xo_2, yo_2, theta_2, sigma_x_2, sigma_y_2
    fmodel.set_param_hint('sigma_x_1', min=5)
    fmodel.set_param_hint('sigma_y_1', min=5)

    fmodel.set_param_hint('sigma_x_2', min=0.1, max=5)
    fmodel.set_param_hint('sigma_y_2', min=0.1, max=5)

    fmodel.set_param_hint('theta_1', value=0, min=0, max=180)
    fmodel.set_param_hint('theta_2', value=0, min=0, max=180)
    return (fmodel)


def plot_test_gausians():
    xary = np.linspace(1, 100, 100, endpoint=True)
    yary = np.linspace(1, 100, 100, endpoint=True)
    for sx in [12, 20]:
        for sy in [8, 13]:
            for th in [0, -90, 45]:
                zary = np.array([[lm_gaus2d(x_v, y_v, amplitude=1, offset=0, xo=50, yo=50, theta=th, sigma_x=sx, sigma_y=sy) for x_v in xary] for y_v in yary])
                fig, ax = plt.subplots(1, 1)
                ax.imshow(zary)
                ax.set_title("s_x={}, s_y={}, theta={}".format(sx,sy,th))
                fig.show()

    settings.blockingPlot = True

if __name__ == "__main__":
    plot_test_gausians()


    if settings.blockingPlot:
        input("press RETURN key to continue ...")  # this is here to stop plots from closing immediatly if you are not saving them
