from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, cm
from matplotlib.colors import LogNorm
from scipy.signal import convolve
import pointing2d_settings as settings
from pointing2d_lib import get_background 
import pointing2d_backfiltlib as backfilt
import pointing2d_perspective as perspective
from scipy.constants import e as electron_charge
from tqdm import tqdm
import cv2
from PIL import Image


########################################################################
#   Calibration, is done by comparison to a BAS_MP Phosphoir image plate
#    
#    References: 
#       [1] doi:10.1063/1.3531979 
#       [2] http://dx.doi.org/10.1063/1.4936141 
#       [3] http://dx.doi.org/10.1063/1.4950860
#
########################################################################
def psl(g,
        r = 25, 
        l = 5, 
        s = 500, 
        b = 16, 
       ):
    """
    A function for calculating the Photo Luminescence from the greyscale data of the scan taken with the FLA 7000 
    see ref 1

    Parameters
    ----------
    g : b-bit uint       grey value from scan
    r : int              resolution
    l : int  (4 or 5)    latitude
    s : int 500 - 4000   sensitivity of the photo multiplier tube (PMT)
    b : int 8, 12 or 16  bit depth

    Returns
    -------
    p : float  the value of the photo stimulated luminescence for the pixel

    """
    p = ((r**2)/10)*(4000/s)*10**(l*((g/(2**b-1))-0.5))
    
    return(p)


def n_e_MP(psl):
    """
    A function for calculating the number of electrons incident on a BAS-MP image plate from the PSL 

    Parameters
    ----------
    psl : float  the value of the photo stimulated luminescence for the pixel

    Returns
    -------
    electron_count : float  the number of electrons incident on the pixel

    """
    MP_Respose = 16.2E-3 # (ref 2)
    electron_count = psl/MP_Respose
    return(electron_count)

def test_MP(g = 35000):
    # sanity check
    p = psl(g)
    n_e = n_e_MP(p)
    q = n_e * electron_charge
    print("A grey value of {} indicates {:.2f}PSL implying {} electrons at {:.2e} Coulombs".format(g,p,int(n_e),q))

def plot_MP():
    """
    A function for plotting the response curves for the image plate

    """
    G = np.linspace(0,(2**16)-1,2**8, endpoint=True)
    Q = [psl(g)[2] for g in G]
    plt.plot(Q,G)
    plt.xscale("log")
    plt.xlabel("Charge / Coulomb")
    plt.ylabel("25um pixel Grey Value")

def adjust_MP_image(file,crop = [[0,0],[-1,-1]]):
    """
    A function for converting and plotting an image plate image

    Parameters
    ----------
    file : string  path to image file
    crop : a 2d slice to crop the image 
    
    Returns
    -------
    electron_count : float  the number of electrons incident on the pixel

    """
    plate_g = np.array(Image.open(file), np.float32)[crop[0][1]:crop[1][1],crop[0][0]:crop[1][0]]

    plate_q = np.multiply(n_e_MP(psl(plate_g)),electron_charge)

    fig,ax = plt.subplots(2,1)

    fig.suptitle(file.split('/')[-1].split('\\')[-1].split('-')[1].replace("_", ' '))

    im_g = ax[0].imshow(plate_g, cmap=cm.gray_r)
    im_q = ax[1].matshow(plate_q, norm=LogNorm(vmin=plate_q.min(), vmax=plate_q.max()))
    fig.colorbar(im_g, label = "Grey Value")
    fig.colorbar(im_q, label = "Charge /C")
    fig.show()


def integrate_lanex(folder = "", include = [0,-1], percentile = 50, background = None, plot = True, kernel = [[1]], saveas = ""):
    """
    Background subtract and integraate labex images for the given folder

    NB. currently the xray filters and perspective transformation are hard coded in the function as I was not altering them run to run 

    Parameters
    ----------
    folder : string  the folder containing the images to be integrated
    include : tuple (2 array) the slice to take from the image list
    percentile : int 0 < N < 100 the median cutoff for noise suppression
    background: string the file to take as a background for the run
    plot : bool weather or not to plot the result with mpl
    Kernel : 2d array kernel for the convolution filter
    saveas : string the name to save the resulting tiff as

    Returns
    -------
    total_signal : 2darray the resulting image as a numpy array 

    """
    h = 900 # output heigth
    l = 1200 # output width
    src = [[0,0],[0,1288],[964,1288],[964,0]] # source coordinates for the perspective transformation
    dst_flat = [[0,0],[0,l],[h,l],[h,0]]      # destination coordinates for the perspective transformation
    files = backfilt.walkDir(folder)[include[0]:include[1]]
    outFile = folder + "\\OUTPUT\\total_signal{}.tiff".format(saveas)

    lanex_filters = [3,5,3] # the size of the sliding window xray filters to apply

    tot = len(files)
    if background is not None:
        bkg_dat = im_dat = np.array(Image.open(background))
        for f in lanex_filters:
            bkg_dat = backfilt.filterImage(bkg_dat, f)
        
        bkg_transformed, axis = perspective.TransformToLanexPlane(bkg_dat,
                                            np.array(src, np.float32),
                                            np.array(dst_flat, np.float32))
        bkg_smoothed = convolve(bkg_transformed, kernel, mode='same')

    total_signal = np.zeros((h,l),dtype=np.float64)

    for i in tqdm(range(tot)):
        image = files[i]
        im_dat = np.array(Image.open(image))
        for f in lanex_filters:
            im_dat = backfilt.filterImage(im_dat, f)
        im_transformed, _ = perspective.TransformToLanexPlane(im_dat,
                                            np.array(src, np.float32),
                                            np.array(dst_flat, np.float32))
        im_smoothed = convolve(im_transformed, kernel, mode='same')
        im_smoothed = np.clip(im_smoothed-np.percentile(im_smoothed,percentile),0,np.inf)
        
        if background is not None:
            total_signal += np.clip(np.subtract(im_smoothed, bkg_smoothed),0,np.inf)
        else:
            total_signal += np.clip(im_smoothed,0,np.inf)


    print(total_signal.min(), total_signal.max())

    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(total_signal)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.show()

    o_im = Image.fromarray(np.uint16(total_signal))
    o_im.save(outFile,'TIFF')
    return(total_signal)

def transform_and_scale_image(source_image, target_image):
    ##### 
    # 
    #   Unused I was attempting to overplot the imageplate and lanex images but this is incomplete
    # 
    #####

    # Load images
    source = cv2.imread(source_image)
    target = cv2.imread(target_image)

    # Find keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(source, None)
    kp2, des2 = orb.detectAndCompute(target, None)

    # Match keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Find corresponding keypoints
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Calculate homography matrix
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Warp source image to match target image
    transformed = cv2.warpPerspective(source, M, (target.shape[1], target.shape[0]))

    # Scale the transformed image to fit the target image
    scale_ratio = target.shape[1] / transformed.shape[1]
    scaled_transformed = cv2.resize(transformed, None, fx=scale_ratio, fy=scale_ratio)

    return scaled_transformed


if __name__ == "__main__":
    # plot the image plate scan as charge:
    #adjust_MP_image("C:\\Users\\willo\\Documents\\BunkerC\\Charge_Calibrations\\20230502-chargeCal_run003_25um_l5_500PMT-[Phosphor].tif", crop = [[6000,700],[14000,4700]])
    
    # integrate lanex images:
    n = 0
    m = -1
    for p in [20,40,60,80]:
        print("{} : ".format(p))
        integrate_lanex("C:\\Users\\willo\\Documents\BunkerC\\Charge_Calibrations\\Run003\\", 
                        include = [n,n+m], 
                        percentile = p, 
                        background = None, # "C:\\Users\\willo\\Documents\BunkerC\\Charge_Calibrations\\Run003\\BACKGROUND\\AVG_BG.tiff",
                        kernel = backfilt.norm_gaus2d_ary(7, 3, 7, 3),
                        saveas = "_low_filt_cutoff_{}".format(p)
                        )
    input("Done?...")

