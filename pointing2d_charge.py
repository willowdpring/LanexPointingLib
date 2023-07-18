from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, cm
from matplotlib.colors import LogNorm
from scipy.signal import convolve
import pointing2d_settings as settings
from pointing2d_lib import get_background, save_u16_to_tiff
import pointing2d_backfiltlib as backfilt
import pointing2d_perspective as perspective
from scipy.constants import e as electron_charge
from tqdm import tqdm
import cv2
from PIL import Image


########################################################################
#   Calibration, is done by comparison to a BAS_MP Phosphoir image plate
#   
#    At the moment the final comparisson is done manually (with a layer capeable image processor) 
#    as visual feedback is key to linging up the two images
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

def integrate_lanex(folder = "", 
                    include = [0,-1], 
                    lanex_filters = [1],
                    percentile = 50, 
                    bkg_percentile = 50,
                    background = None, 
                    plot = True, 
                    doTransform = False, 
                    src = [], 
                    dst = [], 
                    kernel = [[1]], 
                    save_each = False,
                    saveas = ""):
    """
    Background subtract and integraate labex images for the given folder

    NB. currently the xray filters and perspective transformation are hard coded in the function as I was not altering them run to run 

    Parameters
    ----------
    folder : string  
        the folder containing the images to be integrated
    include : tuple (2 array) 
        the slice to take from the image list
    lanex_filters : array of [3 | 5 | 53] 
        xray filters
    percentile : int 0 < N < 100 
        the median cutoff for noise suppression
    percentile : int 0 < N < 100 
        the median cutoff for background signal to be considered
    background: string 
        the file to take as a background for the run
    plot : bool 
        weather or not to plot the result with mpl
    doTransform : bool 
        transform images with src dst
    src : np.array[,float32]
        source points in pixel coordinates
    dst : np.array[,float32]
        destination points in theta, phi
    Kernel : 2d array 
        kernel for the convolution filter
    save_each : bool 
        if True each lanex image will also be saved
    saveas : string 
        the name to save the resulting tiff as

    Returns
    -------
    total_signal : 2darray the resulting image as a numpy array 

    """

    files = backfilt.walkDir(folder)[include[0]:include[1]]
    exportDir = folder + "\\OUTPUT"
    outFile = exportDir + "\\total_signal{}.tiff".format(saveas)
    if not os.path.exists(exportDir):  # check if it exists
        os.mkdir(exportDir)  # create it if not
    if save_each:
        if not os.path.exists(exportDir + "//individuals_{}".format(saveas)):  # check if it exists
            os.mkdir(exportDir + "//individuals_{}".format(saveas))  # create it if not

    tot = len(files)
    if background is not None:
        bkg_dat = np.array(Image.open(background))

        for f in lanex_filters:
            bkg_dat = backfilt.filterImage(bkg_dat, f)
    else:
        bkg_dat = np.zeros_like(np.array(Image.open(files[0])))
    
    bkg_smoothed = convolve(bkg_dat, kernel, mode='same')
    
    if doTransform:
        bkg_trans, _ = perspective.TransformToLanexPlane(bkg_smoothed,
                                        np.array(src, np.float32),
                                        np.array(dst, np.float32))
    
    bkg_cutoff = np.percentile(bkg_trans, bkg_percentile)
    bkg_trans = np.subtract(np.clip(bkg_trans, bkg_cutoff, np.inf), bkg_cutoff)
    
    total_signal = np.zeros_like(bkg_trans)

    for i in tqdm(range(tot)):
        image = files[i]
        im_dat = np.array(Image.open(image))
        
        for f in lanex_filters:
            im_dat = backfilt.filterImage(im_dat, f)
        
        im_smoothed = convolve(im_dat, kernel, mode='same')

        if doTransform:
            im_trans, _ = perspective.TransformToLanexPlane(im_smoothed,
                                            np.array(src, np.float32),
                                            np.array(dst, np.float32))
        
        im_trans = np.clip(im_trans-np.percentile(im_trans,percentile),0,np.inf)
        
        if background is not None:
            processed = np.clip(np.subtract(im_trans, bkg_trans),0,np.inf)
        else:
            processed = im_trans

        total_signal += processed

        if save_each:
            save_u16_to_tiff(np.uint16(processed),processed.shape[::-1],exportDir + "//individuals_{}".format(saveas)+"//PROC_{}".format(image.split("\\")[-1]))

    print(total_signal.min(), total_signal.max())

    if plot:
        fig, ax = plt.subplots()
        im = ax.imshow(total_signal, extent = [0,270,0,100])
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.show()

    save_u16_to_tiff(np.uint16(total_signal),total_signal.shape[::-1],exportDir + "//{}.tiff".format(saveas))
    return(total_signal)

def clean_image_plate(file, crop=[[0,-1],[0,-1]]):
    im_dat = np.array(Image.open(file))[crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]
    im_dat = im_dat[:,::-1] 
    fig,ax = plt.subplot_mosaic([['orig','slope'],
                                 ['signal','signal']])
    ax['orig'].imshow(im_dat,interpolation='none')
    ax['orig'].set_title("image plate (cropped)")
    slope = np.mean(im_dat[20:80,:],0)
    ax['slope'].plot(slope)
    ax['slope'].set_title("side electron slope")
    ax['slope'].grid('both')
    im_dat_subbed = im_dat
    im_dat += 10000
    for i, row in enumerate(im_dat):
        im_dat_subbed[i] = np.clip(np.subtract(row,slope),0,np.inf)
    im_dat_cropped = im_dat_subbed[1200:-1000,500:7500]
    ax['signal'].imshow(im_dat_cropped,interpolation='none')
    ax['signal'].set_title("Remaining Signal")
    fig.tight_layout()
    fig.show()
    if 'y' in input("save tiff? ...").lower():
        save_u16_to_tiff(np.uint16(im_dat_cropped),im_dat_cropped.shape[::-1], file.split(".")[0] + "_processed.tiff")


if __name__ == "__main__":
    # plot the image plate scan as charge:
    # adjust_MP_image("C:\\Users\\willo\\Documents\\BunkerC\\Charge_Calibrations\\20230502-chargeCal_run003_25um_l5_500PMT-[Phosphor].tif", crop = [[6000,700],[14000,4700]])
    
    # integrate lanex images:

    #backfilt.generateAndPlotBackgrounds("C:\\Users\\willo\\Documents\BunkerC\\Charge_Calibrations\\230707\\Lanex_ChargeCal\\BACKGROUND\\")

    clean_image_plate("C:\\Users\\willo\\Documents\BunkerC\\Charge_Calibrations\\230707\\20230707-chargeCal_run001_25um_l5_500PMT-[Phosphor].tif",
                      crop = [[3000,-1],[0,8000]])

    #input("Done?...")

    #    h = 900 # output heigth
    #    l = 1200 # output width
    """
    l=1288
    h=822

    cc_src = [[20,63],[74,824],[1212,822],[1260,101]]   # source coordinates for the perspective transformation
    cc_dst = [[-260,0],[-250,100],[-30,100],[-10,0]]    # destination coordinates for the perspective transformation
                    
    n = 0
    m = -1
    p_b = 10

    kx = 15
    ky = 7

    res = 20

    test_data = np.array(Image.open("C:\\Users\\willo\\Documents\BunkerC\\Charge_Calibrations\\230707\\Lanex_ChargeCal\\test\\Espec_calibration2_Marked.tiff"))
    fig,ax = perspective.check_lanex_transformation(test_data,
                         cc_src,
                         cc_dst,
                         units=0.001,
                         resolution=res,
                         saveDir=None)

    cc_dst = np.multiply(cc_dst,res)

    for p in [0,20,40,60,80]:
        integrate_lanex("C:\\Users\\willo\\Documents\BunkerC\\Charge_Calibrations\\230707\\Lanex_ChargeCal\\Good_Shots\\", 
                        include = [n,m], 
                        lanex_filters = [3,3,5,5,53,53,5,5,3,3], # the size of the sliding window xray filters to apply
                        percentile = p, 
                        bkg_percentile = p_b,
                        background = "C:\\Users\\willo\\Documents\BunkerC\\Charge_Calibrations\\230707\\Lanex_ChargeCal\\BACKGROUND\\EXPORTED\\AVG_BAK.tiff",
                        doTransform = True,
                        src = cc_src,
                        dst = cc_dst,
                        kernel = backfilt.norm_gaus2d_ary(kx, 3, ky, 3),
                        save_each = True,
                        saveas = "av_bkg_cutoff_{}_Kernel_[{},{}]".format(p,kx,ky)
                        )
    
    input("Done?...")

    """