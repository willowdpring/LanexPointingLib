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
from skimage.feature import peak_local_max
import cv2
from PIL import Image
import diplib as dip
from matplotlib.widgets import Slider, TextBox, Button

########################################################################
#   Calibration, is done by comparison to a BAS_MP Phosphoir image plate
#   
#    At the moment the final comparisson is done manually (with a layer capeable image processor) 
#    as visual feedback is key to linging up the two images
#    
#    References: 
#       [1] http://dx.doi.org/10.1063/1.3531979 
#       [2] http://dx.doi.org/10.1063/1.4936141 
#       [3] http://dx.doi.org/10.1063/1.4950860
#
#    ALTERNATE PSL METHOD FOR NEW SCANNER SOFTWARE:
#       [4] https://doi.org/10.1063/1.4886390
#
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

def alt_psl(g,
        r = 25, 
        l = 5, 
        s = 500, 
        b = 16, 
       ):
    """
    An alternate function for calculating the Photo Luminescence from the greyscale data of the scan taken with the FLA 7000 
    see ref 4

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


    h = 4000/s
    """ from [4]
     " h(V) = 4, 1, and 0.4, which corresponds to the three discrete Fuji-brand sensitivity settings of S1000, S4000, or S10000. "
    """

    p = (g/(2**b-1))**2  * ((r/100)**2) *  h * 10**(l/2)
    return(p)

def fade_MP(ft):
    return(0.565* np.exp(-ft/18.461) + 0.435 * np.exp(-ft/6117.5))


def n_e_MP(psl,fade_time = 110): #0.43):
    """
    A function for calculating the number of electrons incident on a BAS-MP image plate from the PSL 

    Parameters
    ----------
    psl : float  the value of the photo stimulated luminescence for the pixel

    Returns
    -------
    electron_count : float  the number of electrons incident on the pixel

    """
               
    fade = fade_MP(fade_time)


    MP_Respose = 16.2E-3*fade # (ref 2)
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
    Q = [alt_psl(g)[2] for g in G]
    plt.plot(Q,G)
    plt.xscale("log")
    plt.xlabel("Charge / Coulomb")
    plt.ylabel("25um pixel Grey Value")

def adjust_MP_image(file,alt=True,crop = [[0,0],[-1,-1]]):
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

    filetype = file.split('.')[-1]

    plate_g = np.array(Image.open(file), np.float32)[crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]] #[0][1]:crop[1][1],crop[0][0]:crop[1][0]]
    if alt:
        plate_q = np.clip(np.multiply(n_e_MP(alt_psl(plate_g)),electron_charge),0,np.inf)
    else:
        plate_q = np.clip(np.multiply(n_e_MP(psl(plate_g)),electron_charge), 0 , np.inf)

    fig,ax = plt.subplots(2,1,figsize = (12,9))

    alt_str = " With Alternate PSL formula" if alt else ""

    fig.suptitle(file.split('/')[-1].split('\\')[-1].split('-')[1].replace("_", ' ') + " " + filetype + " " + alt_str)

    im_g = ax[0].imshow(plate_g, cmap=cm.gray_r)

    im_q = ax[1].matshow(plate_q, norm=LogNorm(vmin=max(plate_q.min(),1.e-17), vmax=plate_q.max()))
    ax[1].set_xticklabels([])
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

def integrate_circular_patch(arr, center, radius):
    nrows, ncols = arr.shape
    row_indices, col_indices = np.indices((nrows, ncols))
    distances = np.sqrt((row_indices - center[0])**2 + (col_indices - center[1])**2)
    circular_patch = np.where(distances <= radius, arr, 0)
    
    # Trapezoidal rule for integration
    integral = np.sum(circular_patch)

    return integral



def clean_image_plate(file,alt = True, crop=[[0,-1],[0,-1]], bkg_scale = 1):
    im_dat = np.array(Image.open(file),dtype = np.float64)[crop[0][0]:crop[0][1],crop[1][0]:crop[1][1]]
    im_dat = im_dat[:,::-1]
  
    slope_g = np.mean(im_dat[20:80,:],0)
    if alt:
        slope_psl = alt_psl(slope_g)
        
    else:
        slope_psl = psl(slope_g)
        
    slope_n_e = n_e_MP(slope_psl)

    fig,ax = plt.subplot_mosaic([['orig'],
                                 ['slope'],
                                 ['signal']], figsize = (9,12), height_ratios = [1,0.5,1])

    fig.suptitle(file.split('/')[-1].split('\\')[-1].split('-')[1].replace("_", ' '))

    ax['slope'].plot(np.multiply(slope_n_e,electron_charge))
    ax['slope'].set_yscale('log')
    ax['slope'].set_title("side electron slope (C / px)")
    ax['slope'].grid('both')
    
    if alt:
        im_dat_psl = alt_psl(im_dat)
    else:
        im_dat_psl = psl(im_dat)

    im_dat_n_e = n_e_MP(im_dat_psl)
    
    origIm = ax['orig'].imshow(np.multiply(im_dat_n_e,electron_charge),interpolation='none')
    ax['orig'].set_title("image plate (cropped) (C/px)")
    cax2 = fig.colorbar(origIm,
                                        ax=ax['orig'],
                                        pad=0,
                                        shrink=1,
                                        location='right')
    #print("full integral is {} electrons for {}C".format(im_dat_n_e.sum(),electron_charge*im_dat_n_e.sum()))

    k = (20,30)

    slope_n_e = np.asarray(dip.Gauss(slope_n_e,k[0]))
    im_dat_n_e = np.asarray(dip.Gauss(im_dat_n_e,k))

    im_dat_subbed = np.ones_like(im_dat)
    for i, row in enumerate(im_dat_n_e):
        for j, q in enumerate(row):
            im_dat_subbed[i,j] = q - (bkg_scale * slope_n_e[j])

    im_dat_cropped = np.multiply(np.clip(im_dat_subbed[1200:-1000,500:7500],0.01,np.inf),electron_charge)
    sigIm = ax['signal'].imshow(im_dat_cropped,norm=LogNorm(),interpolation='none')


    ax['signal'].set_title(f"total Signal (C /px) with background scaled by {bkg_scale}")
    cax1 = fig.colorbar(sigIm,
                                        ax=ax['signal'],
                                        pad=0,
                                        shrink=1,
                                        location='right')
    fig.tight_layout()
    fig.show()

    peaks = peak_local_max(im_dat_cropped, num_peaks=10, threshold_rel=1e-16, min_distance = 200)
    #print(peaks)

    """
    result = integrate_circular_patch(arr, center, radius)
    circle = plt.Circle(center, radius, color='r', fill=False)
    ax.add_patch(circle)
    """

    fig2, ax2 = plt.subplots(1,1,figsize=(9,12))
    fullIm = ax2.imshow(im_dat_cropped,interpolation='none')
    ax2.set_title("total Signal (q /px)")
    radius = 200
    for peak in peaks:
        center_y, center_x = peak
        integral = integrate_circular_patch(im_dat_cropped, (center_y, center_x), radius)
        circle = plt.Circle((center_x, center_y), radius, color='r', fill=False)
        ax2.add_patch(circle)
        ax2.text(center_x + radius + 1, center_y, f'Integral: {integral:.2e}', verticalalignment='center', color = 'r')

    
    cax1 = fig2.colorbar(fullIm,
                                        ax=ax2,
                                        pad=0.01,
                                        shrink=0.6,
                                        location='right')
    
    #ax2[1].imshow(peaks,interpolation='none')
    
    fig2.show()

    return im_dat_cropped
    #if 'y' in input("save tiff? ...").lower():
    #    save_u16_to_tiff(im_dat_cropped,im_dat_cropped.shape[::-1], file.split(".")[0] + "_processed.tiff",norm=True)


def transform_image(image, transformation_matrix, output_shape):
    """
    Apply a transformation to the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        transformation_matrix (numpy.ndarray): 2x3 transformation matrix.
        output_shape (tuple): Desired shape (height, width) of the output image.

    Returns:
        numpy.ndarray: Transformed image.
    """
    # Get the desired output shape
    output_height, output_width = output_shape
    
    # Apply the transformation to the input image
    transformed_image = cv2.warpAffine(image, transformation_matrix, (output_width, output_height))
    
    return transformed_image


def plot_overlay_images(image1, image2):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.5)

    # Initial transformation parameters
    translation_x = 0
    translation_y = 0
    rotation_deg = 0
    scale_x = 1
    scale_y = 1

    # Plot the initial overlay image
    overlay_image = overlay_images(image1, image2, translation_x, translation_y, rotation_deg, scale_x, scale_y)
    im = ax.imshow(overlay_image)

    # Define sliders
    ax_translation_x = plt.axes([0.1, 0.4, 0.65, 0.03])
    ax_translation_y = plt.axes([0.1, 0.35, 0.65, 0.03])
    ax_rotation = plt.axes([0.1, 0.3, 0.65, 0.03])
    ax_scale_x = plt.axes([0.1, 0.25, 0.65, 0.03])
    ax_scale_y = plt.axes([0.1, 0.2, 0.65, 0.03])

    slider_translation_x = Slider(ax_translation_x, 'Translation X', -100, 100, valinit=translation_x)
    slider_translation_y = Slider(ax_translation_y, 'Translation Y', -100, 100, valinit=translation_y)
    slider_rotation = Slider(ax_rotation, 'Rotation', -180, 180, valinit=rotation_deg)
    slider_scale_x = Slider(ax_scale_x, 'Scale X', 0.1, 2, valinit=scale_x)
    slider_scale_y = Slider(ax_scale_y, 'Scale Y', 0.1, 2, valinit=scale_y)

    # Text boxes for direct entry
    def submit_translation_x(text):
        slider_translation_x.set_val(float(text))
    textbox_translation_x = TextBox(plt.axes([0.8, 0.4, 0.1, 0.03]), '', initial=str(translation_x))
    textbox_translation_x.on_submit(submit_translation_x)

    def submit_translation_y(text):
        slider_translation_y.set_val(float(text))
    textbox_translation_y = TextBox(plt.axes([0.8, 0.35, 0.1, 0.03]), '', initial=str(translation_y))
    textbox_translation_y.on_submit(submit_translation_y)

    def submit_rotation(text):
        slider_rotation.set_val(float(text))
    textbox_rotation = TextBox(plt.axes([0.8, 0.3, 0.1, 0.03]), '', initial=str(rotation_deg))
    textbox_rotation.on_submit(submit_rotation)

    def submit_scale_x(text):
        slider_scale_x.set_val(float(text))
    textbox_scale_x = TextBox(plt.axes([0.8, 0.25, 0.1, 0.03]), '', initial=str(scale_x))
    textbox_scale_x.on_submit(submit_scale_x)

    def submit_scale_y(text):
        slider_scale_y.set_val(float(text))
    textbox_scale_y = TextBox(plt.axes([0.8, 0.2, 0.1, 0.03]), '', initial=str(scale_y))
    textbox_scale_y.on_submit(submit_scale_y)

    def update(val):
        translation_x = slider_translation_x.val
        translation_y = slider_translation_y.val
        rotation_deg = slider_rotation.val
        scale_x = slider_scale_x.val
        scale_y = slider_scale_y.val

        overlay_image = overlay_images(image1, image2, translation_x, translation_y, rotation_deg, scale_x, scale_y)
        im.set_data(overlay_image)
        plt.draw()

    slider_translation_x.on_changed(update)
    slider_translation_y.on_changed(update)
    slider_rotation.on_changed(update)
    slider_scale_x.on_changed(update)
    slider_scale_y.on_changed(update)

    # Button to save/print transformation matrix
    ax_button = plt.axes([0.7, 0.1, 0.1, 0.05])
    button = Button(ax_button, 'Save/Print')
    
    def save_or_print(event):
        transformation_matrix = get_transformation_matrix(image1, translation_x, translation_y, rotation_deg, scale_x, scale_y)
        print("Transformation matrix:")
        print(transformation_matrix)
        # You can save the transformation matrix to a file or use it as needed
        
    button.on_clicked(save_or_print)

    plt.show()

def overlay_images(image1, image2, translation_x, translation_y, rotation_deg, scale_x, scale_y):
    # Apply transformations to image2
    transformation_matrix = cv2.getRotationMatrix2D((image2.shape[1] / 2, image2.shape[0] / 2), rotation_deg, scale_x)
    transformation_matrix[:, 2] += [translation_x, translation_y]
    transformed_image2 = transform_image(image2, transformation_matrix, image1.shape)

    # Overlay images
    overlay_image = np.zeros((image1.shape[0], image1.shape[1], 3))
    overlay_image[:, :, 0] = image1 / np.max(image1)  # Red channel
    overlay_image[:, :, 2] = transformed_image2 / np.max(transformed_image2)  # Blue channel

    return overlay_image

if __name__ == "__main__": 
    """
    counts = np.linspace(2**12,2**16,1000)
    fig,ax=plt.subplots(1,1)
    ax.set_title("Comparison of PSL functions")
    ax.plot(counts,psl(counts),'g',label='original')
    ax.plot(counts,alt_psl(counts),'r',label='alternate')
    ax.set_yscale('log')
    ax.grid('both','both')
    ax.legend()
    fig.show()
    # plot the image plate scan as charge:
    #imageplate_tif = "D:\\Bunker C\\20230502-chargeCal\\20230502-chargeCal_run003_25um_l5_500PMT-[Phosphor].tif" 
    #imageplate_gel = "D:\\Bunker C\\20230502-chargeCal\\20230502-chargeCal_run003_25um_l5_500PMT-[Phosphor].gel" 
    imageplate_tif = "D:\\Bunker C\\Charge_Calibrations\\230707\\20230707-chargeCal_run001_25um_l5_500PMT-[Phosphor].tif" 
    
    endpoint = -1
    """

    imageplate_gel = "D:\\Bunker C\\Charge_Calibrations\\230707\\20230707-chargeCal_run001_25um_l5_500PMT-[Phosphor].gel" 
    x1 = 3000
    x2 = -1
    y1 = 0
    y2 = 7500
    crop = [[x1,x2],[y1,y2]]

    for bkg_scale in [0.98,1,1.02]:
        plate_im = clean_image_plate(imageplate_gel,alt = True, crop = crop, bkg_scale=bkg_scale)

    """  
    clean_image_plate(imageplate_tif,crop = [[3000,-1],[0,8000]])
    #integrate lanex images:

    # LANEX:
    # backfilt.generateAndPlotBackgrounds("D:\\Bunker C\\Charge_Calibrations\\230707\\Lanex_ChargeCal\\BACKGROUND\\")

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

    test_data = np.array(Image.open("D:\\Bunker C\\Charge_Calibrations\\230707\\Lanex_ChargeCal\\test\\Espec_calibration2_Marked.tiff"))
    fig,ax = perspective.check_lanex_transformation(test_data,
                         cc_src,
                         cc_dst,
                         units=0.001,
                         resolution=res,
                         saveDir=None)

    cc_dst = np.multiply(cc_dst,res)

    lan_im = integrate_lanex("D:\\Bunker C\\Charge_Calibrations\\230707\\Lanex_ChargeCal\\Good_Shots\\",
                    include = [n,m], 
                    lanex_filters = [3,3,5,5,53,53,5,5,3,3], # the size of the sliding window xray filters to apply
                    percentile = 20, 
                    bkg_percentile = p_b,
                    background = "D:\\Bunker C\\Charge_Calibrations\\230707\\Lanex_ChargeCal\\BACKGROUND\\EXPORTED\\AVG_BAK.tiff",
                    doTransform = True,
                    src = cc_src,
                    dst = cc_dst,
                    kernel = backfilt.norm_gaus2d_ary(kx, 3, ky, 3),
                    save_each = False,
                    saveas = "av_bkg_cutoff_{}_Kernel_[{},{}]".format(20,kx,ky)
                    )

    np.save("./plate_data",plate_im)
    np.save("./lanex_data",lan_im)

    """
    


#    plate_im = np.load("./plate_data.npy")
#    lan_im = np.load("./lanex_data.npy")
#    
#    plot_overlay_images(plate_im,lan_im)

    input("Done?...")

