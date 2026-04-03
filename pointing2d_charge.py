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

    try:
        fig.suptitle(file.split('/')[-1].split('\\')[-1].split('-')[1].replace("_", ' '))
    except IndexError:
        fig.suptitle(file)

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

def plot_overlay_images(image1, image2, tmax = 200, Mmax = 2):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.6)  # More space for controls

    # Initial transformation parameters
    translation_x = 0
    translation_y = 0
    rotation_deg = 0
    scale = 1  # Unified scale parameter
    flip_horizontal = False
    flip_vertical = False

    # Store initial values for reset functionality
    initial_values = {
        'translation_x': translation_x,
        'translation_y': translation_y,
        'rotation_deg': rotation_deg,
        'scale': scale,
        'flip_horizontal': flip_horizontal,
        'flip_vertical': flip_vertical
    }

    # Plot the initial overlay image
    overlay_image = overlay_images(image1, image2, translation_x, translation_y, rotation_deg, 
                                 scale, flip_horizontal, flip_vertical)
    im = ax.imshow(overlay_image)

    # Define sliders with improved ranges
    ax_translation_x = plt.axes([0.1, 0.5, 0.6, 0.025])
    ax_translation_y = plt.axes([0.1, 0.45, 0.6, 0.025])
    ax_rotation = plt.axes([0.1, 0.4, 0.6, 0.025])
    ax_scale = plt.axes([0.1, 0.35, 0.6, 0.025])

    slider_translation_x = Slider(ax_translation_x, 'Translation X', -tmax, tmax, valinit=translation_x)
    slider_translation_y = Slider(ax_translation_y, 'Translation Y', -tmax, tmax, valinit=translation_y)
    slider_rotation = Slider(ax_rotation, 'Rotation', -10, 10, valinit=rotation_deg)
    slider_scale = Slider(ax_scale, 'Scale', 1/Mmax, Mmax, valinit=scale)  # Unified scale slider

    # Text boxes for direct entry
    def create_textbox_handler(slider, param_name):
        def submit_handler(text):
            try:
                value = float(text)
                # Clamp values to slider ranges
                if param_name in ['translation_x', 'translation_y']:
                    value = max(-2000, min(2000, value))
                elif param_name == 'rotation_deg':
                    value = max(-180, min(180, value))
                elif param_name == 'scale':
                    value = max(0.1, min(5.0, value))
                slider.set_val(value)
            except ValueError:
                print(f"Invalid input for {param_name}: {text}")
        return submit_handler

    textbox_translation_x = TextBox(plt.axes([0.75, 0.5, 0.08, 0.025]), '', initial=str(translation_x))
    textbox_translation_x.on_submit(create_textbox_handler(slider_translation_x, 'translation_x'))

    textbox_translation_y = TextBox(plt.axes([0.75, 0.45, 0.08, 0.025]), '', initial=str(translation_y))
    textbox_translation_y.on_submit(create_textbox_handler(slider_translation_y, 'translation_y'))

    textbox_rotation = TextBox(plt.axes([0.75, 0.4, 0.08, 0.025]), '', initial=str(rotation_deg))
    textbox_rotation.on_submit(create_textbox_handler(slider_rotation, 'rotation_deg'))

    textbox_scale = TextBox(plt.axes([0.75, 0.35, 0.08, 0.025]), '', initial=str(scale))
    textbox_scale.on_submit(create_textbox_handler(slider_scale, 'scale'))

    # Flip checkboxes using buttons (matplotlib doesn't have native checkboxes)
    ax_flip_h = plt.axes([0.1, 0.3, 0.12, 0.04])
    ax_flip_v = plt.axes([0.25, 0.3, 0.12, 0.04])
    
    button_flip_h = Button(ax_flip_h, 'Flip H: OFF')
    button_flip_v = Button(ax_flip_v, 'Flip V: OFF')

    def toggle_flip_horizontal(event):
        nonlocal flip_horizontal
        flip_horizontal = not flip_horizontal
        button_flip_h.label.set_text(f'Flip H: {"ON" if flip_horizontal else "OFF"}')
        update_image()

    def toggle_flip_vertical(event):
        nonlocal flip_vertical
        flip_vertical = not flip_vertical
        button_flip_v.label.set_text(f'Flip V: {"ON" if flip_vertical else "OFF"}')
        update_image()

    button_flip_h.on_clicked(toggle_flip_horizontal)
    button_flip_v.on_clicked(toggle_flip_vertical)

    # Reset button
    ax_reset = plt.axes([0.4, 0.3, 0.08, 0.04])
    button_reset = Button(ax_reset, 'Reset')

    def reset_all(event):
        nonlocal flip_horizontal, flip_vertical
        # Reset sliders
        slider_translation_x.reset()
        slider_translation_y.reset()
        slider_rotation.reset()
        slider_scale.reset()
        
        # Reset flip states
        flip_horizontal = initial_values['flip_horizontal']
        flip_vertical = initial_values['flip_vertical']
        button_flip_h.label.set_text('Flip H: OFF')
        button_flip_v.label.set_text('Flip V: OFF')
        
        # Reset text boxes
        textbox_translation_x.set_val(str(initial_values['translation_x']))
        textbox_translation_y.set_val(str(initial_values['translation_y']))
        textbox_rotation.set_val(str(initial_values['rotation_deg']))
        textbox_scale.set_val(str(initial_values['scale']))
        
        update_image()

    button_reset.on_clicked(reset_all)

    def update_image():
        """Helper function to update the displayed image"""
        translation_x = slider_translation_x.val
        translation_y = slider_translation_y.val
        rotation_deg = slider_rotation.val
        scale = slider_scale.val

        overlay_image = overlay_images(image1, image2, translation_x, translation_y, rotation_deg, 
                                     scale, flip_horizontal, flip_vertical)
        im.set_data(overlay_image)
        
        # Update text boxes to reflect slider values (useful when sliders are moved)
        textbox_translation_x.set_val(f"{translation_x:.1f}")
        textbox_translation_y.set_val(f"{translation_y:.1f}")
        textbox_rotation.set_val(f"{rotation_deg:.1f}")
        textbox_scale.set_val(f"{scale:.3f}")
        
        plt.draw()

    def update(val):
        update_image()

    # Connect slider events
    slider_translation_x.on_changed(update)
    slider_translation_y.on_changed(update)
    slider_rotation.on_changed(update)
    slider_scale.on_changed(update)

    # Enhanced save/print button
    ax_button = plt.axes([0.7, 0.3, 0.12, 0.04])
    button = Button(ax_button, 'Save/Print')
    
    def save_or_print(event):
        translation_x = slider_translation_x.val
        translation_y = slider_translation_y.val
        rotation_deg = slider_rotation.val
        scale = slider_scale.val
        
        transformation_matrix = get_transformation_matrix(image1, translation_x, translation_y, 
                                                        rotation_deg, scale, scale)
        print("=" * 50)
        print("TRANSFORMATION PARAMETERS:")
        print(f"Translation X: {translation_x:.2f}")
        print(f"Translation Y: {translation_y:.2f}")
        print(f"Rotation: {rotation_deg:.2f}°")
        print(f"Scale: {scale:.3f}")
        print(f"Flip Horizontal: {flip_horizontal}")
        print(f"Flip Vertical: {flip_vertical}")
        print("\nTransformation matrix:")
        print(transformation_matrix)
        print("=" * 50)
        
    button.on_clicked(save_or_print)

    plt.show()


def overlay_images(image1, image2, translation_x, translation_y, rotation_deg, scale, 
                  flip_horizontal=False, flip_vertical=False):
    """
    Enhanced overlay function with flip support and unified scaling
    """
    # Start with a copy of image2 to avoid modifying the original
    working_image = image2.copy()
    
    # Apply flips first (before other transformations)
    if flip_horizontal:
        working_image = cv2.flip(working_image, 1)  # Horizontal flip
    if flip_vertical:
        working_image = cv2.flip(working_image, 0)  # Vertical flip
    
    # Get image center for rotation
    center_x = working_image.shape[1] / 2
    center_y = working_image.shape[0] / 2
    
    # Create rotation matrix (without scaling first)
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), rotation_deg, 1.0)
    
    # Apply uniform scaling by directly modifying the transformation matrix
    rotation_scale_matrix = rotation_matrix.copy()
    rotation_scale_matrix[0, 0] *= scale  # Scale x-component of x-axis
    rotation_scale_matrix[0, 1] *= scale  # Scale x-component of y-axis  
    rotation_scale_matrix[1, 0] *= scale  # Scale y-component of x-axis
    rotation_scale_matrix[1, 1] *= scale  # Scale y-component of y-axis
    
    # Add translation
    rotation_scale_matrix[0, 2] += translation_x
    rotation_scale_matrix[1, 2] += translation_y
    
    # Apply the complete transformation
    transformed_image2 = transform_image(working_image, rotation_scale_matrix, image1.shape)

    # Create overlay with improved normalization
    overlay_image = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.float32)
    
    # Normalize images safely (avoid division by zero)
    img1_norm = image1.astype(np.float32)
    img2_norm = transformed_image2.astype(np.float32)
    
    if np.max(img1_norm) > 0:
        img1_norm = img1_norm / np.max(img1_norm)
    if np.max(img2_norm) > 0:
        img2_norm = img2_norm / np.max(img2_norm)
    
    overlay_image[:, :, 0] = img1_norm  # Red channel
    overlay_image[:, :, 2] = img2_norm  # Blue channel

    return overlay_image

if __name__ == "__main__":
    plate_file = r"C:\Users\willo\Documents\Lab\Experiments\ELI25\Electrons\reconsructed_imageplate.tiff"
    plate_im = np.array(Image.open(plate_file),dtype = np.float64)
    lan_file = r"C:\Users\willo\Documents\Lab\Experiments\ELI25\Electrons\ELI_CC\L1\RunChargeCal\L1-1_16.00.49.890.tif"
    lan_im = np.array(Image.open(lan_file),dtype = np.float64)
    # dy = -560, dx = -120, M =2.65
    # Enable interactive mode
    plt.ion()
    plot_overlay_images(plate_im, lan_im)
    plt.show(block=True)  # Force blocking show
    
    input("Done?...")