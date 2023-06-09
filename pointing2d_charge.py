from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import figure, cm
from matplotlib.colors import LogNorm


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
        r = 25, # resolution /um
        l = 5, # latitude
        s = 500, # PMT sensitivity
        b = 16, # bitdepth
       ):
    """
    A function for calculating the Photo Luminescence from the greyscale data of the scan 
    see ref 1
    """
    p = ((r**2)/10)*(4000/s)*10**(l*((g/(2**b-1))-0.5))
    
    return(p)

from scipy.constants import e as electron_charge

def n_e_MP(psl):
    MP_Respose = 16.2E-3 # (ref 2)
    electron_count = psl/MP_Respose
    return(electron_count)

def test_MP(g = 35000):
    p = psl(g)
    n_e,q = q_MP(p)
    print("A grey value of {} indicates {:.2f}PSL implying {} electrons at {:.2e} Coulombs".format(g,p,int(n_e),q))

def plot_MP():
    G = np.linspace(0,(2**16)-1,2**8, endpoint=True)
    Q = [psl(g)[2] for g in G]
    plt.plot(Q,G)
    plt.xscale("log")
    plt.xlabel("Charge / Coulomb")
    plt.ylabel("25um pixel Grey Value")

def adjust_MP_image(file,plot=True,crop = [[0,0],[-1,-1]]):
    plate_g = np.array(Image.open(file), np.float32)[crop[0][1]:crop[1][1],crop[0][0]:crop[1][0]]

    plate_q = np.multiply(n_e_MP(psl(plate_g)),electron_charge)

    fig,ax = plt.subplots(2,1)    

    fig.suptitle(file.split('/')[-1].split('\\')[-1].split('-')[1].replace("_", ' '))

    im_g = ax[0].imshow(plate_g, cmap=cm.gray_r)
    im_q = ax[1].matshow(plate_q, norm=LogNorm(vmin=plate_q.min(), vmax=plate_q.max()))
    fig.colorbar(im_g, label = "Grey Value")
    fig.colorbar(im_q, label = "Charge /C")
    fig.show()
    input("Done?...")

if __name__ == "__main__":
    adjust_MP_image("D:/Image_Plates_2023/20230502-chargeCal_run003_25um_l5_500PMT-[Phosphor].tif", crop = [[6000,700],[14000,4700]])