# LanexPointingLib
A Program for justifying and analysing images of lanex screens as a diagnostic for high energy electrons

settings can be configured in the 
pointing2d_settings.py file then the 
pointing2d_main.py file can be run to execute the program

REQUIREMENTS:
    numpy
    numba
    scipy
    pillow
    matplotlib
    lmfit
    opencv

installation : 
    conda install numpy numba scipy pillow matplotlib -y
    conda install -c conda-forge lmfit opencv -y
    
the environment spec-file is also available i.e.:
    conda create --name pointing2d --file spec-file.txt