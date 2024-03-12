# Description
This is the code to generate the figures of the paper "Examining the SST Tendency Forced by Atmospheric Rivers" submitted to XXX journal.

# Prerequisite

1. Python >= 3.7
    - Matplotlib
    - Cartopy
    - Numpy
    - Scipy
    - netCDF4
2. ImageMagick >= 6.9.10

# Reproducing Figures

1. Clone this project.
2. Download the file `data-paperfigures-2023-AR-SST-response_20240122.zip` from [zenodo repository](https://doi.org/10.5281/zenodo.10039181)
3. Unzip to get folder `data20240122` generated in this git project root folder (i.e., the same folder containing this `README.md` file).
4. Run `00_runall.sh`.
5. The figures are generated in the folder `final_figures`.
