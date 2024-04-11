# Description
This is the code to generate the figures of the paper "Response of Sea Surface Temperature to Atmospheric Rivers" submitted to Nature Communications.

# Prerequisite

Python >= 3.7
  - matplotlib
  - cartopy
  - numpy
  - scipy
  - netcdf4
  - xarray 
  - pandas
  - svg\_stack\
  - cairosvg

# Reproducing Figures

1. Clone this project.
2. Download the file `data-paperfigures-2023-AR-SST-response_20240122.zip` from [zenodo repository](https://doi.org/10.5281/zenodo.10039181)
3. Unzip to get folder `data20240122` generated in this git project root folder (i.e., the same folder containing this `README.md` file).
4. Run `00_main.sh`.
5. The figures are generated in the folder `final_figures`.
