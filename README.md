# Master-Thesis-Code (M.R.D.)
This is the code used in the Master thesis of Maximilian Roy Drozd to perform an oligomeric analysis of fluorescence microscopy data. In particular the oligomeric analysis for Thinning Out Clusters while Conserving the Stoichiometry of Labeling (TOCCSL) measurements can be performed with this code. The code is based on a template by Marina Bishara, it was adapted and extended by Maximilian Roy Drozd.
## Features
The software contains the following features, all either in direct connection to, or in preparation to perform an oligomeric analysis:
* Brightness correction of the raw data using the recorded laser illumination profile.
* Localisation of particle features and their brightnesses using the `sdt-python` package (Schrangl, L. https://doi.org/10.5281/zenodo.6802801).
* Tools to set and help in choosing a certain region of interest.
* An oligomeric analysis based on a brightness based differentiation of differently sized oligomers.
* An oligomeric analysis differentiating differently sized oligomer populations based on their bleaching behaviour.
* A colocalization analysis, using measurements performed in 2 color channels to determine the dimer fraction of a probe consisting of monomers and dimers.
## Dependencies
* sdt-python
* numpy
* matplotlib
* scipy
* pandas
* pims
* trackpy
* pathlib
* tqdm
* tifffile
* ipywidgets
* random
* seaborn
* warnings
* math
* os
* sklearn
* skimage
* time
## Getting Started
You can get started immediately by opening the jupyter notebook `TOCCSL_main.ipynb` and following the comments present there. The code was originally designed to analyse tracking data from a folder called `sm` and TOCCSL data from a folder called `TOCCSL`, both withing the same data path that can be set in the notebook. Example code on how to perform a two-color colocalization analysis can be found in the jupyter notebook `2color_analysis_example.ipynb`. The python file `TOCCSL.py` is there to import necessary packages. The oligomeric analysis code is all found within the python file `toccsl_analysis.py`. The other python files are there to provide utility functions, or tools necessary for the data preparation, such as correcting the raw data, or setting the region of interest.
## Credits
The template for this code was written by Marina Bishara. The changes made by Maximilian Drozd are the following: The python file `toccsl_analysis.py` and the jupyter nutebook `TOCCSL_main.ipynb` were heavily altered and extended, `2color_analysis_example.ipynb` was newly created. The changes in the remaining python files restrict themselves to minor adaptations and bug-fixes.
