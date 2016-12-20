# Read Me

## Overview of Package Structure and Files
### Directory Tree
* PyLDA
  * LICENSE.txt
  * Manual.pdf
  * README.md
  * setup.py
  * SynthData.ipynb
  * pylda
    * \_\_init\_\_.py
    * main.py
    * pylda.py
    * fit
      * \_\_init\_\_.py
      * data.py
      * discreteslider.py
      * lda.py
      * svd_ga.py
    * data
      * dynamic.csv
      * dynamic_noise10.csv
      * fulldata.csv
      * fulldata_noise10.csv
      * hettaus.csv
      * hettaus_noise10.csv
    * test_data.py

### Code overview
    * SynthData.ipynb : a jupyter (ipython) notebook that contains the code for the construction of the synthetic data sets in the data folder.
    * main.py : Contains code for GUI
    * pylda.py : Wrapper file to run main.  Separate from main to allow implementation of new fitting modules
    * data.py : Data class.  Stores loaded csv files.  Keeps track of working data set by user specified bounds.  Simple data manipulation, such as low rank approximation is also implemented here.  All other objects access data through this class.
    * discreteslider.py : Make discrete integer matplotlib slider widgets for plots.
    * lda.py : Class containing routines for lifetime density analysis.  All regularization methods are implemented here, as well as relevant statistics.
    * svd_ga.py : Class containing routines for global analysis of the working data set based upon SVD.
    * All test_ files contain unittests for the respective classes and routines.

### .fit.data
#### \_\_init\_\_(f_in)
f_in is the path to the file to load.  Loading will store the data, set bounds to the nominal 0 time, and all wavelengths.  The SVD is also stored.

#### truncData(wLSVs)
Will perform a low rank approximation to the data based on the SVD.  wLSVs is a list of singular vector indices.  If there is only a single entry, e.g. [5], all the vectors upto, but not including that index, will be included in the approximation.  Otherwise, the specific vectors corresponding to all indices in the list will be included in the approximation.

#### display()
Plots the *entire* data set as a filled contour plot using a symlog plot for the time axis.

#### updateBounds(wl_lb, wl_ub, t0, t)
Updates the bounds for the working dataset such that the lower and upper bound wavelength *indices* are wl_lb and wl_ub.  The corresponding upper and lower bound time indices are t0 and t.

#### updateIRF(order, fwhm, munot, lamnot)

## Using the GUI
Please refer to the manual.pdf file for GUI instructions.

## For Future Development

## Known Issues
