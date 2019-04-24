A COSMOSIS wrapper for hi_class.
================================

Installation
------------

- Clone this repository
- Set up COSMOSIS:
    
    source cosmosis/config/setup-cosmosis

- Make the hi_class installation:

    cd hi_class_public
    
    make

Running
-------

- See example ini files in the repo:

    [hi_class]
    
    file = INSTALLDIR/hi_class/hi_class_interface.py
    
    sBBN_file = INSTALLDIR/hi_class/hi_class_public/bbn/sBBN.dat
    
    expansion_model = lcdm
    
    gravity_model = propto_omega

- ...where INSTALLDIR refers to where you cloned the repo.


Plots
-----

- The plot in plots/models_Dl_fisgma8.png shows a quick validation test, and should be compared to Figures 3 and 4 of Noller and Nicola 2018 (https://arxiv.org/abs/1811.12928)