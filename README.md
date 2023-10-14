# Rational Approximation for many body spectral functions
Repository corresponding to the paper https://arxiv.org/abs/2306.11038


`spectral-fit.jl` contains the code we used to do spectral fitting, including functions to generate some plots from the manuscript. 

`example-functions.jl` contains example functions which the rational approximation method can do to fit general functions of the user's choice.

`generate_spectra_for_calc.ipynb` is the python code used to generate the spectral dataset for fitting.

`Example_22_04_01.json` is the JSON file containing 100 spectral functions of the Holstein model with differing phonon frequencies. It is a sample of the full dataset which you can generate using `generate_spectra_for_calc.ipynb`. The functions given in `spectral-fit.jl` can be used to fit these sample spectra and generate the necessary plots.  
