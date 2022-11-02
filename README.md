Model code: Comparison of artificial neural networks and reservoir
models for simulating karst spring discharge on five test sites in the
Alpine and Mediterranean regions
================
Guillaume Cinkus, Andreas Wunsch, Naomi Mazzilli, Tanja Liesch, Zhao
Chen, Nataša Ravbar, Joanna Doummar, Jaime Fernández-Ortega, Juan
Antonio Barberá, Bartolomé Andreo, Nico Goldscheider and Hervé Jourde

Preprint:

Cinkus, G., Wunsch, A., Mazzilli, N., Liesch, T., Chen, Z., Ravbar, N., Doummar, J., Fernández-Ortega, J., Barberá, J. A., Andreo, B., Goldscheider, N., and Jourde, H.: Comparison of artificial neural networks and reservoir models for simulating karst spring discharge on five test sites in the Alpine and Mediterranean regions, Hydrol. Earth Syst. Sci., 1–41, https://doi.org/10.5194/hess-2022-365, 2022.

Author ORCIDs:

- Guillaume Cinkus [0000-0002-2877-6551](https://orcid.org/0000-0002-2877-6551)  
- Andreas Wunsch [0000-0002-0585-9549](https://orcid.org/0000-0002-0585-9549)  
- Naomi Mazzilli [0000-0002-9145-5160](https://orcid.org/0000-0002-9145-5160)
- Tanja Liesch [0000-0001-8648-5333](https://orcid.org/0000-0001-8648-5333)
- Zhao Chen [0000-0003-0076-7079](https://orcid.org/0000-0003-0076-7079)
- Nataša Ravbar [0000-0002-0160-1460](https://orcid.org/0000-0002-0160-1460)
- Joanna Doummar [0000-0001-6146-1917](https://orcid.org/0000-0001-6146-1917)
- Jaime Fernández-Ortega [0000-0003-0183-3015](https://orcid.org/0000-0003-0183-3015)
- Juan Antonio Barberá [0000-0003-3379-0953](https://orcid.org/0000-0003-3379-0953)
- Bartolomé Andreo [0000-0002-3769-7329](https://orcid.org/0000-0002-3769-7329)
- Nico Goldscheider [0000-0002-8428-5001](https://orcid.org/0000-0002-8428-5001)
- Hervé Jourde [0000-0001-7124-4879](https://orcid.org/0000-0001-7124-4879)

# Description

This repository contains the following elements:

-   Reservoir model code
    -   KarstMod files for each studied system
    -   R script for performing the snow routine
-   ANN model code
	-	Code for each individual site (only slight differences between the files)
	-	Dummy data files to illustrate the input data structure

# Workflows

## Snow routine

The snow routine is detailed in the appendix D of the manuscript. The
routine is inspired from the work of Chen et al. (2018), which
successfully simulated spring discharge of a mountainous karst system
heavily influenced by snow accumulation and melt. The workflow is:

1.  Get time series of (i) precipitation, (ii) temperature, and (iii)
    potential clear-sky solar radiation (if needed)
2.  Define subcatchment (if needed) then calculate their areas and their
    relative proportion to the whole catchment
3.  Apply the snow routine function for each subcatchment. We recommend
    to shift the temperature time series according to an appropriate
    temperature gradient scaling with altitude. The inputs for the snow
    routine function are:
    -   temperature vector (T,1)
    -   precipitation vector (T,1)
    -   potential clear-sky solar radiation vector (T,1)
    -   model parameters vector: temperature threshold, melt factor,
        refreezing factor, water holding capacity of snow and radiation
        coefficient (T,5)
4.  Apply the relative proportion of each subcatchment to their
    corresponding P time series (output of the snow routine)
5.  Sum up the P time series of each subcatchment

**If working without solar radiation, radiation coefficient parameter
needs to be `0` and potential clear-sky solar radiation must be a vector
of `0` of the same length as temperature and precipitation time
series.**

## KarstMod

<img src="miscellaneous/karstmod.png" width="70" />

Information on the KarstMod platform can be found in the section 3.2 of
the manuscript. The main workflow is:

1.  Prepare the input data
2.  Open the appropriate KarstMod file (if needed)
3.  Import the input data
4.  Define warm-up/calibration/validation periods
5.  Define Output directory
6.  Run calibration

It is possible to modify the model parameters, the objective function,
the number of iterations, the maximum time, and other options. The
`Save` button allows to save the new modifications and to get a new
KarstMod file.

## ANNs

1D-Convolutional Neural Networks for karst spring discharge modeling. For details please see the according publication.

Dependencies: [Python 3.8](https://www.python.org/), [Tensorflow 2.7](https://www.tensorflow.org/), [BayesianOptimization 1.2](https://github.com/fmfn/BayesianOptimization), [Numpy 1.21](https://numpy.org/), [Pandas 1.4](https://pandas.pydata.org/), [Scipy 1.7](https://scipy.org/), [Scikit-learn 1.0](https://scikit-learn.org/stable/), [Matplotlib 3.5](https://matplotlib.org/)


# Resources

For more details about the KarstMod platform, please refer to the User
manual provided below.

For more details about the hydrological models, please refer to the
section 3 of the manuscript.

Download KarstMod: <https://sokarst.org/en/softwares-en/karstmod-en/>

Download KarstMod User manual:
<https://hal.archives-ouvertes.fr/hal-01832693>

# References

Chen, Z., Hartmann, A., Wagener, T., Goldscheider, N., 2018. Dynamics of
water fluxes and storages in an Alpine karst catchment under current and
potential future climate conditions. Hydrology and Earth System Sciences
22, 3807–3823. <https://doi.org/10.5194/hess-22-3807-2018>
