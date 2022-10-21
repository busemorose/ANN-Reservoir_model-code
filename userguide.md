Model code: Comparison of artificial neural networks and reservoir
models for simulating karst spring discharge on five test sites in the
Alpine and Mediterranean regions
================
Guillaume Cinkus, Andreas Wunsch, Naomi Mazzilli, Tanja Liesch, Zhao
Chen, Nataša Ravbar, Joanna Doummar, Jaime Fernández-Ortega, Juan
Antonio Barberá, Bartolomé Andreo, Nico Goldscheider and Hervé Jourde

# Description

This repository contains the following elements:

-   Reservoir model code
    -   KarstMod files for each studied system
    -   R script for performing the snow routine
-   ANN model code

# Workflow

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

<img src="data/karstmod.png" width="70" />

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

# Resources

For more details about the KarstMod platform, please refer to the User
manual provided below.

For more details about the hydrological models, please refer to the
section 3 of the manuscript.

Download KarstMod: <https://sokarst.org/en/softwares-en/karstmod-en/>

Download KarstMod User manual:
<https://hal.archives-ouvertes.fr/hal-01832693>

## References

Chen, Z., Hartmann, A., Wagener, T., Goldscheider, N., 2018. Dynamics of
water fluxes and storages in an Alpine karst catchment under current and
potential future climate conditions. Hydrology and Earth System Sciences
22, 3807–3823. <https://doi.org/10.5194/hess-22-3807-2018>
