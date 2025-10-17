# FEM_CoSTA

This repository contains the code used for my experiments with using CoSTA for heat and elasticity modelling.


## File description

 - FEM.py : finite element source code. Includes a class "Heat" for solving the heat equation using backwards euler method. Heat.step() takes one temporal step, using an optional correction, this is used in the CoSTA approach. Similar classes "Heat_2D", and "Elasticity_2D" for these problems. "Disc" class holds discretization details and functions for initializing hte other FEM.py classes, as is done several times in solvers.py, e.g. for plotting

 - functions.py : manufactured solutions to test the models on

 - methods.py : data driven models, and CoSTA models, for solving the equations in FEM.py

 - parameter.py : parameters used for discretization, DNN architecture and training, for the different modes (bug fix, quick test ++)

 - plot_all : plot figures summarizing all results. 
 
 - plot_nonlinearity.py : plot figures showing the correct and approximate conductivity and youngs modulus in the non-linear cases

 - plotlearn.py : plot old learning curves

 - plotsol.py : plot solutions from function.py for visualization

 - preproj.py : code for running experiments in preproject (pgnn-costa for 1D heat eqn)

 - quadrature.py : Source code for numerical (gaussian) integration, used in FEM.py

 - replot.sh : Shell script for running run_files with several configurations (to create plots of solutions)

 - run_dimred : code for running experiments on heat with reduced dimensionality

 - run_elastics : code for running experiments on elasticity, with various modelling errors specifiable

 - run_extravark : code for running experiments on heat with extra varying conductivity

 - run_heat: code for running experiments on heat with varying conductivity

 - run_lowdimred : code for running experiments on heat with reduced dimensionality, with only 1 dimension

 - run_lowdimred_elastics : unfinished file for elastics dimensional reduction with few dimensions

 - solvers.py : The class "Solvers" is used for training, testing and making various result plots of the different methods (FEM and those in methods.py).

 - tests.py : Tests to see that certain parts of code works as intended.

 - tuner.py : script for finding optimal configurations.
