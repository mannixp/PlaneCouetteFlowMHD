# PlaneCouetteFlowMHD
DNS script, initial condition and instructions required to simulate and reproduce the quasi-Keplerian plane Couette flow dynamo (P. M. Mannix, Y. Ponty, F. Marcotte 2021/2). The script is written for execution with the [Dedalus code (Burns et al. 2020)](https://doi.org/10.1103/PhysRevResearch.2.023068), for installation details see [Dedalus](https://dedalus-project.org) and for an animation of the resulting dynamo see [Dedalus gallery](https://dedalus-project.org/gallery/). 

<!-- Having installed Dedalus and activated its conda environment, unzip the initial condition and execute the script by running `unzip InitCond_Re20Pm75_T0.125Rm_M5e-05_MinSeed.h5.zip && mpiexec -np procs python3 QKEP_PCF_3D_MHD.py` -->

Having installed Dedalus and activated its conda environment, unzip the initial condition and execute by running `unzip InitCond_Re20Pm75_T0.125Rm_M5e-05_RandomSeed.h5.zip && mpiexec -np procs python3 QKEP_PCF_3D_MHD.py`

where procs is to be replaced with the desired/availabe number. The time-series and snapshots produced by the simulation, can be viewed by executing

`python3 Plot_Paper_figures.py && mkdir Plotted_Data && mv *.pdf Plotted_Data`

Figure 1. time-series can be reproduced by adding

`Plot_TimeSeries_KeMe("scalar_data_s1.h5")` 

in the main of Plot_Paper_figures.py

Figure 1. time-series can also be reproduced by specifying, *a minimum resolution* of 

`Nx=64,Ny=128,Ny=64, dt=0.0125, N_ITERS = 6*((Pm*Re)/dt)`

along with the parameters and *minimal seed* (currently not provided) specified therin, and running the script and plotting the data as above. 

Typically this execution demands a compute cluster or large workstation. Referencence using 32 cores, with a Modified Crank-Nicolson Adams-basforth integration is

```
2 AMD Epyc 7302 @ 3 GHz – 2*16 cores processors          - 180 cpu-hr - N_ITERS = 2*((Pm*Re)/dt)
```

however we strongly recommend using L-stable Runge-Kutta integrators, such as 2nd order Runge-Kutta which demand mutliples of this time - 580 cpu-hr.

Figure 1. snapshots can be reproduced in Paraview. To prepare the data for Paraview: add the pyevtk to your conda environment

`conda install -c conda-forge pyevtk `

and execute

`python3 ViewHDF5_SLAB.py`

