"""
Dedalus script for 3D Quasi-Keplerian MHD Plane Couette flow dynamo (P. M. Mannix, Y. Ponty, F. Marcotte 2021/2)

This script uses a Fourier basis in the (stream-wise) x and (span-wise) y directions,
and Chebshev basis in the (shear-wise) z direction. No-slip velocity U = \pm 1 is
enforced by decomposing \vec{U}(x,y,z,t) = V(z)\vec{x} + \vec{u}(x,y,z,t).
Perfectly conducting magnetic boundary conditions B_z = dz(B_x) = dz(B_y) = 0, are
enforced alongside the div(B) = 0 condition using a Lagrange mutiplier \Pi. For
details of this approach see (A. Guseva et al. 2015 New J. Phys. 17). The zero net flux
condition (1/V)int \vec{B} dV = 0 is enforced in a manner similar to that of pipe flow.

The equations are scaled using the: 
half channel width L = d L^*
shear time-scale   t = (d/U)t^*
where ^* denotes non-dimensional length/time respectively.

This script should be ran in parallel using a compute cluster, and would be most efficient 
using a 2D process mesh.  It uses the built-in analysis framework to save 3D snapshots
and diagnostics such as volume integrated magnetic <B,B> and kinetic energy <U,U>
in HDF5 files.  These are merged at the end of the script, alternatively `merge_procs` 
command can be used to merge distributed analysis sets from parallel runs, and 
the `QKEP_PCF_3D_MHD.py` script can be used to plot the time-series and some 2D slices.

To run, and plot using 4 processes, for instance, you could use:
    $ mpiexec -n 4 python3 QKEP_PCF_3D_MHD.py
    $ mpiexec -n 4 python3 Plot_Paper_figures.py

The simulation should take roughly 90 cpu-hrs/Ohmic time to run for the specified parameters,
this result is based on using a AMD Epyc 7302 (2x16 cores) node.

"""
import sys,os,mpi4py,time

# Prevent multi-threading upon initialising mpi4py
os.environ["OMP_NUM_THREADS"] = "1";
mpi4py.rc.thread_level = 'single';

import numpy as np
from mpi4py import MPI
from dedalus import public as de
from dedalus.extras import flow_tools

import logging
root = logging.root
for h in root.handlers:
	h.setLevel("INFO");
	#h.setLevel("DEBUG")
logger = logging.getLogger(__name__)


# Parameters
alpha = 0.375; beta = 1.0; 
Lx, Ly, Lz = ( (2.*np.pi)/alpha, (2.*np.pi)/beta, 2. )
Re = 25.;   # Re = |U|d/\nu,     where /nu is the kinematic viscoity
mu = 4./3.; # mu = -2*Omega*d/U, where Omega is the rotation rate
Pm = 75.0;  # Pm = \nu/\eta,	 where \eta is the Ohmic diffusivity
Rm = Re*Pm;

Nx,Ny,Nz = 64,128,64; dt = 1.*(0.0125);
MESH_SIZE = None; # Process mesh, for example use [16,16] for 256 cores with Nx,Ny,Nz =256,256,64

T_opt = 2.0;
N_ITERS = int(T_opt*(Rm/dt));
N_SUB_ITERS = N_ITERS//50;

# Create bases and domain
start_init_time = time.time()
x_basis = de.Fourier('x', Nx, interval=(0, Lx), dealias=3/2)
y_basis = de.Fourier('y', Ny, interval=(0, Ly), dealias=3/2)
z_basis = de.Chebyshev('z', Nz, interval=(-Lz/2, Lz/2), dealias=3/2)
domain = de.Domain([x_basis, y_basis, z_basis], grid_dtype=np.float64,mesh=MESH_SIZE)

# Velocity field vector \vec{u}  = u x^ + v y^ + w z^; 
# Magnetic field vector \vec{B}  = A x^ + B y^ + C z^;
# where x^,y^,z^ denote unit vectors.

PCF = de.IVP(domain, variables=['Flux_x','Flux_y', 'p',  'u', 'v', 'w',   'uz', 'vz', 'wz',	'Pi', 'A', 'B', 'C',  'Az', 'Bz', 'Cz'], time='t');
PCF.meta[:]['z']['dirichlet'] = True;
PCF.meta['Flux_x']['z']['constant'] = True;
PCF.meta['Flux_y']['z']['constant'] = True;


PCF.parameters['Re'] = Re;
PCF.parameters['Rm'] = Rm;
PCF.parameters['Re_rot'] = mu;

PCF.parameters['inv_Vol'] = 1./domain.hypervolume;
PCF.parameters['inv_Lx'] = 1./abs( domain.bases[0].interval[1] - domain.bases[0].interval[0] )

PCF.substitutions['Avg_Bx(f)'] = "inv_Lx*integ(f,'x')"; 
PCF.substitutions['Lap(f,fz)'] = "dx(dx(f)) + dy(dy(f)) + dz(fz)";

# For u x ( NAB x u ), and, B x (NAB x B)
# 1.B) ~~~~~~ (NAB x A) ~~~~~~~~~~ Correct
PCF.substitutions['W_x(A_y ,A_z )'] = "dy(A_z) - dz(A_y)"; 
PCF.substitutions['W_y(A_x ,A_z )'] = "dz(A_x) - dx(A_z)"; 
PCF.substitutions['W_z(A_x ,A_y )'] = "dx(A_y) - dy(A_x)";

# 1.B) ~~~~~~ A x (NAB x A) ~~~~~~~~~~ Correct
PCF.substitutions['F_x(A_x, A_y, A_z )'] = "A_y*W_z(A_x ,A_y) - A_z*W_y(A_x ,A_z)"; 
PCF.substitutions['F_y(A_x, A_y, A_z )'] = "A_z*W_x(A_y ,A_z) - A_x*W_z(A_x ,A_y)"; 
PCF.substitutions['F_z(A_x ,A_y, A_z )'] = "A_x*W_y(A_x ,A_z) - A_y*W_x(A_y ,A_z)";

# 1.C) ~~~~~~ (NAB x (A x B) ~~~~~~~~~~ Correct
PCF.substitutions['EMF_x(A2,A3,B2,B3)'] = "A2*B3 - A3*B2";
PCF.substitutions['EMF_y(A1,A3,B1,B3)'] = "A3*B1 - A1*B3";
PCF.substitutions['EMF_z(A1,A2,B1,B2)'] = "A1*B2 - A2*B1";

PCF.substitutions['INDx(A1,A2,A3,B1,B2,B3)'] = "dy( EMF_z(A1,A2,B1,B2) ) - dz( EMF_y(A1,A3,B1,B3) )"; 
PCF.substitutions['INDy(A1,A2,A3,B1,B2,B3)'] = "dz( EMF_x(A2,A3,B2,B3) ) - dx( EMF_z(A1,A2,B1,B2) )"; 
PCF.substitutions['INDz(A1,A2,A3,B1,B2,B3)'] = "dx( EMF_y(A1,A3,B1,B3) ) - dy( EMF_x(A2,A3,B2,B3) )";

#######################################################
# add equations
#######################################################
logger.info("--> Adding Equations");

# Navier-Stokes
PCF.add_equation("dx(u) + dy(v) + wz = 0");
PCF.add_equation("uz - dz(u) = 0"); 
PCF.add_equation("vz - dz(v) = 0");
PCF.add_equation("wz - dz(w) = 0");
PCF.add_equation("dt(u) - (1./Re)*Lap(u,uz) + dx(p) - Re_rot*w + 1.*w                  = F_x(u,v,w) - F_x(A,B,C)"); 
PCF.add_equation("dt(v) - (1./Re)*Lap(v,vz) + dy(p)                    + 1.*z*W_z(u,v) = F_y(u,v,w) - F_y(A,B,C)");
PCF.add_equation("dt(w) - (1./Re)*Lap(w,wz) + dz(p) + Re_rot*u - 1.*u  - 1.*z*W_y(u,w) = F_z(u,v,w) - F_z(A,B,C)");

# Induction Equation
PCF.add_equation("dx(A) + dy(B) + Cz =0");
PCF.add_equation("Az - dz(A) = 0"); 
PCF.add_equation("Bz - dz(B) = 0");
PCF.add_equation("Cz - dz(C) = 0");
PCF.add_equation("dt(A) - (1./Rm)*Lap(A,Az) + dx(Pi) - 1.*z*dy(B) - 1.*C - 1.*z*Cz + Flux_x = INDx(u,v,w,A,B,C)");
PCF.add_equation("dt(B) - (1./Rm)*Lap(B,Bz) + dy(Pi) + 1.*z*dx(B)                  + Flux_y = INDy(u,v,w,A,B,C)");
PCF.add_equation("dt(C) - (1./Rm)*Lap(C,Cz) + dz(Pi) + 1.*z*dx(C)                  			= INDz(u,v,w,A,B,C)");

# Zero net-flux int_v \vec{B} dV = 0
PCF.add_equation("Flux_x 			   = 0",     condition="(nx != 0) or  (ny != 0)");
PCF.add_equation("inv_Vol*integ(A,'z') = 0",     condition="(nx == 0) and (ny == 0)");

PCF.add_equation("Flux_y 			   = 0",     condition="(nx != 0) or  (ny != 0)");
PCF.add_equation("inv_Vol*integ(B,'z') = 0",     condition="(nx == 0) and (ny == 0)");

# u = 0 - No-slip 

# @ z = -1
PCF.add_bc("left(u) = 0")
PCF.add_bc("left(v) = 0");
PCF.add_bc("left(w) = 0");

# @ z = 1
PCF.add_bc("right(u) = 0");
PCF.add_bc("right(v) = 0")
PCF.add_bc("right(w) = 0",     condition="(nx != 0) or  (ny != 0)");
PCF.add_bc("integ(p,'z') = 0", condition="(nx == 0) and (ny == 0)");

# n x J = n.B = 0  - Perfectly Conducting

# @ z = -1
PCF.add_bc("left(Az) = 0");
PCF.add_bc("left(Bz) = 0");
PCF.add_bc("left(C)  = 0");

# @ z = 1
PCF.add_bc("right(Az)    = 0");
PCF.add_bc("right(Bz)    = 0");
PCF.add_bc("right(C)      = 0", condition="(nx != 0) or  (ny != 0)");
PCF.add_bc("integ(Pi,'z') = 0", condition="(nx == 0) and (ny == 0)");

# Build solver
IVP_FWD = PCF.build_solver(de.timesteppers.MCNAB2);
logger.info('Solver built')

# Initial condition t=0 is index=0, index=1 is t=T_opt = Rm/8 
IVP_FWD.load_state("InitCond_Re20Pm75_T0.125Rm_M5e-05_RandomSeed.h5",index=0); # initial condition
IVP_FWD.sim_tim = IVP_FWD.initial_sim_time = 0.
IVP_FWD.iteration = IVP_FWD.initial_iteration = 0    

# Integration parameters
IVP_FWD.stop_sim_time = np.inf; #100
IVP_FWD.stop_wall_time = np.inf; #60 * 60.
IVP_FWD.stop_iteration = N_ITERS; 

N_PRINT = N_SUB_ITERS//100; # This controls how often we print simulation data out

# CFL
CFL=flow_tools.CFL(IVP_FWD,initial_dt=dt,cadence=10,safety=0.5,max_change=1.125, min_change=0.125,max_dt=1.*dt)
CFL.add_velocities(('u', 'v', 'w'))
CFL.add_velocities(('A', 'B', 'C'))

# analysis tasks
analysis_CPT = IVP_FWD.evaluator.add_file_handler('CheckPoints', iter=N_SUB_ITERS, mode='overwrite');
analysis_CPT.add_system(IVP_FWD.state, layout='g', scales=3/2);
analysis_CPT.add_task("inv_Vol*integ( (z + u)**2 + v*v + w*w, 'z')", name='KE per k',layout='c');
analysis_CPT.add_task("inv_Vol*integ(  A*A       + B*B + C*C, 'z')", name='BE per k',layout='c');

analysis_CPT.add_task("inv_Vol*integ( (z + u)**2 + v*v + w*w, 'x')", name='KE per k yz',layout='c');
analysis_CPT.add_task("inv_Vol*integ(  A*A       + B*B + C*C, 'x')", name='BE per k yz',layout='c');

# B.2) Save-scalar data
analysis1 = IVP_FWD.evaluator.add_file_handler("scalar_data", iter=100, mode='overwrite'); # This will save all the scalar data, for every run

analysis1.add_task("inv_Vol*integ( (z + u)**2 )", name="u_x total kinetic energy"); # Add the base state V(z) to ensure <U,U> is calculated
analysis1.add_task("inv_Vol*integ( u*u )", name="u kinetic energy")
analysis1.add_task("inv_Vol*integ( v*v )", name="v kinetic energy")
analysis1.add_task("inv_Vol*integ( w*w )", name="w kinetic energy") 

analysis1.add_task("inv_Vol*integ( Avg_Bx(A)**2 )", name="B_xM0 magnetic energy"); # Useful reduction to isolate the Omega-effect
analysis1.add_task("inv_Vol*integ( A*A )", name="B_x   magnetic energy")
analysis1.add_task("inv_Vol*integ( B*B )", name="B_y magnetic energy")
analysis1.add_task("inv_Vol*integ( C*C )", name="B_z magnetic energy")

# Flow properties
flow = flow_tools.GlobalFlowProperty(IVP_FWD, cadence=N_PRINT);
flow.add_property("inv_Vol*integ( (z + u)**2 + v*v + w*w )", name='UU');
flow.add_property("inv_Vol*integ( A*A        + B*B + C*C )", name='BB');

# Zero flux, div(U), div(B)
flow.add_property("inv_Vol*integ(A + B + C)", name='B_FLUX');
flow.add_property("abs(dx(u) + dy(v) + wz )", name='divU');
flow.add_property("abs(dx(A) + dy(B) + Cz )", name='divB');

# Boundary conditions
flow.add_property("abs( interp( Az ,x='right',y='right',z='right') )",  name='< dz(Bx) = 1 >');
flow.add_property("abs( interp( Bz ,x='right',y='right',z='right') )",  name='< dz(By) = 1 >');
flow.add_property("abs( interp( C  ,x='right',y='right',z='right') )",  name='<    Bz  = 1 >');

flow.add_property("abs( interp( Az ,x='right',y='right',z='left') )",  name='< dz(Bx) = -1 >');
flow.add_property("abs( interp( Bz ,x='right',y='right',z='left') )",  name='< dz(By) = -1 >');
flow.add_property("abs( interp( C  ,x='right',y='right',z='left') )",  name='<    Bz  = -1 >');


# Main loop
end_init_time = time.time()
logger.info('Initialization time: %f' %(end_init_time-start_init_time))
try:
	logger.info('Starting loop')
	start_run_time = time.time()
	while IVP_FWD.ok:
		dt = CFL.compute_dt()
		IVP_FWD.step(dt)
		if (IVP_FWD.iteration-1) % N_PRINT == 0:
			logger.info('Iteration: %i, Time: %e, dt: %e' %(IVP_FWD.iteration, IVP_FWD.sim_time, dt));
			logger.info('Kin (1/V)<U,U> = %e,  Mag (1/V)<B,B> = %e'%( flow.volume_average('UU'),flow.volume_average('BB')  ) );

			# Flux & Divergence
			logger.info('FLUX (1/V)<B> = %e,  Max |div(U)| = %e,  Max |div(B)| = %e'%( abs(flow.volume_average('B_FLUX')), abs(flow.max('divU')),  abs(flow.max('divB')) ) );

			# BCs
			logger.info('dz(Bx) @ z= 1     = %e,  dz(By) @ z= 1      = %e, Bz @ z= 1       = %e'%( flow.max('< dz(Bx) = 1 >')     ,flow.max('< dz(By) = 1 >') ,flow.max('<    Bz  = -1 >') ) );
			logger.info('dz(Bx) @ z=-1     = %e,  dz(By) @ z=-1      = %e, Bz @ z=-1       = %e'%( flow.max('< dz(Bx) = -1 >')    ,flow.max('< dz(By) = -1 >'),flow.max('<    Bz  = -1 >') ) );

			
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_run_time = time.time()
    logger.info('Iterations: %i' %IVP_FWD.iteration)
    logger.info('Sim end time: %f' %IVP_FWD.sim_time)
    logger.info('Run time: %.2f sec' %(end_run_time-start_run_time))
    logger.info('Run time: %f cpu-hr' %((end_run_time-start_run_time)/60/60*domain.dist.comm_cart.size))

from dedalus.tools import post
post.merge_process_files("CheckPoints", cleanup=True, comm=MPI.COMM_WORLD);
post.merge_process_files("scalar_data", cleanup=True, comm=MPI.COMM_WORLD);
