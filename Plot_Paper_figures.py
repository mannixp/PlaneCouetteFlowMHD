import numpy as np
import matplotlib.pyplot as plt
import h5py

##########################################################################
# Fig 1. Time-Series of the volume integrated kinetic & magnetic energies
##########################################################################

def Plot_TimeSeries_KeMe(file_name):

	"""
	Plot the volume integrated Kinetic & Magnetic energies 

	Input Parameters:

	file_name with type .h5

	Returns: None

	"""

	file = h5py.File(file_name,"r")
	#print(file['scales/'].keys()); 
	#print(file['tasks/'].keys()) # Useful commands	

	# Set the time interval we take
	index = 0; index_end = -1; 	
	time = file['scales/sim_time'][index:index_end];

	# All of these are <u,u> = (1/V)*(int_v u*u dV) where dV = rdr*ds*dz
	u2 = file['tasks/u_x total kinetic energy'][index:index_end,0,0,0];
	v2 = file['tasks/v kinetic energy'][index:index_end,0,0,0];
	w2 = file['tasks/w kinetic energy'][index:index_end,0,0,0];
	Kinetic = u2 + v2 + w2;
	
	#print(len(Kinetic))
	print('\n int_t <U^2> dt = ',np.mean(Kinetic[len(Kinetic)//2:-1]))
	

	A2 = file['tasks/B_x   magnetic energy'][index:index_end,0,0,0];
	B2 = file['tasks/B_y magnetic energy'][index:index_end,0,0,0];
	C2 = file['tasks/B_z magnetic energy'][index:index_end,0,0,0];
	Magnetic = A2 + B2 + C2;
	
	#print(len(Magnetic))
	print('int_t <B^2> dt = ',np.mean(Magnetic[len(Kinetic)//2:-1]),'\n')

	dpi = 1200;
	fig1, ax = plt.subplots(figsize=(12,6));

	ax.semilogy(time,Kinetic[0]*np.ones(len(time)) ,'b:',linewidth=1.5,label=r'$<U_0(z)^2>$');
	ax.semilogy(time,Kinetic ,'b-',linewidth=1.5,label=r'$<U^2>$');
	ax.semilogy(time,Magnetic,'k-',linewidth=1.5,label=r'$<B^2>$');

	plt.grid()
	plt.legend(fontsize=20)
	plt.xlim([min(time),max(time)])
	plt.tight_layout(pad=1, w_pad=1.5)
	fig1.savefig('Kinetic_And_Magnetic_Energy_Timeseries.pdf', dpi=dpi);
	#plt.show();

	return None;


##########################################################################
# Fig 2. Plot the reduced kinetic and magnetic spectra
# Ke(k_x,k_y) = FFT_2D(int_z |U|^2 dz), Me(k_x,k_y) = FFT_2D(int_z |B|^2 dz),
# in wave-number space (k_x, k_y).
# This requires uncommenting (BE per K & KE per k) in "QKEP_PCF_3D_MHD.py"
##########################################################################

def Plot_ReducedSpectra_KeMe(file_names,Plot_Cadence,Just_B=False):

	"""
	Plot Reduced Kinetc <U,U>(k_x,k_y) & Magentic <B,B>(k_x,k_y) spectra as 3D surfaces

	The Integrals are indexed by the waveniumbers as follows
	# k_x - axial/span-wise   wavenumber, 
	# k_y - azim/stream-wise  wavenumber, 
	# Radial/Shearwise dependancy has been integrated out

	Input Parameters:

	file_name = 'some_file.h5' with file-types .hdf5
	Plot_Cadence - int: the cadence at which to plot of the details of files contained
	Just_B - bool: If True Plots only the magnetic field components

	Returns: None

	"""
	from matplotlib import patches, cm

	# Make data.
	file = h5py.File(file_names,"r")
	#print(file['tasks/'].keys()); 
	#print(file['scales/'].keys()); 
	
	time = file['scales/sim_time'][()];

	# Get wavenumbers and create a Mesh-grid for plotting
	kx = file['scales/kx']; ky = file['scales/ky']

	Ny = int( 1 + (len(ky[:]) - 1)/2);
	inds_z = np.r_[0:Ny]; #Nz+1:len(kz[:])
	ky = ky[inds_z]; kx = kx[:];
	X, Y = np.meshgrid(kx, ky);

	##########################################################################
	# ~~~~~~~~~~~~~~~~~~~ plotting Magnetic B ~~~~~~~~~~~~~~~~~~~~~
	##########################################################################

	if Just_B == True:

		# BE(kx,ky) = FFT{ int_z B**2 dz )
		for ii in range(0,len(time),Plot_Cadence):
			B = file['tasks/BE per k'][ii,:,inds_z,0]; #(time,k_x,k_y,0)
			BE = np.log10(abs(B) + 1e-16);

			fig = plt.figure(figsize=(8,6)); 
			ax = fig.gca(projection='3d'); dpi = 400;
			plt.title("Energy <B,B>(k_x,k_y) Field Iter=i%i Time=t%i"%(k,index) );

			# Plot the surface. cmap=cm.Greys
			surf = ax.plot_surface(X, Y, BE[:,:].T,cmap=cm.Greys,linewidth=0, antialiased=False);

			# Customize the z axis.
			ax.set_zlim(-15.,2.)
			ax.set_xlabel(r'$k_x$ - Streamwise',fontsize=18);
			ax.set_ylabel(r'$k_y$ - Spanwise  ',fontsize=18);
			ax.set_zlabel(r'$log10(\hat{E}_B(k_x,k_y))$',fontsize=18);

			# Save figure
			plt.tight_layout(pad=1, w_pad=1.5)
			fig.savefig('Magnetic_Reduced_Spectra_Time=t%i.pdf'%time[ii], dpi=dpi)
			#plt.show()

	elif Just_B == False:	
		
		##########################################################################
		# ~~~~~~~~~~~~~~~ plotting Velocity Ke & Magnetic Me ~~~~~~~~~~~~~~~~~~~~~
		##########################################################################

		# BE(kx,ky) = FFT{ int_z B**2 dz }
		for ii in range(0,len(time),Plot_Cadence):
			B = file['tasks/BE per k'][ii,:,inds_z,0];
			BE = np.log10(abs(B) + 1e-16);

			fig = plt.figure(figsize=plt.figaspect(0.5))
			ax = fig.add_subplot(1, 2, 1, projection='3d'); dpi = 1200;

			plt.title(r'Energy $\hat{E}_B = <B,B>(k_x,k_y)$',fontsize=12)# Field Iter=i%i Time=t%i'%(k,index) );

			#cmaps['Perceptually Uniform Sequential'] = ['viridis', 'plasma', 'inferno', 'magma', 'cividis'];
			# Plot the surface. cmap=cm.Greys,
			surf = ax.plot_surface(X, Y, BE[:,:].T, cmap=cm.Greys,linewidth=0, antialiased=False);

			# Customize the z axis.
			#ax.set_zlim(np.min(BE),np.max(BE))
			ax.set_xlim(0,np.max(kx))
			ax.set_ylim(0,np.max(ky));

			ax.set_xlabel(r'$k_x$ - Streamwise',fontsize=12);
			ax.set_ylabel(r'$k_y$ - Spanwise  ',fontsize=12);
			#ax.set_zlabel(r'$log_{10}(\hat{E}_B(m_{\phi},k_{z}))$',fontsize=12);
			ax.view_init(30, 30)

			##########################################################################
			# ~~~~~~~~~~~~~~~~~~~ plotting Magnetic U ~~~~~~~~~~~~~~~~~~~~~
			##########################################################################

			# KE(k,m) = FFT{ int_r U**2 dr }
			U = file['tasks/KE per k'][ii,:,inds_z,0];
			KE = np.log10(abs(U) + 1e-16);

			ax = fig.add_subplot(1, 2, 2, projection='3d')
			plt.title(r'Energy $\hat{E}_U = <U,U>(k_x,k_y)$',fontsize=12)# Field Iter=i%i Time=t%i'%(k,index) );

			# Plot the surface. cmap=cm.Greys,
			surf = ax.plot_surface(X, Y, KE[:,:].T, cmap=cm.Greys,linewidth=0, antialiased=False);

			# Customize the z axis.
			#ax.set_zlim(np.min(KE),np.max(KE))
			ax.set_xlim(0,np.max(kx))
			ax.set_ylim(0,np.max(ky));

			ax.set_xlabel(r'$k_x$ - Streamwise',fontsize=12);
			ax.set_ylabel(r'$k_y$ - Spanwise  ',fontsize=12);
			#ax.set_zlabel(r'$log_{10}(\hat{E}_U(m_{\phi},k_{z}))$',fontsize=12);
			ax.view_init(30,30)

			# Save figure
			plt.tight_layout(pad=1, w_pad=1.5)
			fig.savefig('Kinetic_AND_Magnetic_Reduced_Spectra_Time=t%i.pdf'%time[ii], dpi=dpi)
			#plt.show()

	return None;


##########################################################################
# Fig 3. Plot 2D slices of the 3D vector fields 
# velocity deviation u(x,y,z,t_i) = U - V(z), 
# magnetic field 	 B(x,y,z,t_i)
##########################################################################

def Plot_Fields_U_and_B(file_name,Plot_Cadence,Just_B=False):

	"""
	Plot the Magnetic & Velocity (Optional) fields, as determined by the full MHD equations

	Input Parameters:

	file_names = 'one_file.h5' - with file-types .hdf5
	Plot_Cadence - int: the cadence at which to plot of the details of files contained
	Just_B - bool: If True Plots only the magnetic field components
	
	Returns: None
	
	"""
	file = h5py.File(file_name,"r")
	# print(file['scales/'].keys());
	# print(file['tasks/'].keys()) #useful commands

	time = file['scales/sim_time'][()];
	x = file['scales/x/1.5']; y = file['scales/y/1.5']; z = file['scales/z/1.5']
	
	# Fields array-structire (time,x,y,z)
	u   = file['tasks/u']; v   = file['tasks/v']; w   = file['tasks/w'];
	B_x = file['tasks/A']; B_y = file['tasks/B']; B_z = file['tasks/C'];
	
	SLICE = 12; # Index location of x,y,z slice to take

	for ii in range(0,len(time),Plot_Cadence):
		
		outfile_U = "".join(['U_PLOTS_Iter_Time_t%i.pdf'%time[ii] ]);	
		outfile_B = "".join(['B_PLOTS_Iter_Time_t%i.pdf'%time[ii] ]);	
		
		##########################################################################
		# ~~~~~~~~~~~~~~~~~~~ plotting Magnetic B ~~~~~~~~~~~~~~~~~~~~~
		##########################################################################

		fig = plt.figure(figsize=(8,6))
		plt.suptitle("B Field Time=t%i"%time[ii] ); dpi = 400;

		#------------------------------------ #------------------------------------
		ax1 = plt.subplot(221)
		Y,Z = np.meshgrid(y,z);
		cs = ax1.contourf(Z,Y,B_x[ii,SLICE,:,:].T,cmap='PuOr',levels=30)
		
		skip=(slice(None,None,2),slice(None,None,2))
		ax1.quiver(Z[skip],Y[skip],B_z[ii,SLICE,:,:][skip].T,B_y[ii,SLICE,:,:][skip].T,width=0.005);
		fig.colorbar(cs,ax=ax1);

		ax1.set_title(r'$B_x$, vecs - $(B_z,B_y)$');
		ax1.set_xlabel(r'Shearwise - $z$')
		ax1.set_ylabel(r'Spanwise  - $y$')

		#------------------------------------ #------------------------------------
		ax2 = plt.subplot(222)
		X,Z = np.meshgrid(x,z);
		SLICE_y = 12
		cs = ax2.contourf(Z,X,B_y[ii,:,SLICE_y,:].T,cmap='PuOr',levels=30)
		
		skip=(slice(None,None,4),slice(None,None,4))
		ax2.quiver(Z[skip],X[skip], B_z[ii,:,SLICE_y,:][skip].T,B_x[ii,:,SLICE_y,:][skip].T,width=0.005);
		fig.colorbar(cs,ax=ax2);

		ax2.set_title(r'$B_y$, vecs - ($B_z,B_x$)');
		ax2.set_xlabel(r'Shearwise  - $z$')
		ax2.set_ylabel(r'Streamwise - $x$')

		#------------------------------------ #------------------------------------
		ax3 = plt.subplot(212)
		Y,X = np.meshgrid(y,x);
		cs = ax3.contourf(X,Y,B_y[ii,:,:,SLICE],cmap='PuOr',levels=30)
		
		skip=(slice(None,None,4),slice(None,None,4))
		ax3.quiver(X[skip],Y[skip], B_x[ii,:,:,SLICE][skip],B_y[ii,:,:,SLICE][skip],width=0.005);
		fig.colorbar(cs,ax=ax3)

		ax3.set_title(r'$B_y$, vecs - ($B_x,B_y$)');
		ax3.set_xlabel(r'Streamwise - $x$');
		ax3.set_ylabel(r'Spanwise   - $y$')

		#------------------------------------ #------------------------------------
		# Save figure
		plt.tight_layout(pad=1, w_pad=1.5)
		fig.savefig(outfile_B, dpi=dpi)
		#plt.show()

		##########################################################################
		# ~~~~~~~~~~~~~~~~~~~ plotting Velocity U ~~~~~~~~~~~~~~~~~~~~~
		##########################################################################
		if Just_B == False:

			fig = plt.figure(figsize=(8,6))
			plt.suptitle("U Field Time=t%i"%time[ii] ); dpi = 400;

			#------------------------------------ #------------------------------------
			ax1 = plt.subplot(221)
			Y,Z = np.meshgrid(y,z);
			cs = ax1.contourf(Z,Y,u[ii,SLICE,:,:].T,cmap='PuOr',levels=10)

			skip=(slice(None,None,2),slice(None,None,2))
			ax1.quiver(Z[skip],Y[skip],w[ii,SLICE,:,:][skip].T,v[ii,SLICE,:,:][skip].T,width=0.005);
			fig.colorbar(cs,ax=ax1);

			ax1.set_title(r'$u$, vecs - $(w,v)$');
			ax1.set_xlabel(r'Shearwise - $z$')
			ax1.set_ylabel(r'Spanwise  - $y$')

			#------------------------------------ #------------------------------------
			ax2 = plt.subplot(222)
			X,Z = np.meshgrid(x,z);
			cs = ax2.contourf(Z,X,w[ii,:,SLICE,:].T,cmap='PuOr',levels=10)
			
			skip=(slice(None,None,4),slice(None,None,4))
			ax2.quiver(Z[skip],X[skip], w[ii,:,SLICE,:][skip].T,u[ii,:,SLICE,:][skip].T,width=0.005);
			fig.colorbar(cs,ax=ax2);

			ax2.set_title(r'$w$, vecs - ($w,u$)');
			ax2.set_xlabel(r'Shearwise  - $z$')
			ax2.set_ylabel(r'Streamwise - $x$')

			#------------------------------------ #------------------------------------
			ax3 = plt.subplot(212)
			Y,X = np.meshgrid(y,x);
			cs = ax3.contourf(X,Y,w[ii,:,:,SLICE],cmap='PuOr',levels=10)
			
			skip=(slice(None,None,4),slice(None,None,4))
			ax3.quiver(X[skip],Y[skip], u[ii,:,:,SLICE][skip],v[ii,:,:,SLICE][skip],width=0.005);
			fig.colorbar(cs,ax=ax3)#,loc='right');

			ax3.set_title(r'$w$, vecs - ($u,v$)');
			ax3.set_xlabel(r'Streamwise - $x$');
			ax3.set_ylabel(r'Spanwise   - $y$')

			#------------------------------------ #------------------------------------
			# Save figure
			plt.tight_layout(pad=1, w_pad=1.5)
			fig.savefig(outfile_U, dpi=dpi)
			#plt.show()

	return None;


if __name__ == "__main__":

	
	'''
	# Uncomment this ....
	# Incase the DNS terminates prematurely & files were left unmerged 
	from dedalus.tools import post
	from mpi4py import MPI
	Checkpoints_filenames = ['TestJ1_DAL_T8_M5e-05_Re25Pm75_Nx64Ny128Nz64_dt0.00625_2Ohmic/scalar_data/scalar_data_s1.h5','TestJ1_DAL_T8_M5e-05_Re25Pm75_Nx64Ny128Nz64_dt0.00625_2to4_Ohmic/scalar_data/scalar_data_s1.h5','TestJ1_DAL_T8_M5e-05_Re25Pm75_Nx64Ny128Nz64_dt0.00625_4to6_Ohmic/scalar_data/scalar_data_s1.h5']
	post.merge_sets("Scalar_data_Merged", Checkpoints_filenames,cleanup=False, comm=MPI.COMM_WORLD)
	#post.merge_process_files("CheckPoints", cleanup=True, comm=MPI.COMM_WORLD);
	#post.merge_process_files("scalar_data", cleanup=True, comm=MPI.COMM_WORLD);
	MPI.COMM_WORLD.Barrier()
	sys.exit()
	'''

	try:
		atmosphere_file = h5py.File('./Params.h5', 'r+')
		print(atmosphere_file.keys())

		Re = atmosphere_file['Re'][()];
		Pm = atmosphere_file['Pm'][()];
		mu = atmosphere_file['mu'][()];

		alpha = atmosphere_file['alpha'][()];
		beta = atmosphere_file['beta'][()];

		# Numerical Params
		Nx = atmosphere_file['Nx'][()];
		Ny = atmosphere_file['Ny'][()];
		Nz = atmosphere_file['Nz'][()];
		dt = atmosphere_file['dt'][()];

		print("Nx = ",Nx);
		print("Ny = ",Ny);
		print("Nz = ",Nz);
		print("dt = ",dt,"\n");

		M_0 = atmosphere_file['M_0'][()];
		E_0 = atmosphere_file['M_0'][()];
		T   = atmosphere_file['T'][()]

		atmosphere_file.close()
	except:

		pass;		
		
	##########################################################################
	# Call Fig 1, 2, 3
	##########################################################################	
	
	#Plot_TimeSeries_KeMe('scalar_data_s1.h5') 

	Plot_TimeSeries_KeMe('scalar_data/scalar_data_s1.h5') 

	Plot_Cadence = 10; # i.e. plot every 5th snapshot of those saved in CheckPoints_s1.h5
	Plot_ReducedSpectra_KeMe('CheckPoints/CheckPoints_s1.h5',Plot_Cadence);
	Plot_Fields_U_and_B('CheckPoints/CheckPoints_s1.h5',Plot_Cadence)
