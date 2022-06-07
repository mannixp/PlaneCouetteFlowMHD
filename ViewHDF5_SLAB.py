import numpy as np
from numba import jit # conda install -c numba numba 
import os,time, h5py, glob, sys
from pyevtk.hl import gridToVTK # 

#@jit(nopython=True)

def Make_VTK(V_Full,time_index,DIR_ABS_PATH,OUT_FILE):

    """ 
    # Prepares data in paraview readable format by Changing from numpy to VTS filetype

    -- Options
    time_index, int
    time_index = 0 or -1; # take first or -1 last
    
    V_Full, bool
    V_Full = True;  # Full Velocity field U = V_0 + u ;
    V_Full = False; # Just the deviation  u = U - V_0;
    

    DIR_ABS_PATH, string
    # Where to find the dedalus simulation data
    DIR_ABS_PATH = "/home/pmannix/MinSeeds_DAL_Paper/Test_DAL_Re100_Pm9_M0.20";  
    
    OUT_FILE, string
    # Name of vts file created
    OUT_FILE = "./paul_TF_Pm9_M0.20";

    """

    try:
        F1 = glob.glob(DIR_ABS_PATH + '/CheckPoints/C**');
        print(F1)
        file = h5py.File(F1[0], 'r'); 
    except:
        F1 = glob.glob(DIR_ABS_PATH + '/CheckPoints**');
        print(F1)
        file = h5py.File(F1[0], 'r');     

    #F2 = glob.glob(DIR_ABS_PATH + '/Params**');
    #print(F2)
    #file2 = h5py.File(F2[0], 'r');

    Iterations= file["scales/write_number"][()]
    sim_time = file["scales/sim_time"][()]

    x_coords = file['scales']['x']['1.5']
    y_coords = file['scales']['y']['1.5']
    z_coords = file['scales']['z']['1.5']

    x = np.array(x_coords)
    y = np.array(y_coords)
    z = np.array(z_coords)

    nx = len(x)
    ny = len(y)
    nz = len(z)

    print("nx=", nx, "ny=", ny, "nz=", nz)

    s_shape = (nx, ny, nz);
    Vx = np.zeros(s_shape); Vy = np.zeros(s_shape); Vz = np.zeros(s_shape);
    #DivU = np.zeros(s_shape);
    Bx = np.zeros(s_shape); By = np.zeros(s_shape); Bz = np.zeros(s_shape);
    #DivB = np.zeros(s_shape);
    
    Vx[:,:,:] = file['tasks/u'][time_index,:,:,:]; Vy[:,:,:] = file['tasks/v'][time_index,:,:,:]; Vz[:,:,:] = file['tasks/w'][time_index,:,:,:];
    #divU = file['tasks/divU'][time_index,:,:,:];

    Bx[:,:,:] = file['tasks/A'][time_index,:,:,:]; By[:,:,:] = file['tasks/B'][time_index,:,:,:]; Bz[:,:,:] = file['tasks/C'][time_index,:,:,:];
    #divB = file['tasks/divB'][time_index,:,:,:];


    #~~~~~ Add V_0(z) base state ~~~~~~~~~~~~~
    if V_Full == True:
    	for i in range(nx):
    		for j in range(ny):
        		Vx[i,j,:] += z;
    # ~~~~~~~~~~~~~ # ~~~~~~~~~~~~~~~~~~~~~~~~

    os.chdir(DIR_ABS_PATH)
    gridToVTK(OUT_FILE, x,y,z, pointData = {"Vx":Vx,"Vy":Vy,"Vz":Vz,  "Bx":Bx,"By":By,"Bz":Bz})

    return None;

# Execute as main
if __name__ == "__main__":

    time_index = 0; # How many iters to take
    #time_index == True; # How many iters to take
    V_Full = True; # Full Velocity field U = V_0 + u  or just the deviation u = U - V_0;

    PATH = "/workspace/pmannix/DAL_PCF_KEP_MHD/";# + "Compare_Re20Pm75_T0.125_DAL/"

    #IN_FILE = "Results_DAL_Re15Pm75_M1e-04_T0.25_Ny96"; OUT_FILE = "./IC_Pm75Re15_M1e-04";

    IN_FILE = "TestJ1_DAL_M5e-05_T8_Re20Pm75_Nx64Ny128Nz64_V1Noise_10Ohmic/"; OUT_FILE = "./DNS_M5e-05_T8_Re20Pm75";

    DIR_ABS_PATH = PATH + IN_FILE; 

    OUT_FILE_DIR = PATH;
    OUT_FILE = OUT_FILE_DIR + IN_FILE; # Don't include any extension

    #print(help(Cylindrical_to_Cart_VTK));
    #sys.exit();

    #Make_VTK(V_Full, time_index, DIR_ABS_PATH, OUT_FILE);
    #sys.exit()

    for time_index in range(100):
        if time_index%10 == 0:
            print('Time_index = ',time_index)
        
        OUT_FILE = "./DNS_M5e-05_T8_Re20Pm75" + "_t%i"%time_index;
        Make_VTK(V_Full, time_index, DIR_ABS_PATH, OUT_FILE);

    
