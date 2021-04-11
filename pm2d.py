#Import modules
import numpy as np
import sys
import matplotlib.pyplot as plt
from numba import njit, prange
import time

#============ Numba requires parallelized functions to lay outside of a class ============ 

#Use the CIC method to deposit the mass onto the grid
@njit(parallel=True, fastmath=True)
def _grid_mass_to_nodes(r, node_pos, m, Delta, m_r, n_part, Nx):

    #Find the nearest cell center using interpolation
    grid_i = (Nx - 1) * (r[:, 0] - node_pos[0, 0, 0]) / (node_pos[0, -1, 0] - node_pos[0, 0, 0])
    grid_j = (Nx - 1) * (r[:, 1] - node_pos[0, 0, 1]) / (node_pos[0, -1, 0] - node_pos[0, 0, 0])

    #Numpy only allows for integer indices, so find the nearest nodes via floor and ceil functions
    x_ceil, x_floor = np.ceil(grid_i).astype(np.int64), np.floor(grid_i).astype(np.int64)
    y_ceil, y_floor = np.ceil(grid_j).astype(np.int64), np.floor(grid_j).astype(np.int64)

    #Go through each particle...
    for i in prange(n_part):
        #...and make sure it's still on the grid
        if 0 < x_ceil[i] < Nx and 0 < y_ceil[i] < Nx and 0 < y_floor[i] < Nx and 0 < y_floor[i] < Nx:

            #CIC weights
            w_x_ceil, w_x_floor = 1.0 - np.abs((r[i, 0] - node_pos[0, x_ceil[i], 0]) / Delta), 1.0 - np.abs((r[i, 0] - node_pos[0, x_floor[i], 0]) / Delta)
            w_y_ceil, w_y_floor = 1.0 - np.abs((r[i, 1] - node_pos[y_ceil[i], 0, 1]) / Delta), 1.0 - np.abs((r[i, 1] - node_pos[y_floor[i], 0, 1]) / Delta)

            #Hard coding these four calculations speed up performance
            m_r[y_ceil[i],  x_ceil[i]]  += m[i] * w_x_ceil  *  w_y_ceil
            m_r[y_floor[i], x_floor[i]] += m[i] * w_x_floor *  w_y_floor
            m_r[y_ceil[i],  x_floor[i]] += m[i] * w_x_ceil  *  w_y_floor
            m_r[y_floor[i], x_ceil[i]]  += m[i] * w_x_floor *  w_y_ceil
            
    #Return mass-gridded nodes
    return m_r

@njit(parallel=True, fastmath=True)
#Grid the node accelerations to the masses
def _grid_acc_to_points(r, node_pos, v, Delta, a, fixed, v_c, f_d, n_part, Nx):
    #Matrix of acceleration vectors for the actual particles
    m_a = np.zeros(v.shape)

    #Same as above but need to subtract Delta/2 to get rid of self forces
    grid_i = (Nx - 1) * (r[:, 0] - node_pos[0, 0, 0] - Delta / 2) / (node_pos[0, -1, 0] - node_pos[0, 0, 0])
    grid_j = (Nx - 1) * (r[:, 1] - node_pos[0, 0, 1] - Delta / 2) / (node_pos[0, -1, 0] - node_pos[0, 0, 0])

    x_ceil, x_floor = np.ceil(grid_i).astype(np.int64), np.floor(grid_i).astype(np.int64)
    y_ceil, y_floor = np.ceil(grid_j).astype(np.int64), np.floor(grid_j).astype(np.int64)
    
    for i in prange(n_part):
        if 0 < x_ceil[i] < Nx and 0 < y_ceil[i] < Nx and 0 < y_floor[i] < Nx and 0 < y_floor[i] < Nx:
            w_x_ceil, w_x_floor = 1.0 - np.abs((r[i, 0] - node_pos[0, x_ceil[i], 0] - Delta / 2) / Delta), 1.0 - np.abs((r[i, 0] - node_pos[0, x_floor[i], 0] - Delta / 2) / Delta)
            w_y_ceil, w_y_floor = 1.0 - np.abs((r[i, 1] - node_pos[y_ceil[i], 0, 1] - Delta / 2) / Delta), 1.0 - np.abs((r[i, 1] - node_pos[y_floor[i], 0, 1] - Delta / 2) / Delta)
            #The way I have my acceleration and particle acceleration matrices setup allows me to loop over x and y
            #coordinates with the same syntax
            for j in [0, 1]:
                m_a[i, j] += a[int(y_ceil[i]),  int(x_ceil[i]),  j] * w_x_ceil *  w_y_ceil
                m_a[i, j] += a[int(y_floor[i]), int(x_floor[i]), j] * w_x_floor * w_y_floor
                
                m_a[i, j] += a[int(y_ceil[i]), int(x_floor[i]), j] * w_x_floor * w_y_ceil
                m_a[i, j] += a[int(y_floor[i]), int(x_ceil[i]), j] * w_x_ceil * w_y_floor

            #If we have a fixed halo potential, also add in at acceleration
            if fixed:
                a_c = -1.0 * (1 - f_d) * v_c**2.0 / (r[i, 0]**2.0 + r[i, 1]**2.0)
                m_a[i, 0] += a_c * r[i, 0]
                m_a[i, 1] += a_c * r[i, 1]
    #Return accelerations of the masses
    return m_a

@njit(parallel=True, fastmath=True)
def _calculate_potential(r, node_pos, Phi, Delta, fixed, v_c, f_d, grid_size, n_part):
    phi = np.zeros((n_part))
    
    #Same as above
    grid_i = (Nx - 1) * (r[:, 0] - node_pos[0, 0, 0]) / (node_pos[0, -1, 0] - node_pos[0, 0, 0])
    grid_j = (Nx - 1) * (r[:, 1] - node_pos[0, 0, 1]) / (node_pos[0, -1, 0] - node_pos[0, 0, 0])

    
    x_ceil, x_floor = np.ceil(grid_i).astype(np.int64), np.floor(grid_i).astype(np.int64)
    y_ceil, y_floor = np.ceil(grid_j).astype(np.int64), np.floor(grid_j).astype(np.int64)

    for i in prange(n_part):
        if 0 < x_ceil[i] < Nx and 0 < y_ceil[i] < Nx and 0 < y_floor[i] < Nx and 0 < y_floor[i] < Nx:
            w_x_ceil, w_x_floor = 1.0 - np.abs((r[i, 0] - node_pos[0, x_ceil[i], 0]) / Delta), 1.0 - np.abs((r[i, 0] - node_pos[0, x_floor[i], 0]) / Delta)
            w_y_ceil, w_y_floor = 1.0 - np.abs((r[i, 1] - node_pos[y_ceil[i], 0, 1]) / Delta), 1.0 - np.abs((r[i, 1] - node_pos[y_floor[i], 0, 1]) / Delta)

            #The total gravitational potential due to the masses is the sum of the weighted
            #Phi at the particle positions
            phi[i] = Phi[int(y_ceil[i]),  int(x_ceil[i])] * w_x_ceil *  w_y_ceil
            phi[i] = Phi[int(y_floor[i]), int(x_floor[i])] * w_x_floor * w_y_floor
                
            phi[i] = Phi[int(y_ceil[i]), int(x_floor[i])] * w_x_floor * w_y_ceil
            phi[i] = Phi[int(y_floor[i]), int(x_ceil[i])] * w_x_ceil * w_y_floor

    #Add in the fixed halo potential if applicable. Since a \propto 1/r, the potential is proportional to ln(r/r0),
    #where, here, r0 is the relative radius, taken to be the grid size.
    if fixed:
        return np.nansum(phi) + np.nansum((1.0 - f_d) * v_c**2.0 * np.log( np.sqrt(r[:, 0]**2.0 + r[:, 1]**2.0) / (grid_size) ))
    else:
        return np.nansum(phi)


class PM:
    #Initalize particle mesh class
    def __init__(self, input_file, integr, t, dt, eps, gridsize, N, dt_out, output):
        #Data input/output
        self.input_file  = input_file
        self.output      = output
        
        #Simulation parameters
        self.integr     = integr
        self.t_final    = t
        self.dt         = dt
        self.dt_out     = dt_out
        self.t          = 0
        self.G          = 1
        
        #Grid parameters
        self.eps        = eps
        self.N          = N
        self.gridsize   = gridsize
        self.Delta      = self.gridsize / self.N
        
        #Other necessary attributes
        self.S_hat      = None
        self.m_r        = None
        self.m_a        = None

        #Fixed acceleration
        self.fixed = True
        self.v_c   = 2 * np.pi
        self.f_d   = 0.15
        
        #Load data
        self.load_data()
        
        #Initialize node grid
        self.initialize_grid()

        #Calculate S(r) matrix
        self.calculate_S_hat()

        #Run the simulation
        self.run()
        
    #Load simulation data
    def load_data(self):
        dat = np.loadtxt(self.input_file)
        
        self.m = 1*dat[:, 1]
        self.r = 1*dat[:, 2:4]
        self.v = 1*dat[:, 4:]

    #Initialize node grid
    def initialize_grid(self):
        #Initialize the node centers for the grid
        node_points = np.linspace(-self.gridsize / 2.0 + self.Delta / 2, self.gridsize / 2.0 - self.Delta / 2, num=self.N, endpoint=True)

        #Make a 2D array of x and y nodes
        node_x, node_y = np.meshgrid(node_points, node_points)

        #NxNx2 matrix of node coordinates
        self.node_pos = np.array(list(zip(node_x.flatten(), node_y.flatten()))).reshape(self.N, self.N, 2)
        
    #Calculate \hat{S}(k)
    def calculate_S_hat(self):
        #Calculate the distance offset relative to the left edge of the simulation
        offset = np.linalg.norm(self.node_pos - self.node_pos[0, 0, :] - np.array([self.Delta/2, self.Delta/2]), axis=2)

        #Calculate the Plummer softening kernel
        S_r = -1.0 / np.sqrt(offset**2.0 + self.eps**2.0)
        
        #Now mirror S_r over x axis
        x_flip = np.vstack((S_r, S_r[::-1]))
        
        #Now mirror over y axis
        xy_flip = np.hstack((x_flip, np.fliplr(x_flip)))
        
        #Now calculate \hat{S}(k)
        self.S_hat = np.fft.rfftn(xy_flip)
        
    #Calculate \hat{m}(k)
    def calculate_m_hat(self):
        #Pad mass array with extra zeros to make the same size as S(r)
        m_padded = np.hstack((self.m_r, np.zeros(shape=(self.N, self.N))))
        m_padded = np.vstack((m_padded, np.zeros(shape=(self.N, 2*self.N))))
        
        #Calculate FFT
        self.m_hat = np.fft.rfftn(m_padded)
        
    #Calculate Phi(x)
    def calculate_Phi(self):
        # Calculate \hat{\Phi}
        Phi_hat = self.G * self.S_hat * self.m_hat
        
        #Calculate \Phi and strip off the unphysical region
        self.Phi = np.fft.irfftn(Phi_hat)[:self.N, :self.N]
        
    #Weighting function for CIC method
    def weights(self, dist, delta):
        w = 1.0 - np.abs(dist) / delta
        w[w <= 0] = 0
        return w
    
    #Interpolate mass onto grid
    def grid_mass_to_nodes(self):
        m_r = np.zeros(self.node_pos.shape[:2])
        self.m_r = _grid_mass_to_nodes(self.r, self.node_pos, self.m, self.Delta, m_r, self.m.size, self.N)

    #Calculate acceleration at nodes
    def calculate_acc_at_nodes(self):
        a_x = -np.gradient(self.Phi, self.Delta, axis=1)
        a_y = -np.gradient(self.Phi, self.Delta, axis=0)

        #NxNx2 matrix to store the accelerations at the cell centers
        self.a = np.c_[a_x.flatten(), a_y.flatten()].reshape(self.N, self.N, 2)

    #Interpolate acceleration to points
    def grid_acc_to_points(self):
        self.m_a = _grid_acc_to_points(self.r, self.node_pos, self.v, self.Delta, self.a, self.fixed, self.v_c, self.f_d, self.m.size, self.N)
        
    #Integration functions
    def leapfrog(self):
        #Calculate first half-time step
        if np.isclose(self.t, 0):
            v_half = self.v + self.m_a * self.dt / 2.0

        #Calculate other half-time steps
        else:
            v_half = self.v + self.m_a * self.dt
            
        #Update position and velocity
        self.r += v_half * self.dt
        self.v = 1.0 * v_half

    #Calculate total Phi of system
    def calculate_potential(self):
        return _calculate_potential(self.r, self.node_pos, self.Phi, self.Delta, self.fixed, self.v_c, self.f_d, self.gridsize, self.m.size)
    
    #Run the simulation
    def run(self):
        #Open dt_out diagnostic file
        global_output = open(self.output + "_global.dat", 'a')
        
        for cnt in range(int(self.t_final / self.dt)+2):
            #Grid the mass to nodes
            self.grid_mass_to_nodes()
            
            #Calculate the mass at the nodes
            self.calculate_m_hat()
            
            #Calculate the potential at the nodes
            self.calculate_Phi()
            # =========== GLOBAL DIAGNOSTICS =========== #
                            # Energy #
            _ke = np.nansum(0.5 * np.sum(self.v**2.0, axis=1))
            _u  = _calculate_potential(self.r, self.node_pos, self.Phi, self.Delta, self.fixed, self.v_c, self.f_d, self.gridsize, self.m.size) #self.calculate_potential()
           
                        # Angular momentum #
            _l = np.abs(np.sum(np.cross(self.r, self.v)))
            np.savetxt(global_output, np.c_[self.t, _u, _ke, _u + _ke , _l], delimiter=' ', newline='\n')
            # ========================================== #

            # =========== DT OUT DIAGNOSTICS =========== #
            if cnt % int(self.dt_out / self.dt) == 0:
                np.savetxt(self.output + "_parts_%g.dat" % cnt,\
                           np.c_[np.arange(len(self.m)), self.m, np.hstack((self.r, self.v))], delimiter=' ')
            # ========================================== #
            
            #Calculate acceleration at nodes
            self.calculate_acc_at_nodes()

            #Grid acceleration to original mass points
            self.grid_acc_to_points()

            #Do step
            if self.integr == "leapfrog":
                self.leapfrog()

            #Throw away points at or past the edges
            off_grid = np.where(np.greater_equal(np.abs(self.r[:, 0]), self.gridsize/2) | np.greater_equal(np.abs(self.r[:, 1]), self.gridsize/2) == True)[0]
            self.m = np.delete(self.m, off_grid , axis=0)
            self.r = np.delete(self.r, off_grid, axis=0)
            self.v = np.delete(self.v, off_grid, axis=0)
            self.m_a = np.delete(self.m_a, off_grid, axis=0)
            
            #Time step
            self.t += self.dt

        global_output.close()
        
#Read command line parameters and run the simulation
if __name__ == "__main__":
    assert len(sys.argv) == 10, "Input should be 9 parameters long"
    
    input_data, integr, output = sys.argv[1], sys.argv[2], sys.argv[-1]
    t_final, dt, eps = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
    gridsize, Nx, dt_out = float(sys.argv[6]), int(sys.argv[7]), float(sys.argv[8])
    pm_obj = PM(input_data, integr, t_final, dt, eps, gridsize, Nx, dt_out, output)
