import numpy as np
import time
from multiprocessing import Manager,Pool
from worker_rect import nr_helper_rect
import scipy
from scipy.sparse import coo_array, csr_array, csc_array
import matplotlib.pyplot as plt
from gmsh_util import plot_distribution

def flatten(xss):
    return [x for xs in xss for x in xs] 

class transient_fem:
    def __init__(self,nodecoords,elecon,centre):
        self.data_line = {"ips":{2:[-1/np.sqrt(3),1/np.sqrt(3)],3:[-np.sqrt(3/5),0,np.sqrt(3/5)]},\
        "weights":{2:[1,1],3:[5/9,8/9,5/9]}}
        self.data_tle = {"ips":{1:[[1/3,1/3]],3:[[1/6,1/6],[1/6,2/3],[2/3,1/6]]},\
       "weights":{1:[1/2],3:[1/6,1/6,1/6]}}
        self.nodecoords = nodecoords
        self.elecon = elecon
        self.centre = centre
    
    def solver(self,theta_prev_time = None,theta_prev2_time=None,theta_prev_nr = None,\
               theta_phase_temp = None,type = "static",dt = None,mode = "linear",verbose = False):
        '''
        Return mass and stiffness matrices alongside the forcing vector
        '''


        nodes = self.nodecoords
        ele = self.elecon
        source = self.centre

        #getting the boundary nodes
        
        ln = np.where(nodes[:,0] == 0)[0]
        rn = np.where(nodes[:,0] == np.max(nodes[:,0]))[0]
        bn = np.where(nodes[:,1] == 0)[0]
        tn = np.where(nodes[:,1] == np.max(nodes[:,1]))[0]

        #Data for FEA
        gp = 3
        nop = nodes.shape[0]
        ips = np.array(self.data_tle["ips"][gp])
        weights = np.array(self.data_tle["weights"][gp])

        #Parallel processing for matrix computations
        items = [(nodes,elei,eleind,source,theta_prev_time,theta_prev2_time,theta_prev_nr,theta_phase_temp,mode)\
                  for eleind,elei in enumerate(ele)]
        st = time.time()
        with Pool() as pool:
            results = pool.map(nr_helper_rect, items)
        if verbose == True:
            print(f"Time for pooling to end {time.time()-st}")

        #Accumulating data collected over multiprocessing
        st = time.time()
        M_row, M_col, M_data,K_row, K_col, K_data, G_row, G_col, G_data, \
        dMT_row, dMT_col, dMT_data,dKT_row, dKT_col, dKT_data, dGT_row, dGT_col, dGT_data,\
        F_row,F_data, BT_row, BT_data,phase_ind,phase,areas = list(zip(*results))

        mega = [M_row, M_col, M_data,K_row, K_col, K_data, G_row, G_col, G_data, \
        dMT_row, dMT_col, dMT_data,dKT_row, dKT_col, dKT_data, dGT_row, dGT_col, dGT_data,\
        F_row,F_data, BT_row, BT_data,phase_ind,phase]

        M_row, M_col, M_data,K_row, K_col, K_data, G_row, G_col, G_data, \
        dMT_row, dMT_col, dMT_data,dKT_row, dKT_col, dKT_data, dGT_row, dGT_col, dGT_data,\
        F_row,F_data, BT_row, BT_data,phas_ind,phase = [flatten(mini) for mini in mega]
        if verbose == True:
            print(f"Time for accumulation of data to end {time.time()-st}")

        h = np.sqrt(np.mean(areas)) 
        if verbose == True:
            print(f"Mesh size is {h} mm")
            print(f"Spot radius is 2 mm")

        #Preparing the matrices for calculations
        # csc array because column slicing is easy and inversion is faster than coo

        st = time.time()
        phase_prev_nr = np.zeros((ele.shape[0],1))
        if mode == "phase_change":
            phase_prev_nr[phas_ind,0] = phase
        F= csc_array((F_data,(F_row,[0]*len(F_row))),shape = ((nop,1))).toarray()
        boundary_term = csc_array((BT_data,(BT_row,[0]*len(BT_row))),shape = ((nop,1))).toarray()
        M_sparse = csc_array((M_data,(M_row,M_col)),shape=(nop,nop))
        K_sparse = csc_array((K_data,(K_row,K_col)),shape=(nop,nop))
        dMT_sparse = csc_array((dMT_data,(dMT_row,dMT_col)),shape=(nop,nop))
        dKT_sparse = csc_array((dKT_data,(dKT_row,dKT_col)),shape=(nop,nop))

        if verbose == True:
            print(f"Time for matrices creation {time.time()-st}")
        if mode == "no_source":
            F = np.zeros((nop,1))

        non_ln = np.setdiff1d(np.arange(nop),ln)
        if type == "static":
            R = F+boundary_term - K_sparse@theta_prev_nr 
            dR = - dKT_sparse
        elif type == "transient":
            R = F+boundary_term - K_sparse@theta_prev_nr - M_sparse@(theta_prev_nr-theta_prev_time)/dt
            dR =  - dKT_sparse - dMT_sparse/dt

        #Sub matrix extractions
        st = time.time()
        R_sparse = R[non_ln].reshape(-1,1)
        dR_sparse = dR[:,non_ln][non_ln,:]
        if verbose == True:
            print(f"Time for sub matrix extraction {time.time()-st}")

            
        st = time.time()
        dtheta_sub = -scipy.sparse.linalg.spsolve(dR_sparse,R_sparse).reshape(-1,1)
        if verbose == True:
            print(f"Time for inversion {time.time()-st}")

        #Final solution with the dirichlet imposed
        theta = theta_prev_nr.copy()
        theta[non_ln,:] += dtheta_sub
      
        return [h,theta,phase_prev_nr]

def nr_iterative_rect(nodecoords,ele_con,source,theta_init,dt = 1,t_final = 1,type = "transient",mode="non_linear"):
## the idea is you initialize a temperature profile and find corresponding multipliers for the non linear terms
    #non is number of nodes

    non = nodecoords.shape[0]
    nel = ele_con.shape[0]
    times = np.arange(0,t_final,dt)
    temperatures = np.zeros((non,times.shape[0]))
    phases = np.zeros((nel,times.shape[0]))
    #0 is alpha, 1 is beta, 2 is liquid
    theta_prev_time = theta_init
    theta_prev2_time = None
    theta_prev_nr = theta_init
    theta_phase_temp = theta_init
    counter = 0
    for t in times:
        e = 1e5
        tolerance = 1e-3
        iter = 0
        while(e>tolerance):
            iter +=1
            h,theta_cur_nr,_ = transient_fem(nodecoords,ele_con,source).\
                       solver(theta_prev_time = theta_prev_time,theta_prev2_time=theta_prev2_time,
                              theta_prev_nr = theta_prev_nr,theta_phase_temp = theta_phase_temp\
                              ,type = type,dt = dt,mode = mode)
            #only final phase_cur_nr will be used
            e = np.linalg.norm(theta_cur_nr - theta_prev_nr) 
            
            theta_prev_nr = theta_cur_nr.copy()
            print(f"Error at {iter} iteration at time {t} is {e:.2E}")

        if type == "static":
            break

        #  #now running the same loop for phase change equilibrium

        if mode == "non_linear":
            break
        #finding the converged phase

        _,_,phase_cur = transient_fem(nodecoords,ele_con,source).\
                    solver(theta_prev_time = theta_prev_time,theta_prev2_time=theta_prev2_time,
                            theta_prev_nr = theta_prev_nr,theta_phase_temp = theta_phase_temp,\
                            type = type,dt = dt,mode = mode)
        
        temperatures[:,counter] = theta_cur_nr.flatten()
        phases[:,counter] = phase_cur.flatten()
        theta_prev2_time = theta_prev_time.copy()
        theta_prev_time = theta_prev_nr.copy()
        source[0,0] = source[0,0] - 2*dt  #moving left with 2 mm/s
        counter +=1
    return temperatures,phases
