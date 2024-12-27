import numpy as np
import time
from multiprocessing import Manager,Pool
from worker_rect import nr_helper_rect
import scipy
from scipy.sparse import coo_array, csr_array, csc_array
import matplotlib.pyplot as plt

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
    
    def solver(self,theta_prev_time = None,theta_prev_nr = None,type = "static",dt = None,mode = "linear",verbose = False):
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
        items = [(nodes,elei,source,theta_prev_time,theta_prev_nr,mode) for elei in ele]
        st = time.time()
        with Pool(processes = 4) as pool:
            results = pool.map(nr_helper_rect, items)
        if verbose == True:
            print(f"Time for pooling to end {time.time()-st}")

        #Accumulating data collected over multiprocessing
        st = time.time()
        M_row, M_col, M_data,K_row, K_col, K_data, G_row, G_col, G_data, \
        dMT_row, dMT_col, dMT_data,dKT_row, dKT_col, dKT_data, dGT_row, dGT_col, dGT_data,\
        F_row,F_data, BT_row, BT_data,areas = list(zip(*results))

        mega = [M_row, M_col, M_data,K_row, K_col, K_data, G_row, G_col, G_data, \
        dMT_row, dMT_col, dMT_data,dKT_row, dKT_col, dKT_data, dGT_row, dGT_col, dGT_data,\
        F_row,F_data, BT_row, BT_data]

        M_row, M_col, M_data,K_row, K_col, K_data, G_row, G_col, G_data, \
        dMT_row, dMT_col, dMT_data,dKT_row, dKT_col, dKT_data, dGT_row, dGT_col, dGT_data,\
        F_row,F_data, BT_row, BT_data = [flatten(mini) for mini in mega]
        if verbose == True:
            print(f"Time for accumulation of data to end {time.time()-st}")

        h = np.sqrt(np.mean(areas)) 
        if verbose == True:
            print(f"Mesh size is {h} mm")
            print(f"Spot radius is 2 mm")

        #Preparing the matrices for calculations
        # csc array because column slicing is easy and inversion is faster than coo

        st = time.time()
        F= csc_array((F_data,(F_row,[0]*len(F_row))),shape = ((nop,1))).toarray()
        boundary_term = csc_array((BT_data,(BT_row,[0]*len(BT_row))),shape = ((nop,1))).toarray()
        M_sparse = csc_array((M_data,(M_row,M_col)),shape=(nop,nop))
        K_sparse = csc_array((K_data,(K_row,K_col)),shape=(nop,nop))
        G_sparse = csc_array((G_data,(G_row,G_col)),shape=(nop,nop))
        dMT_sparse = csc_array((dMT_data,(dMT_row,dMT_col)),shape=(nop,nop))
        dKT_sparse = csc_array((dKT_data,(dKT_row,dKT_col)),shape=(nop,nop))
        dGT_sparse = csc_array((dGT_data,(dGT_row,dGT_col)),shape=(nop,nop))

        if verbose == True:
            print(f"Time for matrices creation {time.time()-st}")
        if mode == "no_source":
            F = np.zeros((nop,1))
            G_sparse = csc_array(np.zeros((nop,nop)))

        #Setting up the residuals and derivatives by subtracting the dirichlet terms
        T_l = 20
        non_ln = np.setdiff1d(np.arange(nop),ln)
        if type == "static":
            R = F+boundary_term - K_sparse@theta_prev_nr 
            dR = - dKT_sparse
        elif type == "transient":
            R = F+boundary_term + G_sparse@theta_prev_nr - K_sparse@theta_prev_nr - M_sparse@(theta_prev_nr-theta_prev_time)/dt
            dR = dGT_sparse - dKT_sparse - dMT_sparse/dt

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
      
        return [h,theta]

def nr_iterative_rect(nodecoords,ele_con,source,theta_init,dt = 1,t_final = 1,type = "transient",mode="non_linear"):
## the idea is you initialize a temperature profile and find corresponding multipliers for the non linear terms
    #non is number of nodes

    times = np.arange(0,t_final,dt)

    theta_prev_time = theta_init
    theta_prev_nr = theta_init
    for t in times:
        
        e = 1e5
        tolerance = 1e-4
        iter = 0
        
        while(e>tolerance):
            iter +=1
            h,theta_cur_nr = transient_fem(nodecoords,ele_con,source).\
                       solver(theta_prev_time = theta_prev_time,theta_prev_nr = theta_prev_nr,type = type,dt = dt,mode = mode)
            e = np.linalg.norm(theta_cur_nr - theta_prev_nr) 
            
            theta_prev_nr = theta_cur_nr
            print(f"Error at {iter} iteration at time {t} is {e:.2E}")
        
        theta_prev_time = theta_cur_nr
        # print(theta_init)
        plt.figure(figsize=(8,4))
        plt.tricontourf(nodecoords[:,0],nodecoords[:,1],theta_cur_nr.flatten(),cmap='jet')
        plt.title(f"Source at {source[0,0],source[0,1]}")
        plt.colorbar()
        plt.show()

        if type == "static":
            break
        source[0] = source[0] - 2*dt  #moving left with 2 mm/s

    return h,theta_cur_nr
