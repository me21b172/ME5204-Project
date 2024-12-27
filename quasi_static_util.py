import numpy as np
import time
from multiprocessing import Manager,Pool
from worker_rect import qs_helper
import scipy
from scipy.sparse import coo_array, csr_array, csc_array
import matplotlib.pyplot as plt

def flatten(xss):
    return [x for xs in xss for x in xs] 

class quasi_static_fem:
    def __init__(self,nodecoords,elecon,centre):
        self.data_line = {"ips":{2:[-1/np.sqrt(3),1/np.sqrt(3)],3:[-np.sqrt(3/5),0,np.sqrt(3/5)]},\
        "weights":{2:[1,1],3:[5/9,8/9,5/9]}}
        self.data_tle = {"ips":{1:[[1/3,1/3]],3:[[1/6,1/6],[1/6,2/3],[2/3,1/6]]},\
       "weights":{1:[1/2],3:[1/6,1/6,1/6]}}
        self.nodecoords = nodecoords
        self.elecon = elecon
        self.centre = centre
    
    def fit_ele(self,theta_prev_time = None,theta_prev_pic = None,mode = "linear",verbose = False):
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

        K_row,K_col,K_data = [],[],[]
        G_row,G_col,G_data = [],[],[]

        #Parallel processing for matrix computations
        items = [(nodes,elei,source,theta_prev_time,theta_prev_pic,mode) for elei in ele]
        st = time.time()
        with Pool(processes = 4) as pool:
            results = pool.map(qs_helper, items)
        if verbose == True:
            print(f"Time for pooling to end {time.time()-st}")

        #Accumulating data collected over multiprocessing
        st = time.time()
        K_row, K_col, K_data, G_row, G_col, G_data, F_row,F_data, BT_row, BT_data,areas = list(zip(*results))
        mega = [K_row, K_col, K_data, G_row, G_col, G_data, F_row,F_data, BT_row, BT_data]
        K_row, K_col, K_data, G_row, G_col, G_data, F_row,F_data, BT_row, BT_data = [flatten(mini) for mini in mega]
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
        K_sparse = csc_array((K_data,(K_row,K_col)),shape=(nop,nop))
        G_sparse = csc_array((G_data,(G_row,G_col)),shape=(nop,nop))

        if verbose == True:
            print(f"Time for matrices creation {time.time()-st}")
        if mode == "no_source":
            F = np.zeros((nop,1))
            G_sparse = csc_array(np.zeros((nop,nop)))

        #Setting up the right hand side by subtracting the dirichlet terms
        T_l = 20
        non_ln = np.setdiff1d(np.arange(nop),ln)
        rhs = F+boundary_term-T_l*np.sum((K_sparse[:,ln]+G_sparse[:,ln]).toarray(),axis = 1).reshape(-1,1)

        #Sub matrix extractions
        st = time.time()
        K_sparse = K_sparse[:,non_ln][non_ln,:]
        G_sparse = G_sparse[:,non_ln][non_ln,:]
        rhs_sub = rhs[np.ix_(non_ln,[0])]
        rhs_sparse = csc_array(rhs_sub)
        if verbose == True:
            print(f"Time for sub matrix extraction {time.time()-st}")

            
        st = time.time()
        theta_sub = scipy.sparse.linalg.spsolve(K_sparse+G_sparse,rhs_sparse)
        if verbose == True:
            print(f"Time for inversion {time.time()-st}")

        #Final solution with the dirichlet imposed
        theta = np.zeros((nop,1))+T_l
        theta[non_ln,:] = theta_sub.reshape(-1,1)

        K = K_sparse
        G = G_sparse
        
        return [h,K,G,F,boundary_term,theta]

def quasi_static_picard(nodecoords,ele_con,theta_init,source,mode="non_linear"):
## the idea is you initialize a temperature profile and find corresponding multipliers for the non linear terms
    #non is number of nodes
    theta_old = theta_init

    e = 1e5
    tolerance = 1e-4
    iter = 0
    
    while(e>tolerance):
        iter +=1
        _,_,_,_,_,theta_new = quasi_static_fem(nodecoords,ele_con,source).fit_ele(theta_prev_time = theta_init,theta_prev_pic = theta_old,mode = mode)
        e = np.linalg.norm(theta_new - theta_old) 
        
        theta_old = theta_new
        print(f"Error at {iter} iteration is {e:.2E}")
        # print(theta_init)
    plt.figure(figsize=(8,4))
    plt.tricontourf(nodecoords[:,0],nodecoords[:,1],theta_new.flatten(),cmap='jet')
    plt.title(f"Source at {source[0,0],source[0,1]}")
    plt.colorbar()
    plt.show()

    return theta_new
