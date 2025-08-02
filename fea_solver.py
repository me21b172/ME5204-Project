import numpy as np
import time
import scipy
from multiprocessing import Pool
from worker import nr_helper
from scipy.sparse import csc_array


def flatten(xss):
    return [x for xs in xss for x in xs]

def zeroQ(point, source, ro):
    return 0
class FEMSolver:
    def __init__(self, nodecoords, elecon, boundary_nodes, problem_params, boundary_conditions, props_chooser):
        self.data_line = {"ips": {2: [-1/np.sqrt(3), 1/np.sqrt(3)], 3: [-np.sqrt(3/5), 0, np.sqrt(3/5)]},
                          "weights": {2: [1, 1], 3: [5/9, 8/9, 5/9]}}
        self.data_tle = {"ips": {1: [[1/3, 1/3]], 3: [[1/6, 1/6], [1/6, 2/3], [2/3, 1/6]]},
                         "weights": {1: [1/2], 3: [1/6, 1/6, 1/6]}}
        self.nodecoords = nodecoords
        self.elecon = elecon
        self.boundary_nodes = boundary_nodes
        self.problem_params = problem_params
        self.boundary_conditions = boundary_conditions
        self.props_chooser = props_chooser
        if problem_params["source"]["mode"] == "laser":
            self.source_pos = problem_params["source"]["position"]
            self.ro = problem_params["source"]["ro"]
            self.Q = problem_params["source"]["Q"]
        elif problem_params["source"]["mode"]=="volumetric":
            self.source_pos = None
            self.ro = None
            self.Q = problem_params["source"]["Q"]
        elif problem_params["source"]["mode"] == "absent":
            self.source_pos = None
            self.ro = None
            self.Q = zeroQ
        else:
            raise Exception("Invalid mode of source")
        
    def solver(self, theta_prev_time=None, theta_prev2_time=None, theta_prev_nr=None,
               mode="static", dt=None, verbose=False):
        '''
        Return mass and stiffness matrices alongside the forcing vector
        '''

        nodes = self.nodecoords
        ele = self.elecon
        source_pos = self.source_pos
        props_chooser = self.props_chooser
        boundary_conditions = self.boundary_conditions
        boundary_nodes = self.boundary_nodes

        # Data for FEA
        nop = nodes.shape[0]

        # Parallel processing for matrix computations
        items = [(self.ro, self.Q, nodes, elei, source_pos, theta_prev_time, theta_prev2_time, theta_prev_nr,    
                  boundary_nodes, props_chooser, boundary_conditions) for elei in ele]

        st = time.time()
        with Pool() as pool:
            results = pool.map(nr_helper, items)

        if verbose:
            print(f"Time for pooling to end {time.time()-st}")

        # Accumulating data collected over multiprocessing
        st = time.time()
        (M_row, M_col, M_data, K_row, K_col, K_data, dMT_row, 
         dMT_col, dMT_data, dKT_row, dKT_col, dKT_data, F_row, 
         F_data, BT_row, BT_data, areas) = list(zip(*results))

        mega = [M_row, M_col, M_data, K_row, K_col, K_data, dMT_row, dMT_col, dMT_data, 
                dKT_row, dKT_col, dKT_data, F_row, F_data, BT_row, BT_data]

        flattened = [flatten(mini) for mini in mega]
        (M_row, M_col, M_data, K_row, K_col, K_data, dMT_row, dMT_col, dMT_data, dKT_row, 
        dKT_col, dKT_data, F_row, F_data, BT_row, BT_data) = flattened

        if verbose:
            print(f"Time for accumulation of data to end {time.time()-st}")

        mesh_h = np.sqrt(np.mean(areas))
        if verbose:
            print(f"Mesh size is {mesh_h} mm")

        # Preparing the matrices for calculationsnr_pipeline
        # csc array because column slicing is easy and inversion is faster than coo
        st = time.time()
        F = csc_array(
            (F_data, (F_row, [0]*len(F_row))), shape=((nop, 1))).toarray()
        boundary_term = csc_array(
            (BT_data, (BT_row, [0]*len(BT_row))), shape=((nop, 1))).toarray()

        M_sparse = csc_array((M_data, (M_row, M_col)), shape=(nop, nop))
        K_sparse = csc_array((K_data, (K_row, K_col)), shape=(nop, nop))
        dMT_sparse = csc_array(
            (dMT_data, (dMT_row, dMT_col)), shape=(nop, nop))
        dKT_sparse = csc_array(
            (dKT_data, (dKT_row, dKT_col)), shape=(nop, nop))

        if verbose:
            print(f"Time for matrices creation {time.time()-st}")

        if self.problem_params["source"]["mode"] == "absent":
            F = np.zeros((nop, 1))

        if mode == "static":
            R = F+boundary_term - K_sparse@theta_prev_nr
            dR = - dKT_sparse


            
        elif mode == "transient":
            R = F+boundary_term - K_sparse@theta_prev_nr - \
                M_sparse@(theta_prev_nr-theta_prev_time)/dt
            dR = - dKT_sparse - dMT_sparse/dt

        theta = theta_prev_nr.copy()
        
        ln = boundary_nodes["ln"]
        rn = boundary_nodes["rn"]
        bn = boundary_nodes["bn"]
        tn = boundary_nodes["tn"]
                
        dirichlet_nodes = []
        if boundary_conditions["top"]["mode"] == "const_T":
            dirichlet_nodes.extend(tn.tolist())
            theta[tn, :] = boundary_conditions["top"]["T_b"]
        if boundary_conditions["bottom"]["mode"] == "const_T":
            dirichlet_nodes.extend(bn.tolist())
            theta[bn, :] = boundary_conditions["bottom"]["T_b"]
        if boundary_conditions["left"]["mode"] == "const_T":
            dirichlet_nodes.extend(ln.tolist())
            theta[ln, :] = boundary_conditions["left"]["T_b"]
        if boundary_conditions["right"]["mode"] == "const_T":
            dirichlet_nodes.extend(rn.tolist())
            theta[rn, :] = boundary_conditions["right"]["T_b"]
            
        non_dirichlet_nodes = np.setdiff1d(np.arange(nop),np.array(dirichlet_nodes))
        # Sub matrix extractions
        st = time.time()
        R_sparse = R[non_dirichlet_nodes].reshape(-1, 1)
        dR_sparse = dR[:, non_dirichlet_nodes][non_dirichlet_nodes, :]
        # R_sparse = R.reshape(-1, 1)
        # dR_sparse = dR

        st = time.time()
        dtheta_sub = - \
            scipy.sparse.linalg.spsolve(dR_sparse, R_sparse).reshape(-1, 1)
        if verbose:
            print(f"Time for inversion {time.time()-st}")

        # Final solution with the dirichlet imposed
        theta[non_dirichlet_nodes, :] += dtheta_sub
        return theta


def nr_pipeline(nodecoords, ele_con, boundary_nodes, theta_init, problem_params, 
                boundary_conditions, props_chooser, dt=1, t_final=1, mode="transient"):

    non = nodecoords.shape[0]
    times = np.arange(0, t_final+0.9*dt, dt) #include t_final if it is exactly divisible by dt
    temperatures = np.zeros((non, times.shape[0] if mode=="transient" else 1))
    
    if problem_params["source"]["mode"] == "laser":
        laser_speed = problem_params["source"]["vo"]  # mm/s (assumed to move left)
    
    theta_prev_time = theta_init
    theta_prev2_time = None
    theta_prev_nr = theta_init

    solver_object = FEMSolver(nodecoords, ele_con, boundary_nodes, problem_params, 
                              boundary_conditions, props_chooser)
    for i, t in enumerate(times):
        e = 1e5
        tolerance = 1e-3
        iter = 0
        while (e > tolerance):
            iter += 1
            theta_cur_nr = solver_object.solver(theta_prev_time=theta_prev_time,       
                                                theta_prev2_time=theta_prev2_time,
                                                theta_prev_nr=theta_prev_nr, mode=mode, 
                                                dt=dt)

            e = np.linalg.norm(theta_cur_nr-theta_prev_nr)
            theta_prev_nr = theta_cur_nr.copy()
            print(f"Error at {iter} iteration at time {t} is {e:.2E}")

        temperatures[:, i] = theta_cur_nr.flatten()
        if mode == "static":
            break
        print(
            f" Max temperature at {t}s: {np.max(theta_cur_nr)-273} degree celcius")
        print(
            f" Min temperature at {t}s: {np.min(theta_cur_nr)-273} degree celcius")
        theta_prev2_time = theta_prev_time.copy()
        theta_prev_time = theta_prev_nr.copy()
        if problem_params["source"]["mode"] == "laser":
            solver_object.source_pos[0, 0] = solver_object.source_pos[0, 0] - laser_speed * dt  # laser left with 10 mm/s

    return temperatures
